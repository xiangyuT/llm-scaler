#pragma once

#include <sycl/sycl.hpp>
#include <torch/python.h>

// GDN Attention: Causal Conv1d + Gated Delta Rule
//
// Extracted from vllm-xpu-kernels (non-XE2 native path).
// Two-phase kernel:
//   Phase 1: causal_conv1d — splits projected_states_qkvz into q,k,v,z,
//            applies conv1d with activation, updates conv_state cache.
//   Phase 2: gated_delta_rule — SSM recurrence with gated delta update,
//            produces core_attn_out.

namespace gdn {

// ---- Shared types ----

enum class ActMode {
  silu = 0,
  swish = 1,
};

static constexpr int sub_group_size = 32;

// ===========================================================================
// Phase 1: Causal Conv1d
// ===========================================================================

template <typename T, int Width, bool ReorderInput>
struct causal_conv1d_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  causal_conv1d_kernel(
      T* q_out,
      T* k_out,
      T* v_out,
      T* z_out,
      T* b_out,
      T* a_out,
      const T* mixed_qkvz,
      const T* mixed_ba,
      const T* conv_weights,
      const T* conv_bias,
      T* conv_states,
      const int conv_states_stride_0,
      T* conv_states_tmp,
      int* query_start_loc,
      int* cache_indices,
      bool* has_initial_state,
      const ActMode& act_mode,
      const int& pad_slot_id,
      const int& batch_size,
      const int& num_actual_tokens,
      const int& num_k_heads,
      const int& head_k_dim,
      const int& num_v_heads,
      const int& head_v_dim,
      const int& qkvz_elems,
      const int& conv_elems)
      : q_out(q_out),
        k_out(k_out),
        v_out(v_out),
        z_out(z_out),
        b_out(b_out),
        a_out(a_out),
        mixed_qkvz(mixed_qkvz),
        mixed_ba(mixed_ba),
        conv_weights(conv_weights),
        conv_bias(conv_bias),
        conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_states_tmp(conv_states_tmp),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        act_mode(act_mode),
        pad_slot_id(pad_slot_id),
        batch_size(batch_size),
        num_actual_tokens(num_actual_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim),
        qkvz_elems(qkvz_elems),
        conv_elems(conv_elems) {}

  static inline sycl::nd_range<2>
  get_nd_range(const int total_seqlen, const int qkvz_elems) {
    const int groups_per_token =
        (qkvz_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<2> local(1, group_size);
    sycl::range<2> global(total_seqlen, groups_per_token);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline void act_swish(float& x, float beta = 1.0f) {
    x = x / (1.0f + sycl::exp(-x * beta));
  }
  static inline void act_silu(float& x) { act_swish(x, 1.0f); }

  void operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int local_group_id = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int qkvz_elems_id =
        local_group_id * elems_per_group + local_id * elems_per_item;

    if (qkvz_elems_id >= qkvz_elems) {
      return;
    }

    const int q_dim = head_k_dim;
    const int k_dim = head_k_dim;
    const int v_dim = head_v_dim * num_v_heads / num_k_heads;
    const int z_dim = head_v_dim * num_v_heads / num_k_heads;
    const int qkvz_dim = q_dim + k_dim + v_dim + z_dim;

    int k_heads_id = qkvz_elems_id / qkvz_dim;
    int qkvz_dim_id = qkvz_elems_id % qkvz_dim;

    // reorder b,a
    if constexpr (ReorderInput) {
      if (qkvz_elems_id < num_v_heads) {
        int step = token_id * num_v_heads;
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          b_out[step + qkvz_elems_id + e] =
              mixed_ba[step * 2 + qkvz_elems_id + e];
          a_out[step + qkvz_elems_id + e] =
              mixed_ba[step * 2 + num_v_heads + qkvz_dim_id + e];
        }
      }
    } else {
      if (qkvz_dim_id < (num_v_heads / num_k_heads)) {
        int step =
            token_id * num_v_heads + k_heads_id * num_v_heads / num_k_heads;
        const int ba_elems_per_item =
            sycl::min(elems_per_item, num_v_heads / num_k_heads);
#pragma unroll
        for (int e = 0; e < ba_elems_per_item; ++e) {
          b_out[step + qkvz_dim_id + e] = mixed_ba[step * 2 + qkvz_dim_id + e];
          a_out[step + qkvz_dim_id + e] =
              mixed_ba[step * 2 + num_v_heads / num_k_heads + qkvz_dim_id + e];
        }
      }
    }

    // get current seq start, end
    int batch_id = batch_size - 1;
    int seq_start_offset = 0;
    int seq_end_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
      if (token_id < query_start_loc[i + 1]) {
        batch_id = i;
        seq_start_offset = query_start_loc[i];
        seq_end_offset = query_start_loc[i + 1];
        break;
      }
    }

    // get states cache location
    int states_id = cache_indices[batch_id];

    if (states_id == pad_slot_id) {
      return;
    }

    int mixed_qkvz_id = qkvz_elems_id;

    bool is_q = false;
    bool is_k = false;
    bool is_v = false;
    bool is_z = false;

    if (qkvz_dim_id < q_dim) {
      is_q = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = k_heads_id * k_dim + qkvz_dim_id;
      }
    } else if (qkvz_dim_id < q_dim + k_dim) {
      is_k = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = num_k_heads * head_k_dim + k_heads_id * k_dim +
                        qkvz_dim_id - (q_dim);
      }
    } else if (qkvz_dim_id < q_dim + k_dim + v_dim) {
      is_v = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim + k_heads_id * v_dim +
                        qkvz_dim_id - (q_dim + k_dim);
      }
    } else {
      is_z = true;
      if constexpr (ReorderInput) {
        mixed_qkvz_id = 2 * num_k_heads * head_k_dim +
                        num_v_heads * head_v_dim + k_heads_id * z_dim +
                        qkvz_dim_id - (q_dim + k_dim + v_dim);
      }
    }

    // reorder z
    if (is_z) {
      int z_elems_id =
          k_heads_id * z_dim + qkvz_dim_id - (q_dim + k_dim + v_dim);
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        z_out[token_id * num_k_heads * z_dim + z_elems_id + e] =
            mixed_qkvz[token_id * qkvz_elems + mixed_qkvz_id + e];
      }
      return;
    }

    // reorder index to map weights
    int reordered_elems_id = 0;
    if (is_q) {
      reordered_elems_id = k_heads_id * q_dim + qkvz_dim_id;
    } else if (is_k) {
      reordered_elems_id =
          num_k_heads * q_dim + k_heads_id * k_dim + qkvz_dim_id - q_dim;
    } else if (is_v) {
      reordered_elems_id = num_k_heads * (q_dim + k_dim) + k_heads_id * v_dim +
                           qkvz_dim_id - (q_dim + k_dim);
    }

    // get states cache ptr
    const bool has_init_conv_states =
        (has_initial_state == nullptr ||
         (has_initial_state != nullptr && has_initial_state[batch_id]));
    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;

    // load weights
    T local_weights[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_weights[Width * e + i] =
            conv_weights[(reordered_elems_id + e) * Width + i];
      }
    }

    // load input
    T local_input[Width * elems_per_item];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + i] = 0.0f;
      }
    }

    int seq_cu_len = token_id - seq_start_offset + 1;
    int input_load_len = seq_cu_len >= Width ? Width : seq_cu_len;
    int states_load_len = seq_cu_len >= Width ? 0 : Width - input_load_len;
    if (states_load_len != 0 && has_init_conv_states) {
#pragma unroll
      for (int i = 0; i < states_load_len; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          local_input[Width * e + i] = conv_states_ptr
              [(Width - 1 - states_load_len + i) * conv_elems +
               reordered_elems_id + e];
        }
      }
    }

#pragma unroll
    for (int i = 0; i < input_load_len; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        local_input[Width * e + states_load_len + i] = mixed_qkvz
            [(token_id - input_load_len + 1 + i) * qkvz_elems + mixed_qkvz_id +
             e];
      }
    }

    float res[elems_per_item];
#pragma unroll
    for (int i = 0; i < elems_per_item; ++i) {
      res[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < Width; ++i) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] += static_cast<float>(local_input[Width * e + i]) *
                  static_cast<float>(local_weights[Width * e + i]);
      }
    }

    if (conv_bias != nullptr) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        res[e] += conv_bias[reordered_elems_id + e];
      }
    }

    // save states
    if (seq_end_offset - seq_start_offset > 1) {
      // because current group is unable to know if old states are needed by
      // other group, hard to update states inplace if prefill
      if (seq_end_offset - 1 == token_id) {
#pragma unroll
        for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
          for (int e = 0; e < elems_per_item; ++e) {
            conv_states_tmp
                [batch_id * (Width - 1) * conv_elems + i * conv_elems +
                 reordered_elems_id + e] = local_input[Width * e + i + 1];
          }
        }
      }
    } else {
// update states inplace if decode
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
#pragma unroll
        for (int e = 0; e < elems_per_item; ++e) {
          conv_states_ptr[i * conv_elems + reordered_elems_id + e] =
              local_input[Width * e + i + 1];
        }
      }
    }

    if (act_mode == ActMode::silu) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        act_silu(res[e]);
      }
    } else if (act_mode == ActMode::swish) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        act_swish(res[e]);
      }
    }

    // reorder q, k, v
    if (is_q) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        q_out
            [token_id * num_k_heads * q_dim + k_heads_id * q_dim + qkvz_dim_id +
             e] = res[e];
      }
    } else if (is_k) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        k_out
            [token_id * num_k_heads * k_dim + k_heads_id * k_dim + qkvz_dim_id -
             q_dim + e] = res[e];
      }
    } else if (is_v) {
#pragma unroll
      for (int e = 0; e < elems_per_item; ++e) {
        v_out
            [token_id * num_k_heads * v_dim + k_heads_id * v_dim + qkvz_dim_id -
             (q_dim + k_dim) + e] = res[e];
      }
    }
  }

 private:
  T* q_out;
  T* k_out;
  T* v_out;
  T* z_out;
  T* b_out;
  T* a_out;
  const T* mixed_qkvz;
  const T* mixed_ba;
  const T* conv_weights;
  const T* conv_bias;
  T* conv_states;
  const int conv_states_stride_0;
  T* conv_states_tmp;
  const int32_t* query_start_loc;
  const int* cache_indices;
  const bool* has_initial_state;
  const ActMode act_mode;
  const int pad_slot_id;
  const int batch_size;
  const int num_actual_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int qkvz_elems;
  const int conv_elems;
};

template <typename T>
struct update_states_kernel {
 public:
  static constexpr int sub_group_size = 32;
  static constexpr int group_size = 256;
  static constexpr int elems_per_item = 4;
  static constexpr int elems_per_group = group_size * elems_per_item;

  update_states_kernel(
      T* conv_states,
      const int conv_states_stride_0,
      const T* conv_states_tmp,
      const int* cache_indices,
      const int width,
      const int conv_elems,
      const int32_t* query_start_loc,
      const int batch_size)
      : conv_states(conv_states),
        conv_states_stride_0(conv_states_stride_0),
        conv_states_tmp(conv_states_tmp),
        cache_indices(cache_indices),
        width(width),
        conv_elems(conv_elems),
        query_start_loc(query_start_loc),
        batch_size(batch_size) {}

  static inline sycl::nd_range<3>
  get_nd_range(const int batch_size, const int width, const int conv_elems) {
    const int groups_per_token =
        (conv_elems + elems_per_group - 1) / elems_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, (width - 1), groups_per_token);
    return sycl::nd_range<3>(global * local, local);
  }

  void operator()(sycl::nd_item<3> item) const {
    const int batch_id = item.get_group(0);
    const int width_id = item.get_group(1);
    const int local_group_id = item.get_group(2);
    const int local_id = item.get_local_linear_id();
    const int elems_start_offset_group = local_group_id * elems_per_group;

    int seq_start_offset = query_start_loc[batch_id];
    int seq_end_offset = query_start_loc[batch_id + 1];
    if (seq_end_offset - seq_start_offset == 1) {
      // only update if prefill
      return;
    }

    int states_id = cache_indices[batch_id];
    T* conv_states_ptr = conv_states + states_id * conv_states_stride_0;
    const T* conv_states_tmp_ptr =
        conv_states_tmp + batch_id * (width - 1) * conv_elems;
    for (int i = elems_start_offset_group + local_id;
         i < (local_group_id + 1) * elems_per_group;
         i += group_size) {
      conv_states_ptr[width_id * conv_elems + i] =
          conv_states_tmp_ptr[width_id * conv_elems + i];
    }
  }

 private:
  T* conv_states;
  const int conv_states_stride_0;
  const T* conv_states_tmp;
  const int* cache_indices;
  const int width;
  const int conv_elems;
  const int32_t* query_start_loc;
  const int batch_size;
};

template <typename T, int Width, bool ReorderInput>
void conv1d_kernel_launcher(
    sycl::queue& queue,
    T* q_out,
    T* k_out,
    T* v_out,
    T* z_out,
    T* b_out,
    T* a_out,
    const T* mixed_qkvz,
    const T* mixed_ba,
    const T* conv_weights,
    const T* conv_bias,
    T* conv_states,
    const int conv_states_stride_0,
    T* conv_states_tmp,
    int* query_start_loc,
    int* cache_indices,
    bool* has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int& batch_size,
    const int& num_actual_tokens,
    const int& num_k_heads,
    const int& head_k_dim,
    const int& num_v_heads,
    const int& head_v_dim,
    const int& qkvz_elems,
    const int& conv_elems,
    const int& num_prefills,
    const int& num_decodes) {
  using KERNEL_MAIN = causal_conv1d_kernel<T, Width, ReorderInput>;
  auto range_main = KERNEL_MAIN::get_nd_range(num_actual_tokens, qkvz_elems);
  assert(head_k_dim % KERNEL_MAIN::elems_per_item == 0);
  assert(num_v_heads % KERNEL_MAIN::elems_per_item == 0);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL_MAIN task(
        q_out,
        k_out,
        v_out,
        z_out,
        b_out,
        a_out,
        mixed_qkvz,
        mixed_ba,
        conv_weights,
        conv_bias,
        conv_states,
        conv_states_stride_0,
        conv_states_tmp,
        query_start_loc,
        cache_indices,
        has_initial_state,
        act_mode,
        pad_slot_id,
        batch_size,
        num_actual_tokens,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        qkvz_elems,
        conv_elems);
    cgh.parallel_for(range_main, task);
  });
  if (num_prefills > 0) {
    using KERNEL_UPDATE = update_states_kernel<T>;
    auto range_update =
        KERNEL_UPDATE::get_nd_range(batch_size, Width, conv_elems);
    queue.submit([&](sycl::handler& cgh) {
      KERNEL_UPDATE task(
          conv_states,
          conv_states_stride_0,
          conv_states_tmp,
          cache_indices,
          Width,
          conv_elems,
          query_start_loc,
          batch_size);
      cgh.parallel_for(range_update, task);
    });
  }
}

inline void causal_conv1d(
    sycl::queue& queue,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    torch::Tensor& z_out,
    torch::Tensor& b_out,
    torch::Tensor& a_out,
    const torch::Tensor& mixed_qkvz,
    const torch::Tensor& mixed_ba,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const ActMode& act_mode,
    const int& pad_slot_id,
    const int num_prefills,
    const int num_decodes,
    const bool reorder_input) {
  if (num_prefills == 0 && num_decodes == 0) {
    return;
  }

  const int batch_size = query_start_loc.size(0) - 1;
  const int num_actual_tokens = q_out.size(0);
  const int num_k_heads = q_out.size(1);
  const int head_k_dim = q_out.size(2);
  const int num_v_heads = v_out.size(1);
  const int head_v_dim = v_out.size(2);
  const int qkvz_elems = mixed_qkvz.size(1);
  const int conv_elems = conv_weights.size(0);
  const int width = conv_weights.size(1);
  const int conv_states_stride_0 = conv_states.stride(0);

  auto dtype = conv_states.dtype();
  auto device = conv_states.device();
  torch::Tensor conv_states_tmp = torch::empty(
      {batch_size, width - 1, conv_elems},
      torch::dtype(dtype).device(device).requires_grad(false));

#define GDN_CONV1D_KERNEL_LAUNCHER(scalar_t, width, reorder_input)  \
  conv1d_kernel_launcher<scalar_t, width, reorder_input>(           \
      queue,                                                        \
      reinterpret_cast<scalar_t*>(q_out.data_ptr()),                \
      reinterpret_cast<scalar_t*>(k_out.data_ptr()),                \
      reinterpret_cast<scalar_t*>(v_out.data_ptr()),                \
      reinterpret_cast<scalar_t*>(z_out.data_ptr()),                \
      reinterpret_cast<scalar_t*>(b_out.data_ptr()),                \
      reinterpret_cast<scalar_t*>(a_out.data_ptr()),                \
      reinterpret_cast<scalar_t*>(mixed_qkvz.data_ptr()),           \
      reinterpret_cast<scalar_t*>(mixed_ba.data_ptr()),             \
      reinterpret_cast<scalar_t*>(conv_weights.data_ptr()),         \
      conv_bias.has_value()                                         \
          ? reinterpret_cast<scalar_t*>(conv_bias->data_ptr())      \
          : nullptr,                                                \
      reinterpret_cast<scalar_t*>(conv_states.data_ptr()),          \
      conv_states_stride_0,                                         \
      reinterpret_cast<scalar_t*>(conv_states_tmp.data_ptr()),      \
      reinterpret_cast<int*>(query_start_loc.data_ptr()),           \
      reinterpret_cast<int*>(cache_indices.data_ptr()),             \
      has_initial_state.has_value()                                 \
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr())  \
          : nullptr,                                                \
      act_mode,                                                     \
      pad_slot_id,                                                  \
      batch_size,                                                   \
      num_actual_tokens,                                            \
      num_k_heads,                                                  \
      head_k_dim,                                                   \
      num_v_heads,                                                  \
      head_v_dim,                                                   \
      qkvz_elems,                                                   \
      conv_elems,                                                   \
      num_prefills,                                                 \
      num_decodes);

#define GDN_CONV1D_WIDTH_DISPATCH(scalar_t, width, reorder_input) \
  switch (width) {                                                \
    case 1:                                                       \
      GDN_CONV1D_KERNEL_LAUNCHER(scalar_t, 1, reorder_input)     \
      break;                                                      \
    case 2:                                                       \
      GDN_CONV1D_KERNEL_LAUNCHER(scalar_t, 2, reorder_input)     \
      break;                                                      \
    case 3:                                                       \
      GDN_CONV1D_KERNEL_LAUNCHER(scalar_t, 3, reorder_input)     \
      break;                                                      \
    case 4:                                                       \
      GDN_CONV1D_KERNEL_LAUNCHER(scalar_t, 4, reorder_input)     \
      break;                                                      \
    case 5:                                                       \
      GDN_CONV1D_KERNEL_LAUNCHER(scalar_t, 5, reorder_input)     \
      break;                                                      \
    default:                                                      \
      break;                                                      \
  }

#define GDN_CONV1D_SPLIT_DISPATCH(scalar_t, width, reorder_input) \
  if (reorder_input) {                                            \
    GDN_CONV1D_WIDTH_DISPATCH(scalar_t, width, true)              \
  } else {                                                        \
    GDN_CONV1D_WIDTH_DISPATCH(scalar_t, width, false)             \
  }

  if (mixed_qkvz.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    GDN_CONV1D_SPLIT_DISPATCH(scalar_t, width, reorder_input)
  } else if (mixed_qkvz.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    GDN_CONV1D_SPLIT_DISPATCH(scalar_t, width, reorder_input)
  } else {
    using scalar_t = float;
    GDN_CONV1D_SPLIT_DISPATCH(scalar_t, width, reorder_input)
  }
#undef GDN_CONV1D_SPLIT_DISPATCH
#undef GDN_CONV1D_WIDTH_DISPATCH
#undef GDN_CONV1D_KERNEL_LAUNCHER
}

// ===========================================================================
// Phase 2: Gated Delta Rule
// ===========================================================================

template <typename T, int k_bucket_size>
struct gated_delta_rule_kernel {
 public:
  static constexpr int group_size = 256;
  static constexpr int sg_per_group = group_size / sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001;

  gated_delta_rule_kernel(
      T* core_attn_out,
      const T* q,
      const T* k,
      const T* v,
      const T* b,
      const T* a,
      const T* A_log,
      const T* dt_bias,
      T* ssm_state,
      const int ssm_state_stride_0,
      const int* query_start_loc,
      const int* cache_indices,
      const bool* has_initial_state,
      const int batch_size,
      const int total_seqlen,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim)
      : core_attn_out(core_attn_out),
        q(q),
        k(k),
        v(v),
        b(b),
        a(a),
        A_log(A_log),
        dt_bias(dt_bias),
        ssm_state(ssm_state),
        ssm_state_stride_0(ssm_state_stride_0),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        batch_size(batch_size),
        total_seqlen(total_seqlen),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int batch_size, const int num_v_heads, const int head_v_dim) {
    int num_v_bucket = (head_v_dim + v_dim_per_group - 1) / v_dim_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, num_v_heads, num_v_bucket);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  static inline float
  act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
    if (beta * x < threshold) {
      return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
    } else
      return x;
  }

  [[intel::reqd_sub_group_size(sub_group_size)]]
  void operator()(sycl::nd_item<3> item) const {
    int batch_id = item.get_group(0);
    int num_v_heads_id = item.get_group(1);
    int v_bucket_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    // assume num_v_heads is always bigger than num_k_heads
    int kv_ratio = num_v_heads / num_k_heads;
    int head_v_dim_id = v_bucket_id * v_dim_per_group + sg_id * v_dim_per_sg;

    if (head_v_dim_id >= head_v_dim) {
      return;
    }

    const float scale = 1.0f / sycl::sqrt(float(head_k_dim));
    float A_log_local = A_log[num_v_heads_id];
    float dt_bias_local = dt_bias[num_v_heads_id];
    A_log_local = -sycl::exp(A_log_local);

    float state_local[v_dim_per_sg * k_bucket_size];
    float q_local[k_bucket_size];
    float k_local[k_bucket_size];
    float v_local[v_dim_per_sg];

    T* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[batch_id]) * ssm_state_stride_0;

    // load state
    if (has_initial_state == nullptr || has_initial_state[batch_id]) {
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] = ssm_state_ptr
              [num_v_heads_id * head_k_dim * head_v_dim +
               (k_bucket_size * sg_local_id + i) +
               (head_v_dim_id + j) * head_k_dim];
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
#pragma unroll
        for (int j = 0; j < v_dim_per_sg; ++j) {
          state_local[i * v_dim_per_sg + j] = 0.0f;
        }
      }
    }

    int seq_start_offset = query_start_loc[batch_id];
    int seq_end_offset = query_start_loc[batch_id + 1];

    // The state of each token is calculated iteratively.
    for (int t = seq_start_offset; t < seq_end_offset; ++t) {
      // act beta(t), g(t)
      float b_local = b[t * num_v_heads + num_v_heads_id];
      float beta = act_sigmoid(b_local);
      float a_local = a[t * num_v_heads + num_v_heads_id] + dt_bias_local;
      float g = sycl::exp(A_log_local * act_softplus(a_local));

      float q_sum = 0.0f;
      float k_sum = 0.0f;
// load q(t), k(t) and l2norm
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] =
            q[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        k_local[i] =
            k[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        q_sum += q_local[i] * q_local[i];
        k_sum += k_local[i] * k_local[i];
      }
      q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
      k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
      q_sum += eps;
      k_sum += eps;
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] /= sycl::sqrt(q_sum);
        q_local[i] *= scale;
        k_local[i] /= sycl::sqrt(k_sum);
      }

      float kv_mem[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = 0.0f;
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] *= g;
          kv_mem[j] += state_local[j * k_bucket_size + i] * k_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = sycl::reduce_over_group(sg, kv_mem[i], sycl::plus<>());
      }

#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        v_local[i] =
            v[t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim +
              head_v_dim_id + i];
      }
      float delta[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        delta[i] = (v_local[i] - kv_mem[i]) * beta;
      }

      float res[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          // get S(t)
          state_local[j * k_bucket_size + i] += k_local[i] * delta[j];
          // get O(t)
          res[j] += state_local[j * k_bucket_size + i] * q_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = sycl::reduce_over_group(sg, res[i], sycl::plus<>());
      }

      // store O(t)
      if (sg_local_id == 0) {
#pragma unroll
        for (int i = 0; i < v_dim_per_sg; ++i) {
          core_attn_out
              [t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim +
               head_v_dim_id + i] = res[i];
        }
      }
    }

// update state
#pragma unroll
    for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        ssm_state_ptr
            [num_v_heads_id * head_k_dim * head_v_dim +
             (k_bucket_size * sg_local_id + i) +
             (head_v_dim_id + j) * head_k_dim] =
                state_local[j * k_bucket_size + i];
      }
    }
  }

 private:
  T* core_attn_out;
  const T* q;
  const T* k;
  const T* v;
  const T* b;
  const T* a;
  const T* A_log;
  const T* dt_bias;
  T* ssm_state;
  const int ssm_state_stride_0;
  const int* query_start_loc;
  const int* cache_indices;
  const bool* has_initial_state;
  const int batch_size;
  const int total_seqlen;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};

template <typename T, int k_bucket_size>
void gdr_kernel_launcher(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    const T* b,
    const T* a,
    const T* A_log,
    const T* dt_bias,
    T* ssm_state,
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int batch_size,
    const int total_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using KERNEL = gated_delta_rule_kernel<T, k_bucket_size>;
  auto range = KERNEL::get_nd_range(batch_size, num_v_heads, head_v_dim);
  assert(head_v_dim % KERNEL::v_dim_per_group == 0);
  queue.submit([&](sycl::handler& cgh) {
    KERNEL task(
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_stride_0,
        query_start_loc,
        cache_indices,
        has_initial_state,
        batch_size,
        total_seqlen,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim);
    cgh.parallel_for(range, task);
  });
}

inline void gated_delta_rule(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const int num_prefills,
    const int num_decodes) {
  if (num_prefills == 0 && num_decodes == 0) {
    return;
  }

  int batch_size = query_start_loc.size(0) - 1;
  if (num_prefills == 0 && num_decodes > 0) {
    batch_size = num_decodes;
  }
  const int total_seqlen = q.size(0);
  const int num_k_heads = q.size(1);
  const int head_k_dim = q.size(2);
  const int num_v_heads = v.size(1);
  const int head_v_dim = v.size(2);
  const int ssm_state_stride_0 = ssm_state.stride(0);

  TORCH_CHECK(num_v_heads % num_k_heads == 0);
  TORCH_CHECK(head_k_dim % sub_group_size == 0);
  const int k_bucket_size = head_k_dim / sub_group_size;

#define GDN_GDR_KERNEL_LAUNCHER(scalar_t, k_bucket_size)            \
  gdr_kernel_launcher<scalar_t, k_bucket_size>(                     \
      queue,                                                        \
      reinterpret_cast<scalar_t*>(core_attn_out.data_ptr()),        \
      reinterpret_cast<scalar_t*>(q.data_ptr()),                    \
      reinterpret_cast<scalar_t*>(k.data_ptr()),                    \
      reinterpret_cast<scalar_t*>(v.data_ptr()),                    \
      reinterpret_cast<scalar_t*>(b.data_ptr()),                    \
      reinterpret_cast<scalar_t*>(a.data_ptr()),                    \
      reinterpret_cast<scalar_t*>(A_log.data_ptr()),                \
      reinterpret_cast<scalar_t*>(dt_bias.data_ptr()),              \
      reinterpret_cast<scalar_t*>(ssm_state.data_ptr()),            \
      ssm_state_stride_0,                                           \
      reinterpret_cast<int*>(query_start_loc.data_ptr()),           \
      reinterpret_cast<int*>(cache_indices.data_ptr()),             \
      has_initial_state.has_value()                                 \
          ? reinterpret_cast<bool*>(has_initial_state->data_ptr())  \
          : nullptr,                                                \
      batch_size,                                                   \
      total_seqlen,                                                 \
      num_k_heads,                                                  \
      head_k_dim,                                                   \
      num_v_heads,                                                  \
      head_v_dim);

#define GDN_GDR_BUCKET_DISPATCH(scalar_t, k_bucket_size) \
  switch (k_bucket_size) {                               \
    case 1:                                              \
      GDN_GDR_KERNEL_LAUNCHER(scalar_t, 1)               \
      break;                                             \
    case 2:                                              \
      GDN_GDR_KERNEL_LAUNCHER(scalar_t, 2)               \
      break;                                             \
    case 4:                                              \
      GDN_GDR_KERNEL_LAUNCHER(scalar_t, 4)               \
      break;                                             \
    case 8:                                              \
      GDN_GDR_KERNEL_LAUNCHER(scalar_t, 8)               \
      break;                                             \
    default:                                             \
      TORCH_CHECK(false);                                \
  }

  if (core_attn_out.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    GDN_GDR_BUCKET_DISPATCH(scalar_t, k_bucket_size)
  } else if (core_attn_out.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    GDN_GDR_BUCKET_DISPATCH(scalar_t, k_bucket_size)
  } else {
    using scalar_t = float;
    GDN_GDR_BUCKET_DISPATCH(scalar_t, k_bucket_size)
  }
#undef GDN_GDR_BUCKET_DISPATCH
#undef GDN_GDR_KERNEL_LAUNCHER
}

}  // namespace gdn

// ===========================================================================
// Host dispatcher
// ===========================================================================

inline void gdn_attention_host(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    int64_t num_prefills,
    int64_t num_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const torch::Tensor& non_spec_state_indices_tensor,
    int64_t num_actual_tokens,
    int64_t tp_size,
    bool reorder_input,
    sycl::queue& queue)
{
    gdn::ActMode act_mode;
    if (activation == "silu") {
        act_mode = gdn::ActMode::silu;
    } else if (activation == "swish") {
        act_mode = gdn::ActMode::swish;
    } else {
        TORCH_CHECK(false, "gdn_attention: unsupported activation '", activation, "'");
    }

    const int pad_slot_id = -1;
    auto dtype = projected_states_qkvz.dtype();
    auto device = projected_states_qkvz.device();

    torch::Tensor q = torch::empty(
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor k = torch::empty(
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor v = torch::empty(
        {num_actual_tokens, num_v_heads / tp_size, head_v_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor b = torch::empty(
        {num_actual_tokens, num_v_heads / tp_size},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor a = torch::empty(
        {num_actual_tokens, num_v_heads / tp_size},
        torch::dtype(dtype).device(device).requires_grad(false));

    gdn::causal_conv1d(
        queue,
        q, k, v, z, b, a,
        projected_states_qkvz,
        projected_states_ba,
        conv_weights,
        conv_bias,
        conv_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        act_mode,
        pad_slot_id,
        (int)num_prefills,
        (int)num_decodes,
        reorder_input);

    gdn::gated_delta_rule(
        queue,
        core_attn_out,
        q, k, v, b, a,
        A_log,
        dt_bias,
        ssm_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        (int)num_prefills,
        (int)num_decodes);
}
