// Experimental B70 BF16 actual-bool-mask CUTE, derived from the public tuning
// C36 suffix4 component. No fixed segment lengths or inferred causal mask.
// This candidate adds one-partition direct BF16 output while retaining the
// prior partial/merge paths. Its C++ and Torch namespaces differ from both
// installed dense CUTE and the previously built masked sidecar.
#define cutlass cute_qwen21_masked_direct_cutlass
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <c10/core/DeviceGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/device_kernel.h"
#include "cute/util/compat.hpp"

#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "cute_fmha_config.h"

using namespace cute;

namespace qwen21_masked_direct_d128 {

// ---- launch glue: submit cutlass device kernel onto torch's XPU queue --------
// (mirror of sgl-kernel-xpu src/sycl/comm/common.h::launch)
template <typename Kernel, int GrfSize>
class CuteFmhaKernelTag {};

template <typename Kernel, int GrfSize = 256>
static void launch_on_torch_queue(typename Kernel::Params params, int device_index) {
  static_assert(GrfSize == 128 || GrfSize == 256, "GRF size must be 128 or 256");

  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);
  int smem_size = Kernel::SharedStorageSize;

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<GrfSize>};
  compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};

  syclex::launch_config config(policy.get_range(), policy.get_launch_properties());
  auto cgf = [&](::sycl::handler& cgh) {
    auto KernelFunctor =
        compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<Kernel>>(cgh, policy, params);
    syclex::detail::LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())>
        ConfigAccess(config);
    cgh.parallel_for<CuteFmhaKernelTag<Kernel, GrfSize>>(
        ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelFunctor);
  };
  c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_index));
  auto stream = c10::xpu::getCurrentXPUStream(device_index);
  auto q = stream.queue();
  q.submit(cgf);
}

static int checked_int(int64_t value, const char* label) {
  TORCH_CHECK(
      value >= 0 && value <= std::numeric_limits<int>::max(), label,
      " exceeds the CUTE int32 index range: ", value);
  return static_cast<int>(value);
}

// ---- kernel type assembly (128-wide tile, example-06 PREFILL path) ----------
// KV tile = get<1>(ShapeQK). Default 32 (stock example-06). -DCUTE_FMHA_KV64
// switches to a 64-wide KV tile (fewer K-loop iters at large seq — omni uses 64).
template <
    typename Element,
    bool DirectOutput = false,
    int PipelineStagesOverride = 0,
    int QTileOverride = 0,
    int SubgroupLayoutQOverride = 0,
    int MmaKOverride = 0,
    int VTileOverride = 0,
    int HeadDimOverride = 0,
    int KvTileOverride = 0>
struct D128TileKernel {
  using PlatformConfig = cute_fmha_config::ActiveConfig;
  static constexpr int QTile =
      QTileOverride > 0 ? QTileOverride : PlatformConfig::Q_TILE;
  static constexpr int SubgroupLayoutQ =
      SubgroupLayoutQOverride > 0
          ? SubgroupLayoutQOverride
          : PlatformConfig::SUBGROUP_LAYOUT_Q;
  static constexpr int MmaK =
      MmaKOverride > 0 ? MmaKOverride : PlatformConfig::MMA_K;
  static constexpr int VTile =
      VTileOverride > 0 ? VTileOverride : PlatformConfig::V_TILE;
  static constexpr int HeadDim =
      HeadDimOverride > 0 ? HeadDimOverride : PlatformConfig::HEAD_DIM;
#if defined(CUTE_FMHA_KV64)
  // KV tile = get<1>(ShapeQK) = 64. Per get_tiled_mma_pv, the PV tile must be
  // <TileQ, TileV, KVtile> — so ShapePV's K-dim (3rd) MUST equal 64, not 32.
  // TileV=32 -> VTiles = 128/32 = 4. (My earlier <256,32,32> broke the QK->PV
  // K-dim match and tripped the gemm.hpp static_assert.)
  static constexpr int KvTile = 64;
#else
  static constexpr int KvTile =
      KvTileOverride > 0 ? KvTileOverride : PlatformConfig::KV_TILE;
#endif
  using ShapeQK = Shape<
      Int<QTile>, Int<KvTile>, Int<MmaK>>;
  using ShapePV = Shape<
      Int<QTile>, Int<VTile>, Int<KvTile>>;
  using ShapeOutput = Shape<
      Int<QTile>, Int<HeadDim>>;
  using SubgroupLayoutQK = Layout<
      Shape<Int<SubgroupLayoutQ>, _1, _1>>;
#ifdef CUTE_FMHA_STAGES
  static constexpr int PipelineStages = CUTE_FMHA_STAGES;
#else
  static constexpr int PipelineStages =
      PipelineStagesOverride > 0 ? PipelineStagesOverride
                                 : PlatformConfig::PIPELINE_STAGES;
#endif
  static constexpr int GrfSize = PlatformConfig::GRF_SIZE;

  using ElementQ = Element;
  using ElementK = Element;
  using ElementV = Element;
  using ElementO = std::conditional_t<DirectOutput, Element, float>;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;

  static constexpr int SGTileQ =
      get<0>(shape_div(ShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, Element>;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using TiledMMAQK =
      typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV =
      typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;
  static constexpr int VTiles = get<1>(ShapeOutput{}) / get<1>(ShapePV{});

  static auto make_dummy(Element v, StrideQ s) {
    return make_tensor(make_gmem_ptr(&v), make_layout(repeat<rank_v<StrideQ>>(1), s));
  }
  using TensorQ = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideQ>>(1), StrideQ{})));
  using TensorK = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideK>>(1), StrideK{})));
  using TensorV = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideV>>(1), StrideV{})));
  using TensorO = decltype(make_tensor(make_gmem_ptr((ElementO*)nullptr),
                            make_layout(repeat<rank_v<StrideO>>(1), StrideO{})));
  using TensorK_cache = TensorK;
  using TensorV_cache = TensorV;

  using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
      MainloopDispatchPolicy, /*Causal=*/false, /*CachedKV=*/false, /*PagedKV=*/false,
      TiledMMAQK, TiledMMAPV, VTiles,
      TensorQ, TensorK, TensorV, TensorK_cache, TensorV_cache,
      void, void, void, void, void>;

  using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
      CollectiveMainloop, ShapeOutput, TensorO, void, DirectOutput>;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<false>;
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
      ProblemShapeType, CollectiveMainloop, CollectiveEpilogue,
      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
};

template <
    typename Element,
    bool DirectOutput = false,
    int PipelineStagesOverride = 0,
    int QTileOverride = 0,
    int SubgroupLayoutQOverride = 0,
    int MmaKOverride = 0,
    int VTileOverride = 0,
    int HeadDimOverride = 0,
    int KvTileOverride = 0>
void run_d128_tile(
    const void* q_ptr, const void* k_ptr, const void* v_ptr, void* o_ptr,
    int B, int H, int Lq, int Lkv, int D, float scale, const bool* mask,
    int mask_row_stride, int partitions, int device_index,
    int64_t q_stride_seq = -1, int64_t q_stride_head = -1,
    int64_t q_stride_batch = -1, int64_t k_stride_seq = -1,
    int64_t k_stride_head = -1, int64_t k_stride_batch = -1,
    int64_t v_stride_seq = -1, int64_t v_stride_head = -1,
    int64_t v_stride_batch = -1, int64_t o_stride_seq = -1,
    int64_t o_stride_head = -1, int64_t o_stride_batch = -1) {
  using KT = D128TileKernel<
      Element,
      DirectOutput,
      PipelineStagesOverride,
      QTileOverride,
      SubgroupLayoutQOverride,
      MmaKOverride,
      VTileOverride,
      HeadDimOverride,
      KvTileOverride>;
  using K    = typename KT::Kernel;
  using PS   = typename KT::ProblemShapeType;

  const c10::Device tensor_device(c10::DeviceType::XPU, device_index);
  c10::DeviceGuard device_guard(tensor_device);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_index;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  PS shape;
  shape.batch = B;
  shape.num_heads_q = H;
  shape.num_heads_kv = H;
  shape.seq_len_qo = Lq;   // cross-attention: Lq may differ from Lkv
  shape.seq_len_kv = Lkv;
  shape.seq_len_kv_cache = 0;
  shape.head_size_qk = D;
  shape.head_size_vo = D;

  // Logical cute modes are Q/K/O=(seq,dim,head,batch) and
  // V=(dim,seq,head,batch). Default strides consume contiguous BLHD. The D120
  // entry point supplies the actual dense BHLD/BLHD-backed strides so it can
  // match ComfyUI's mixed input layouts without materializing copies.
  const int HD = checked_int(static_cast<int64_t>(H) * D, "H*D");
  const int LqHD = checked_int(static_cast<int64_t>(Lq) * H * D, "Lq*H*D");
  const int LkvHD = checked_int(static_cast<int64_t>(Lkv) * H * D, "Lkv*H*D");
  if (q_stride_seq < 0) {
    q_stride_seq = HD;
    q_stride_head = D;
    q_stride_batch = LqHD;
    k_stride_seq = HD;
    k_stride_head = D;
    k_stride_batch = LkvHD;
    v_stride_seq = HD;
    v_stride_head = D;
    v_stride_batch = LkvHD;
    o_stride_seq = HD;
    o_stride_head = D;
    o_stride_batch = LqHD;
  }
  typename KT::StrideQ stride_Q =
      cute::make_stride(
          checked_int(q_stride_seq, "Q sequence stride"), _1{},
          checked_int(q_stride_head, "Q head stride"),
          checked_int(q_stride_batch, "Q batch stride"));
  typename KT::StrideK stride_K =
      cute::make_stride(
          checked_int(k_stride_seq, "K sequence stride"), _1{},
          checked_int(k_stride_head, "K head stride"),
          checked_int(k_stride_batch, "K batch stride"));
  typename KT::StrideV stride_V =
      cute::make_stride(
          _1{}, checked_int(v_stride_seq, "V sequence stride"),
          checked_int(v_stride_head, "V head stride"),
          checked_int(v_stride_batch, "V batch stride"));
  typename KT::StrideO stride_O =
      cute::make_stride(
          checked_int(o_stride_seq, "output sequence stride"), _1{},
          checked_int(o_stride_head, "output head stride"),
          checked_int(o_stride_batch, "output batch stride"));

  typename K::Arguments arguments{
      {
          shape,
          static_cast<const Element*>(q_ptr), stride_Q,
          static_cast<const Element*>(k_ptr), stride_K,
          static_cast<const Element*>(v_ptr), stride_V,
          static_cast<typename KT::ElementO*>(o_ptr), stride_O,
          nullptr, stride_K,   // k_cache
          nullptr, stride_V,   // v_cache
      },
      {scale, nullptr, 0, nullptr, mask, mask_row_stride, Lq, partitions},
      {},
      hw_info};

  size_t workspace_size = K::get_workspace_size(arguments);
  auto opts = at::TensorOptions().dtype(at::kByte).device(tensor_device);
  at::Tensor workspace = at::empty({(long)workspace_size}, opts);

  TORCH_CHECK(K::can_implement(arguments),
              "cute_fmha: can_implement failed (bad problem shape)");
  K::initialize_workspace(arguments, workspace.data_ptr());
  auto kernel_params = K::to_underlying_arguments(arguments, workspace.data_ptr());
  launch_on_torch_queue<K, KT::GrfSize>(kernel_params, device_index);
}

template <typename Element> class SplitKVMerge;

template <typename Element>
void merge_split_kv(const float* partial, Element* output, int rows,
                    int query_length, int head_stride, int sequence_stride,
                    int partitions, int device_index) {
  c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_index));
  auto queue = c10::xpu::getCurrentXPUStream(device_index).queue();
  const int plane_stride = rows * 144;
  const int work_items = rows * 16;
  queue.parallel_for<SplitKVMerge<Element>>(
      sycl::nd_range<1>(sycl::range<1>((work_items + 255) / 256 * 256), sycl::range<1>(256)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int row = int(item.get_global_id(0)) / 16;
        if (row >= rows) return;
        const float* first = partial + row * 144;
        auto sg = item.get_sub_group();
        const int lane = int(sg.get_local_linear_id());
        float weights[4] = {0, 0, 0, 0};
        float inverse = 0.0f;
        if (lane == 0) {
          float maximum = -INFINITY;
          for (int p = 0; p < partitions; ++p) {
            const float sum = first[p * plane_stride + 129];
            if (sum > 0.0f) maximum = sycl::max(maximum, first[p * plane_stride + 128]);
          }
          float denominator = 0.0f;
          for (int p = 0; p < partitions; ++p) {
            const float sum = first[p * plane_stride + 129];
            weights[p] = sum > 0.0f
                ? sycl::native::exp2(first[p * plane_stride + 128] - maximum)
                : 0.0f;
            denominator += sum * weights[p];
          }
          inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        }
        for (int p = 0; p < partitions; ++p)
          weights[p] = sycl::group_broadcast(sg, weights[p], 0);
        inverse = sycl::group_broadcast(sg, inverse, 0);
        for (int chunk = 0; chunk < 8; ++chunk) {
          const int column = chunk * 16 + lane;
          float numerator = 0.0f;
          for (int p = 0; p < partitions; ++p)
            numerator += first[p * plane_stride + column] * weights[p];
          output[(row / query_length) * head_stride +
                 (row % query_length) * sequence_stride + column] =
              Element(numerator * inverse);
        }
      });
}

static bool dense_bhld_or_blhd(const at::Tensor& t, int length) {
  if (t.stride(3) != 1) return false;
  const bool packed = t.stride(1) == static_cast<int64_t>(length) * 128 &&
                      t.stride(2) == 128;
  const bool blhd = t.stride(1) == 128 && t.stride(2) == 4096;
  return packed || blhd;
}

// Experimental complete API: direct normalizes FP32 state inside one kernel;
// the prior one/four-partition variants retain FP32 partial and merge behavior.
static at::Tensor sdp_impl(const at::Tensor& q, const at::Tensor& k,
                           const at::Tensor& v, const at::Tensor& mask,
                           int partitions, bool direct_output) {
  TORCH_CHECK(partitions == 1 || partitions == 4, "masked D128 supports 1 or 4 KV partitions");
  TORCH_CHECK(!direct_output || partitions == 1,
              "direct masked D128 requires one KV partition");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "masked D128 requires BHLD Q/K/V");
  TORCH_CHECK(q.device().is_xpu() && k.device() == q.device() &&
              v.device() == q.device() && mask.device() == q.device(),
              "masked D128 requires all tensors on one XPU device");
  c10::DeviceGuard device_guard(q.device());
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
              k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "masked D128 requires BF16 Q/K/V");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
              "masked D128 is forward only");
  const int B = checked_int(q.size(0), "batch");
  const int H = checked_int(q.size(1), "heads");
  const int Lq = checked_int(q.size(2), "query length");
  const int D = checked_int(q.size(3), "head width");
  const int Lkv = checked_int(k.size(2), "key/value length");
  TORCH_CHECK(B == 1 && H == 32 && D == 128 && Lq > 0 && Lkv > 0,
              "masked D128 requires positive B1/H32/Q/KV/D128");
  TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == B &&
              k.size(1) == H && k.size(3) == D, "masked D128 K/V shape mismatch");
  TORCH_CHECK(mask.dim() == 2 && mask.size(0) == Lq && mask.size(1) == Lkv &&
              mask.scalar_type() == at::kBool && mask.is_contiguous(),
              "masked D128 requires contiguous bool mask [Q,KV]");
  checked_int(static_cast<int64_t>(Lq) * Lkv, "mask element address span");
  checked_int(static_cast<int64_t>(Lq) * H * D, "Q/output element address span");
  checked_int(static_cast<int64_t>(Lkv) * H * D, "K/V element address span");
  if (!direct_output)
    checked_int(static_cast<int64_t>(partitions) * H * Lq * 144,
                "FP32 partial element address span");
  for (const auto& t : {q, k, v}) {
    const int length = checked_int(t.size(2), "active sequence length");
    TORCH_CHECK(dense_bhld_or_blhd(t, length),
                "masked D128 needs dense packed BHLD or BLHD-backed BHLD");
    checked_int(t.stride(1), "head stride");
    checked_int(t.stride(2), "sequence stride");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "masked D128 requires 16-byte-aligned active Q/K/V pointers");
  }
  const bool q_packed = q.stride(1) == static_cast<int64_t>(Lq) * D;
  at::Tensor output = q_packed
      ? at::empty(q.sizes(), q.options())
      : at::empty({B, Lq, H, D}, q.options()).permute({0, 2, 1, 3});
  if (direct_output) {
    // FP32 online QK/softmax/PV state is normalized in the epilogue. No FP32
    // partial tensor, zero-fill or merge dispatch belongs to this API call.
    run_d128_tile<cutlass::bfloat16_t, true>(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), output.data_ptr(),
        1, H, Lq, Lkv, D, 1.0f / std::sqrt(128.0f),
        mask.data_ptr<bool>(), Lkv, 1, q.get_device(),
        q.stride(2), q.stride(1), 0,
        k.stride(2), k.stride(1), 0,
        v.stride(2), v.stride(1), 0,
        output.stride(2), output.stride(1), 0);
  } else {
    // Keep the prior one/four-partition algorithm as a same-DSO control.
    at::Tensor partial = at::zeros({partitions, H, Lq, 144},
                                    q.options().dtype(at::kFloat));
    run_d128_tile<cutlass::bfloat16_t, false>(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), partial.data_ptr(),
        partitions, H, Lq, Lkv, D, 1.0f / std::sqrt(128.0f),
        mask.data_ptr<bool>(), Lkv, partitions, q.get_device(),
        q.stride(2), q.stride(1), 0,
        k.stride(2), k.stride(1), 0,
        v.stride(2), v.stride(1), 0,
        partial.stride(2), partial.stride(1), partial.stride(0));
    merge_split_kv(partial.data_ptr<float>(),
                   static_cast<cutlass::bfloat16_t*>(output.data_ptr()),
                   checked_int(static_cast<int64_t>(H) * Lq, "merge row count"),
                   Lq, checked_int(output.stride(1), "output head stride"),
                   checked_int(output.stride(2), "output sequence stride"),
                   partitions, q.get_device());
  }
  return output;
}

at::Tensor sdp(const at::Tensor& q, const at::Tensor& k,
               const at::Tensor& v, const at::Tensor& mask) {
  return sdp_impl(q, k, v, mask, 1, false);
}
at::Tensor sdp_split4(const at::Tensor& q, const at::Tensor& k,
                      const at::Tensor& v, const at::Tensor& mask) {
  return sdp_impl(q, k, v, mask, 4, false);
}
at::Tensor sdp_direct(const at::Tensor& q, const at::Tensor& k,
                      const at::Tensor& v, const at::Tensor& mask) {
  return sdp_impl(q, k, v, mask, 1, true);
}
}  // namespace qwen21_masked_direct_d128

TORCH_LIBRARY(qwen21_masked_direct_d128, m) {
  m.def("sdp(Tensor q, Tensor k, Tensor v, Tensor mask) -> Tensor");
  m.def("sdp_split4(Tensor q, Tensor k, Tensor v, Tensor mask) -> Tensor");
  m.def("sdp_direct(Tensor q, Tensor k, Tensor v, Tensor mask) -> Tensor");
}
TORCH_LIBRARY_IMPL(qwen21_masked_direct_d128, XPU, m) {
  m.impl("sdp", TORCH_FN(qwen21_masked_direct_d128::sdp));
  m.impl("sdp_split4", TORCH_FN(qwen21_masked_direct_d128::sdp_split4));
  m.impl("sdp_direct", TORCH_FN(qwen21_masked_direct_d128::sdp_direct));
}
