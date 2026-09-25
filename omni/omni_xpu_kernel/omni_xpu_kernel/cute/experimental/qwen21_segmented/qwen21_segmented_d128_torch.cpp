// Experimental B70 BF16 two-range prefix/current KV CUTE. One FP32 online
// softmax visits the prefix before current KV without materializing torch.cat.
// The C++/Torch/SYCL namespaces are distinct from installed dense and masked.
#define cutlass cute_qwen21_segmented_cutlass
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

namespace qwen21_segmented_d128 {

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
  using ElementO = Element;

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
      MainloopDispatchPolicy, /*Causal=*/false, /*CachedKV=*/true, /*PagedKV=*/false,
      TiledMMAQK, TiledMMAPV, VTiles,
      TensorQ, TensorK, TensorV, TensorK_cache, TensorV_cache,
      void, void, void, void, void>;

  using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
      CollectiveMainloop, ShapeOutput, TensorO, void>;

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
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    const void* k_prefix_ptr, const void* v_prefix_ptr, void* o_ptr,
    int B, int H, int Lq, int Lkv, int Lprefix, int D, float scale,
    int device_index,
    int64_t q_stride_seq = -1, int64_t q_stride_head = -1,
    int64_t q_stride_batch = -1, int64_t k_stride_seq = -1,
    int64_t k_stride_head = -1, int64_t k_stride_batch = -1,
    int64_t v_stride_seq = -1, int64_t v_stride_head = -1,
    int64_t v_stride_batch = -1, int64_t kp_stride_seq = -1,
    int64_t kp_stride_head = -1, int64_t kp_stride_batch = -1,
    int64_t vp_stride_seq = -1, int64_t vp_stride_head = -1,
    int64_t vp_stride_batch = -1, int64_t o_stride_seq = -1,
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
  shape.seq_len_qo = Lq;
  shape.seq_len_kv = Lkv;
  shape.seq_len_kv_cache = Lprefix;
  shape.head_size_qk = D;
  shape.head_size_vo = D;

  // Logical Q/K/O=(seq,dim,head,batch), V=(dim,seq,head,batch).
  // All five source tensors retain their own active strides and pointers.
  const int HD = checked_int(static_cast<int64_t>(H) * D, "H*D");
  const int LqHD = checked_int(static_cast<int64_t>(Lq) * H * D, "Lq*H*D");
  const int LkvHD = checked_int(static_cast<int64_t>(Lkv) * H * D, "Lkv*H*D");
  const int LprefixHD = checked_int(static_cast<int64_t>(Lprefix) * H * D, "Lprefix*H*D");
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
    kp_stride_seq = HD;
    kp_stride_head = D;
    kp_stride_batch = LprefixHD;
    vp_stride_seq = HD;
    vp_stride_head = D;
    vp_stride_batch = LprefixHD;
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
  typename KT::StrideK stride_K_prefix =
      cute::make_stride(
          checked_int(kp_stride_seq, "prefix K sequence stride"), _1{},
          checked_int(kp_stride_head, "prefix K head stride"),
          checked_int(kp_stride_batch, "prefix K batch stride"));
  typename KT::StrideV stride_V_prefix =
      cute::make_stride(
          _1{}, checked_int(vp_stride_seq, "prefix V sequence stride"),
          checked_int(vp_stride_head, "prefix V head stride"),
          checked_int(vp_stride_batch, "prefix V batch stride"));
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
          static_cast<const Element*>(k_prefix_ptr), stride_K_prefix,
          static_cast<const Element*>(v_prefix_ptr), stride_V_prefix,
      },
      {scale, nullptr, 0, nullptr, nullptr, 0, Lq, 1},
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

static bool dense_bhld_or_blhd(const at::Tensor& t, int length) {
  if (t.stride(3) != 1) return false;
  const bool packed = t.stride(1) == static_cast<int64_t>(length) * 128 &&
                      t.stride(2) == 128;
  const bool blhd = t.stride(1) == 128 && t.stride(2) == 4096;
  return packed || blhd;
}

// One online FP32 softmax over two independent physical KV ranges.
static at::Tensor sdp_prefix(const at::Tensor& q,
                             const at::Tensor& k_current,
                             const at::Tensor& v_current,
                             const at::Tensor& k_prefix,
                             const at::Tensor& v_prefix) {
  TORCH_CHECK(q.dim() == 4 && k_current.dim() == 4 && v_current.dim() == 4 &&
              k_prefix.dim() == 4 && v_prefix.dim() == 4,
              "segmented D128 requires BHLD Q/current K/V/prefix K/V");
  TORCH_CHECK(q.device().is_xpu() && k_current.device() == q.device() &&
              v_current.device() == q.device() && k_prefix.device() == q.device() &&
              v_prefix.device() == q.device(),
              "segmented D128 requires one XPU device");
  c10::DeviceGuard device_guard(q.device());
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
              k_current.scalar_type() == q.scalar_type() &&
              v_current.scalar_type() == q.scalar_type() &&
              k_prefix.scalar_type() == q.scalar_type() &&
              v_prefix.scalar_type() == q.scalar_type(),
              "segmented D128 requires BF16 Q/K/V");
  TORCH_CHECK(!q.requires_grad() && !k_current.requires_grad() &&
              !v_current.requires_grad() && !k_prefix.requires_grad() &&
              !v_prefix.requires_grad(), "segmented D128 is forward only");
  const int B = checked_int(q.size(0), "batch");
  const int H = checked_int(q.size(1), "heads");
  const int Q = checked_int(q.size(2), "query length");
  const int D = checked_int(q.size(3), "head width");
  const int C = checked_int(k_current.size(2), "current KV length");
  const int P = checked_int(k_prefix.size(2), "prefix KV length");
  TORCH_CHECK(B == 1 && H == 32 && D == 128 && Q > 0 && C > 0 && P > 0,
              "segmented D128 requires positive B1/H32/Q/current/prefix/D128");
  TORCH_CHECK(k_current.sizes() == v_current.sizes() &&
              k_prefix.sizes() == v_prefix.sizes() &&
              k_current.size(0) == B && k_current.size(1) == H &&
              k_current.size(3) == D && k_prefix.size(0) == B &&
              k_prefix.size(1) == H && k_prefix.size(3) == D,
              "segmented D128 K/V shape mismatch");
  checked_int(static_cast<int64_t>(Q) * H * D, "Q/output element span");
  checked_int(static_cast<int64_t>(C) * H * D, "current KV element span");
  checked_int(static_cast<int64_t>(P) * H * D, "prefix KV element span");
  checked_int(static_cast<int64_t>(C) + P, "logical KV length");
  const int64_t blocks = (static_cast<int64_t>(P) + 63) / 64 +
                         (static_cast<int64_t>(C) + 63) / 64;
  checked_int(blocks * 64, "segmented physical KV tile span");
  for (const auto& tensor : {q, k_current, v_current, k_prefix, v_prefix}) {
    const int length = checked_int(tensor.size(2), "active sequence length");
    TORCH_CHECK(dense_bhld_or_blhd(tensor, length),
                "segmented D128 needs dense packed or BLHD-backed BHLD");
    checked_int(tensor.stride(1), "head stride");
    checked_int(tensor.stride(2), "sequence stride");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 64 == 0,
                "segmented D128 requires 64-byte-aligned active Q/K/V pointers");
  }
  const bool q_packed = q.stride(1) == static_cast<int64_t>(Q) * D;
  at::Tensor output = q_packed
      ? at::empty(q.sizes(), q.options())
      : at::empty({B, Q, H, D}, q.options()).permute({0, 2, 1, 3});
  run_d128_tile<cutlass::bfloat16_t, true>(
      q.data_ptr(), k_current.data_ptr(), v_current.data_ptr(),
      k_prefix.data_ptr(), v_prefix.data_ptr(), output.data_ptr(),
      B, H, Q, C, P, D, 1.0f / std::sqrt(128.0f), q.get_device(),
      q.stride(2), q.stride(1), 0,
      k_current.stride(2), k_current.stride(1), 0,
      v_current.stride(2), v_current.stride(1), 0,
      k_prefix.stride(2), k_prefix.stride(1), 0,
      v_prefix.stride(2), v_prefix.stride(1), 0,
      output.stride(2), output.stride(1), 0);
  return output;
}
}  // namespace qwen21_segmented_d128

TORCH_LIBRARY(qwen21_segmented_d128, m) {
  m.def("sdp_prefix(Tensor q, Tensor k_current, Tensor v_current, Tensor k_prefix, Tensor v_prefix) -> Tensor");
}
TORCH_LIBRARY_IMPL(qwen21_segmented_d128, XPU, m) {
  m.impl("sdp_prefix", TORCH_FN(qwen21_segmented_d128::sdp_prefix));
}
