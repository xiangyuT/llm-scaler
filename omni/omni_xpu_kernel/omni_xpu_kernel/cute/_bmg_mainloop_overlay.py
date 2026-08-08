"""Checked transformations for the private BMG sycl-tla FMHA overlay.

The public source tree keeps the pinned upstream sycl-tla headers unchanged.
The package build applies these transformations to a build-local header copy.
Every source fragment must match exactly once so an upstream header change
cannot silently produce a partially integrated kernel.
"""


WAN_ANIMATE2_POLICY_ORIGINAL = """\
template <int Stages> class XeDefault {};   // Default FMHA mainloop, P in registers.
"""

WAN_ANIMATE2_POLICY_REPLACEMENT = """\
// The second template argument selects the iteration-86 structural mainloop.
// It defaults off so every pre-existing CUTE instantiation retains the pinned
// upstream execution order and online-softmax work.
template <int Stages, bool WanAnimate2I86 = false>
class XeDefault {};   // Default FMHA mainloop, P in registers.
"""

WAN_ANIMATE2_SPECIALIZATION_ORIGINAL = """\
template <int Stages,
          bool CausalMask_, bool CachedKV_, bool PagedKV_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_,
          class TiledCopyK_cache_, class TiledCopyV_cache_>
struct FMHAFwdMainloop<XeDefault<Stages>, CausalMask_, CachedKV_, PagedKV_,
"""

WAN_ANIMATE2_SPECIALIZATION_REPLACEMENT = """\
template <int Stages, bool WanAnimate2I86,
          bool CausalMask_, bool CachedKV_, bool PagedKV_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_,
          class TiledCopyK_cache_, class TiledCopyV_cache_>
struct FMHAFwdMainloop<XeDefault<Stages, WanAnimate2I86>,
                       CausalMask_, CachedKV_, PagedKV_,
"""

WAN_ANIMATE2_Q_FRAGMENT_ORIGINAL = """\
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
"""

WAN_ANIMATE2_Q_FRAGMENT_REPLACEMENT = """\
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    // Only the i86 specialization consumes this fragment. The generic
    // specialization leaves it dead and the compiler removes it.
    decltype(tQrQ) tQrQ_cached0;
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
"""

WAN_ANIMATE2_Q_PRELOAD_ORIGINAL = """\
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
"""

WAN_ANIMATE2_Q_PRELOAD_REPLACEMENT = """\
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    if constexpr (WanAnimate2I86) {
      // Q is invariant across the K loop. Cache one register fragment; this
      // is the only fragment with a repeat load in the validated Q256 policy.
      copy(copy_q, tQgQ(_,_,_,0), tQrQ_cached0);
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
"""

WAN_ANIMATE2_GEMM_ORIGINAL = """\
      /* GEMM 1: S = K * Q */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
      }
"""

WAN_ANIMATE2_GEMM_REPLACEMENT = """\
      if constexpr (WanAnimate2I86) {
        // Start V fetch before Q*K so the long-KV path can overlap its latency
        // with GEMM 1. Other CUTE contracts retain the upstream ordering.
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
        }
      }

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        if constexpr (WanAnimate2I86) {
          if (D == 0) {
            reorder(tQrQ_cached0, tSrQ);
          } else {
            copy(copy_q, tQgQ(_,_,_,D), tQrQ);
            reorder(tQrQ, tSrQ);
          }
        } else {
          copy(copy_q, tQgQ(_,_,_,D), tQrQ);
          reorder(tQrQ, tSrQ);
        }
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      if constexpr (!WanAnimate2I86) {
        /* V prefetch for GEMM 2 */
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
        }
      }
"""

WAN_ANIMATE2_SOFTMAX_CALL_ORIGINAL = """\
      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
      auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);
        if (K != blk_k0) {
"""

WAN_ANIMATE2_SOFTMAX_CALL_REPLACEMENT = """\
      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
      bool subgroup_needs_rescale = K != blk_k0;
      auto rescale = softmax(
          K == blk_k0, tSrS, tA_max, tA_sum,
          subgroup_needs_rescale);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);
        if (subgroup_needs_rescale) {
"""

WAN_ANIMATE2_SOFTMAX_ORIGINAL = """\
  FragSRow
  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum) {    // Softmax row-wise sum accumulator
    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);

    return rescale;
  }
"""

WAN_ANIMATE2_SOFTMAX_REPLACEMENT = """\
  FragSRow
  softmax(bool       first_block,             // First softmax block?
          FragS    & tS,                      // Softmax src/dst block
          FragSRow & tS_max,                  // Row-wise max accumulator
          FragSRow & tS_sum,                  // Row-wise sum accumulator
          bool     & subgroup_needs_rescale) {
    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    if constexpr (WanAnimate2I86) {
      bool local_max_unchanged = true;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
        rescale(i) = tS_max(i) - new_max;
        local_max_unchanged =
            local_max_unchanged && (rescale(i) == ElementS(0));
        tS_max(i) = new_max;
      }
      bool subgroup_max_unchanged = sycl::all_of_group(
          sycl::ext::oneapi::this_work_item::get_sub_group(),
          local_max_unchanged);
      subgroup_needs_rescale = !first_block && !subgroup_max_unchanged;
      if (subgroup_needs_rescale) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < rescale.size(); i++) {
          rescale(i) = sycl::native::exp2(rescale(i));
        }
      } else {
        fill(rescale, ElementS(1));
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
        rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
        tS_max(i) = new_max;
      }
      subgroup_needs_rescale = !first_block;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums */
    if (subgroup_needs_rescale) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);

    return rescale;
  }
"""


WAN_ANIMATE2_I86_TRANSFORMS = (
    (
        "wan-animate2 policy selector",
        WAN_ANIMATE2_POLICY_ORIGINAL,
        WAN_ANIMATE2_POLICY_REPLACEMENT,
    ),
    (
        "wan-animate2 policy specialization",
        WAN_ANIMATE2_SPECIALIZATION_ORIGINAL,
        WAN_ANIMATE2_SPECIALIZATION_REPLACEMENT,
    ),
    (
        "wan-animate2 cached Q fragment",
        WAN_ANIMATE2_Q_FRAGMENT_ORIGINAL,
        WAN_ANIMATE2_Q_FRAGMENT_REPLACEMENT,
    ),
    (
        "wan-animate2 Q preload",
        WAN_ANIMATE2_Q_PRELOAD_ORIGINAL,
        WAN_ANIMATE2_Q_PRELOAD_REPLACEMENT,
    ),
    (
        "wan-animate2 QK/V ordering",
        WAN_ANIMATE2_GEMM_ORIGINAL,
        WAN_ANIMATE2_GEMM_REPLACEMENT,
    ),
    (
        "wan-animate2 softmax call",
        WAN_ANIMATE2_SOFTMAX_CALL_ORIGINAL,
        WAN_ANIMATE2_SOFTMAX_CALL_REPLACEMENT,
    ),
    (
        "wan-animate2 subgroup max skip",
        WAN_ANIMATE2_SOFTMAX_ORIGINAL,
        WAN_ANIMATE2_SOFTMAX_REPLACEMENT,
    ),
)


def apply_checked_transforms(text, transforms):
    """Apply named exact-once transformations to *text*."""
    for name, original, replacement in transforms:
        matches = text.count(original)
        if matches != 1:
            raise RuntimeError(
                f"BMG CUTE overlay transform {name!r} expected one source "
                f"match, found {matches}"
            )
        text = text.replace(original, replacement, 1)
    return text
