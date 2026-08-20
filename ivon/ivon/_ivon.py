import os
from contextlib import contextmanager
from math import pow
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.optim
from torch import Tensor
from torch.distributed.tensor import DTensor

ClosureType = Callable[[], Tensor]

# Optimizer state (momentum, hess, gradient statistics, noise) is kept in fp32
# regardless of parameter dtype: with beta2 ~ 1 - 1e-5 the Hessian EMA update
# is below bf16 resolution and would silently stop updating.
STATE_DTYPE = torch.float32


class IVON(torch.optim.Optimizer):
    """IVON with a memory-lean state layout.

    Persistent state is exactly momentum + hess (fp32, same footprint as AdamW).
    The injected noise is never stored: it is regenerated bit-identically from
    the generator state saved at sampling time, so restoring the parameter
    average and forming the noise*grad Hessian statistic cost one extra randn
    pass instead of a model-sized buffer. Noise is drawn per-parameter (one
    randn call per param, in param-group order); regeneration must follow the
    exact same call pattern or the streams diverge.

    ``train_mc_samples`` declares how many train=True noise samples are
    accumulated between step() calls:

    * ``1`` (single-sample fast path): no gradient statistics are stored at
      all. Gradients are read from ``p.grad`` at step() time and the noise is
      regenerated, so the caller must not scale ``p.grad`` between the
      train=True restore and step(). Sampling again before step() abandons
      the pending gradient sample (the skipped-step flow).
    * ``>1`` (buffered path, M3PO / decoupled-MC): flat fp32 Welford
      accumulators (avg_grad and avg_nxg/avg_gsq) are allocated lazily at the
      first train=True restore, updated in place per-parameter, and freed at
      step().
    """

    hessian_approx_methods = (
        "price",
        "gradsq",
    )

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = "price",
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
        noise_seed: Optional[int] = None,
        isotropic_noise: bool = False,
        train_mc_samples: int = 1,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError("Invalid number of MC samples: {}".format(mc_samples))
        if not 1 <= train_mc_samples:
            raise ValueError("Invalid number of train MC samples: {}".format(train_mc_samples))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not 0.0 < hess_init:
            raise ValueError("Invalid Hessian initialization: {}".format(hess_init))
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(hess_approx))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._local_numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr
        # Variant-B ablation: when True the per-element Hessian is frozen at
        # hess_init (never learned) and the injected noise uses a single scalar
        # scale, so IVON reduces to momentum-SGD + weight decay with isotropic
        # Gaussian parameter noise. See sampled_params / _update.
        self.isotropic_noise = isotropic_noise
        self.train_mc_samples = train_mc_samples
        # Single-sample fast path: with exactly one train=True noise sample per
        # step, no gradient statistics need to be buffered at all (grads are
        # read from p.grad at step time, noise is regenerated). sync=True needs
        # the flat avg_grad buffer for its all-reduce, so it forces buffering.
        self._single_sample = train_mc_samples == 1 and mc_samples == 1 and not sync

        self._has_dtensor = any(
            isinstance(p, DTensor) for pg in self.param_groups for p in pg["params"] if p is not None
        )
        if self.sync and self._has_dtensor:
            raise ValueError(
                "sync=True all-reduces gradient statistics across ranks that own different parameter"
                " shards and is only valid for replicated (DDP-style) parameters; it must be False"
                " with sharded (FSDP/DTensor) parameters."
            )
        self._validate_dtensor_placements()

        # Noise generator owned by the optimizer: per-rank streams must be
        # decorrelated for the joint perturbation to be an iid posterior
        # sample, so never rely on the (possibly rank-identical) default RNG.
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._generator = torch.Generator(device=self._device)
        if noise_seed is None:
            self._generator.manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
        else:
            self._generator.manual_seed(int(noise_seed) + rank)
        # Generator state captured just before the current noise sample was
        # drawn; replaying randn calls from it reproduces the noise exactly.
        self._gen_state: Optional[Tensor] = None

        # set initial temporary running averages
        self._reset_samples()
        # init all states
        self._init_buffers()
        self._is_noised = False

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            pg["local_numel"] = sum((p.to_local().numel() if isinstance(p, DTensor) else p.numel()) for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(f"Parameters are on different devices: {[str(d) for d in devices]}")
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(f"Parameters are on different dtypes: {[str(d) for d in dtypes]}")
        dtype = next(iter(dtypes))
        # global_total uses the global numels we just set in the pg dict
        global_total = sum(pg["numel"] for pg in self.param_groups)
        # local_total is what we use for the final buffer assertions.
        # This is needed because DTensors are sharded across ranks.
        # If we don't use FSDP, local_total will be equal to global_total.
        local_total = sum(pg["local_numel"] for pg in self.param_groups)
        return global_total, local_total, device, dtype

    def _validate_dtensor_placements(self):
        # Each parameter element must be owned by exactly one rank: with a
        # replicated (HSDP) or partial placement, per-rank noise would
        # desynchronize the model replicas.
        for pg in self.param_groups:
            for p in pg["params"]:
                if p is None or not isinstance(p, DTensor):
                    continue
                for mesh_dim, placement in enumerate(p.placements):
                    if p.device_mesh.size(mesh_dim) == 1:
                        continue
                    if not placement.is_shard():
                        raise ValueError(
                            f"IVON requires fully-sharded parameters, but got placement {placement} on"
                            f" device-mesh dim {mesh_dim} (size {p.device_mesh.size(mesh_dim)}) for a"
                            f" parameter of shape {tuple(p.shape)}. Replicated/partial placements"
                            " (e.g. HSDP) are unsupported: replicas would receive uncorrelated noise."
                        )

    def _reset_samples(self):
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None
        # noise is never stored anymore; drop the buffer a pre-memory-lean
        # optimizer checkpoint may have carried in through load_state_dict.
        self.state.pop("noise", None)
        self._gen_state = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["local_numel"]
            group["momentum"] = torch.zeros(numel, device=self._device, dtype=STATE_DTYPE)
            group["hess"] = torch.zeros(numel, device=self._device, dtype=STATE_DTYPE).add(torch.as_tensor(hess_init))

    @staticmethod
    def _local(p: Tensor) -> Tensor:
        return p.to_local() if isinstance(p, DTensor) else p

    def _local_params(self):
        """Yield (group, p, p_local, group_offset) in the canonical order used
        by every noise generation/regeneration pass."""
        for group in self.param_groups:
            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue
                p_local = self._local(p)
                yield group, p, p_local, goffset
                goffset += p_local.numel()
            assert goffset == group["local_numel"]

    def _draw_noise(self, group, hess_slice: Tensor, numel: int, generator: torch.Generator) -> Tensor:
        noise = torch.randn(numel, device=self._device, dtype=STATE_DTYPE, generator=generator)
        if self.isotropic_noise:
            # Isotropic ablation: a single scalar scale derived from hess_init
            # (NOT the per-element hess), so the perturbation is plain scaled
            # identity noise. Matched to a baseline run's mean noise variance
            # by choosing hess_init (scripts/ivon_isotropic_target.py). The
            # ess-schedule still modulates magnitude exactly as in curvature mode.
            noise /= (group["ess"] * (group["hess_init"] + group["weight_decay"])) ** 0.5
        else:
            noise /= (group["ess"] * (hess_slice + group["weight_decay"])).sqrt()
        return noise

    def _regen_generator(self) -> torch.Generator:
        assert self._gen_state is not None, "no noise sample to regenerate"
        gen = torch.Generator(device=self._device)
        gen.set_state(self._gen_state)
        return gen

    @contextmanager
    def sampled_params(self, train: bool = False):
        self._sample_params()
        yield
        self._restore_param_average(train)

    @torch.no_grad()
    def _restore_param_average(self, train: bool):
        if not self._is_noised:
            return
        if train:
            count = self.state["count"] + 1
            self.state["count"] = count
            if self._single_sample:
                assert count == 1  # _sample_params abandons any pending sample
            else:
                if self.state["avg_grad"] is None:
                    self.state["avg_grad"] = torch.zeros(self._local_numel, device=self._device, dtype=STATE_DTYPE)
                # Isotropic ablation freezes the Hessian, so the noise*grad /
                # grad^2 curvature statistics are never consumed — skip them.
                if not self.isotropic_noise:
                    if self.hess_approx == "price" and self.state["avg_nxg"] is None:
                        self.state["avg_nxg"] = torch.zeros(self._local_numel, device=self._device, dtype=STATE_DTYPE)
                    elif self.hess_approx == "gradsq" and self.state["avg_gsq"] is None:
                        self.state["avg_gsq"] = torch.zeros(self._local_numel, device=self._device, dtype=STATE_DTYPE)
        gen = self._regen_generator()
        offset = 0
        for group, p, p_local, goffset in self._local_params():
            pn = p_local.numel()
            noise = self._draw_noise(group, group["hess"][goffset : goffset + pn], pn, gen)
            p_local.data.sub_(noise.view(p_local.shape).to(p_local.dtype))
            if train and not self._single_sample:
                count = self.state["count"]
                if p.grad is not None:
                    g32 = self._local(p.grad).flatten().to(STATE_DTYPE)
                else:
                    g32 = torch.zeros(pn, device=self._device, dtype=STATE_DTYPE)
                sl = slice(offset, offset + pn)
                # In-place Welford: avg += (new - avg) / count. g32 may alias
                # p.grad (fp32 params), so never mutate it; noise is ours.
                if not self.isotropic_noise:
                    if self.hess_approx == "price":
                        nbuf = self.state["avg_nxg"][sl]
                        nbuf.add_(noise.mul_(g32).sub_(nbuf), alpha=1.0 / count)
                    elif self.hess_approx == "gradsq":
                        gbuf = self.state["avg_gsq"][sl]
                        gbuf.add_(g32.square().sub_(gbuf), alpha=1.0 / count)
                abuf = self.state["avg_grad"][sl]
                abuf.add_(g32.sub(abuf), alpha=1.0 / count)
            offset += pn
        assert offset == self._local_numel, f"Offset {offset} does not match total number of parameters {self._local_numel}"
        self._is_noised = False

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():  # explicit sync
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        # avg_nxg is not accumulated in the isotropic ablation (Hessian frozen).
        if not self.isotropic_noise:
            dist.all_reduce(self.state["avg_nxg"])
            self.state["avg_nxg"].div_(world_size)

    @torch.no_grad()
    def _sample_params(self) -> None:
        if self._is_noised:
            return
        if self._single_sample and self.state["count"]:
            # A train=True sample was collected but step() never ran (e.g. the
            # non-finite-grad skip, which also zeroed the grads). Abandon it:
            # the fresh noise pairs with the grads of the upcoming backward.
            self.state["count"] = 0
        self._gen_state = self._generator.get_state()
        offset = 0
        for group, _, p_local, goffset in self._local_params():
            pn = p_local.numel()
            noise = self._draw_noise(group, group["hess"][goffset : goffset + pn], pn, self._generator)
            p_local.data.add_(noise.view(p_local.shape).to(p_local.dtype))
            offset += pn
        assert offset == self._local_numel
        self._is_noised = True

    @torch.no_grad()
    def _regenerate_noise(self) -> Tensor:
        """Rebuild the flat noise vector of the current sample from the saved
        generator state. Debug/test helper: allocates a full local-numel
        buffer, so never call it on the training path."""
        gen = self._regen_generator()
        chunks = []
        for group, _, p_local, goffset in self._local_params():
            pn = p_local.numel()
            chunks.append(self._draw_noise(group, group["hess"][goffset : goffset + pn], pn, gen))
        return torch.cat(chunks, 0)

    def _update(self):
        self.current_step += 1
        if self._single_sample:
            if self.state["count"] != 1:
                raise RuntimeError(
                    "IVON.step() in single-sample mode requires exactly one train=True"
                    f" restore before each step, got {self.state['count']}. For multiple"
                    " MC samples per step construct IVON with train_mc_samples > 1."
                )
            # The price statistic needs the sampled noise; replay it in the
            # same canonical per-param order it was drawn in.
            regen = None if self.isotropic_noise or self.hess_approx != "price" else self._regen_generator()
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            wd = group["weight_decay"]
            ess = group["ess"]
            debias = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0
            lr_eff = lr * (group["hess_init"] + wd) if self.rescale_lr else lr
            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue
                p_local = self._local(p)
                pn = p_local.numel()
                gsl = slice(goffset, goffset + pn)
                m = group["momentum"][gsl]
                h = group["hess"][gsl]

                if self._single_sample:
                    # Grads are read directly from p.grad: the caller must not
                    # have scaled them since the train=True restore (verl skips
                    # grad-clip scaling for IVON for exactly this reason).
                    if p.grad is not None:
                        g32 = self._local(p.grad).flatten().to(STATE_DTYPE)
                    else:
                        g32 = torch.zeros(pn, device=self._device, dtype=STATE_DTYPE)
                    if regen is not None:
                        # Noise std uses the pre-update hess, as at sampling time.
                        nxg = self._draw_noise(group, h, pn, regen).mul_(g32)
                else:
                    g32 = self.state["avg_grad"][offset : offset + pn]
                    if not self.isotropic_noise and self.hess_approx == "price":
                        nxg = self.state["avg_nxg"][offset : offset + pn]

                # momentum <- b1 * m + (1 - b1) * grad
                m.mul_(b1).add_(g32, alpha=1.0 - b1)

                # Isotropic ablation: hess is frozen at hess_init (curvature
                # disabled), so the update reduces to momentum-SGD + weight decay.
                if not self.isotropic_noise:
                    hw = h + wd
                    if self.hess_approx == "price":
                        f = nxg * hw
                        f.mul_(ess)
                    else:  # gradsq
                        f = (g32.square() if self._single_sample else self.state["avg_gsq"][offset : offset + pn]) * ess
                    # hess <- b2*h + (1-b2)*f + 0.5*(1-b2)^2 * (h-f)^2 / (h+wd)
                    hmf = h - f
                    h.mul_(b2).add_(f, alpha=1.0 - b2).add_(hmf.square_().div_(hw), alpha=0.5 * (1.0 - b2) ** 2)

                # param <- param - lr_eff * clip((m/debias + wd*param) / (h_new + wd))
                p32 = p_local.data.flatten().to(STATE_DTYPE)
                upd = m / debias
                upd.add_(p32, alpha=wd).div_(h + wd).clamp_(min=-group["clip_radius"], max=group["clip_radius"])
                if p32.dtype == p_local.dtype and p32.data_ptr() == p_local.data_ptr():
                    p32.sub_(upd, alpha=lr_eff)  # p32 aliases the param: in-place update
                else:
                    p_local.data.copy_(p32.sub_(upd, alpha=lr_eff).view(p_local.shape))

                goffset += pn
                offset += pn

            assert goffset == group["local_numel"]
        assert offset == self._local_numel
