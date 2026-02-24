from contextlib import contextmanager
from math import pow
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.optim
from torch import Tensor
from torch.distributed.tensor import DTensor, distribute_tensor

ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(torch.optim.Optimizer):
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
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError("Invalid number of MC samples: {}".format(mc_samples))
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

    def _reset_samples(self):
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None
        self.state["param_avgs"] = None
        self.state["noise"] = None

    def _init_buffers(self):
        for group in self.param_groups:
            l_numel = group["local_numel"]
            hess_init = group["hess_init"]

            group["momentum"] = torch.zeros(l_numel, device=self._device, dtype=self._dtype)

            group["hess"] = torch.full((l_numel,), hess_init, device=self._device, dtype=self._dtype)

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(self, train: bool, param_avg: Tensor = None, noise: Tensor = None):
        if not self._is_noised:
            return
        if param_avg is None:
            param_avg = self.state["param_avgs"]
        if noise is None:
            noise = self.state["noise"]
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                # Determine the size of the data we actually stored for this rank
                if isinstance(p, DTensor):
                    local_tensor = p.to_local()
                    current_numel = local_tensor.numel()
                else:
                    current_numel = p.numel()

                p_slice = slice(offset, offset + current_numel)
                restore_data = param_avg[p_slice]

                if isinstance(p, DTensor):
                    local_tensor = p.to_local()
                    local_tensor.data.copy_(restore_data.view(local_tensor.shape))
                else:
                    p.data.copy_(restore_data.view(p.shape))

                if train:
                    if p.requires_grad:
                        # Ensure we grab the local grad if it's a DTensor
                        g = p.grad.to_local() if isinstance(p, DTensor) else p.grad
                        param_grads.append(g.flatten())
                    else:
                        # Match the local size for the zero tensor
                        z_size = local_tensor.shape if isinstance(p, DTensor) else p.shape
                        param_grads.append(torch.zeros(z_size, device=p.device, dtype=p.dtype).flatten())

                offset += current_numel
        assert offset == self._local_numel, f"Offset {offset} does not match total number of parameters {self._local_numel}"
        if train:  # collect grad sample for training
            grad_sample = torch.cat(param_grads, 0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(self.state["avg_grad"], grad_sample, count)
            if self.hess_approx == "price":
                self.state["avg_nxg"] = _welford_mean(self.state["avg_nxg"], noise * grad_sample, count)
            elif self.hess_approx == "gradsq":
                self.state["avg_gsq"] = _welford_mean(self.state["avg_gsq"], grad_sample.square(), count)
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
        dist.all_reduce(self.state["avg_nxg"])
        self.state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        if self._is_noised:
            return self.state["param_avgs"], self.state["noise"]
        noise_samples = []
        param_avgs = []
        sync_seed = self.current_step + 42
        g = torch.Generator(device=self._device)
        g.manual_seed(sync_seed)

        local_offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            raw_noise_sample = torch.randn(gnumel, device=self._device, dtype=self._dtype, generator=g)

            goffset = 0
            group_noise = []
            for p in group["params"]:
                if p is None:
                    continue

                p_global_numel = p.numel()
                p_raw_noise_slice = raw_noise_sample[goffset : goffset + p_global_numel]

                if isinstance(p, DTensor):
                    # to_local() returns a torch.Tensor that is a view of the DTensor
                    # any changes to _local.data will be reflected in the DTensor
                    p_local: torch.Tensor = p.to_local()
                    param_avgs.append(p_local.data.flatten().clone())

                    p_noise_dtensor: DTensor = distribute_tensor(
                        p_raw_noise_slice.view(p.shape),
                        p.device_mesh,
                        p.placements,
                    )
                    local_p_numel = p_local.numel()
                    h_slice: torch.Tensor = group["hess"][local_offset : local_offset + local_p_numel]

                    p_noise_local: torch.Tensor = p_noise_dtensor.to_local() / (group["ess"] * (h_slice.view(p_local.shape) + group["weight_decay"])).sqrt()
                    p_local.data.add_(p_noise_local)

                    group_noise.append(p_noise_local.flatten())
                    local_offset += local_p_numel
                else:
                    param_avgs.append(p.data.flatten().clone())
                    scaled_noise = p_raw_noise_slice.view(p.shape) / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
                    p.data.add_(scaled_noise)
                    group_noise.append(scaled_noise.flatten())
                    local_offset += p_global_numel

                goffset += p_global_numel
            assert goffset == gnumel
            noise_samples.append(torch.cat(group_noise, 0))

        assert local_offset == self._local_numel
        self._is_noised = True
        self.state["param_avgs"] = torch.cat(param_avgs, 0)
        self.state["noise"] = torch.cat(noise_samples, 0)
        return self.state["param_avgs"], self.state["noise"]

    def _update(self):
        self.current_step += 1

        local_offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]

            l_numel = group["local_numel"]
            local_pg_slice = slice(local_offset, local_offset + l_numel)

            param_avg_list = []
            for p in group["params"]:
                if p is not None:
                    if isinstance(p, DTensor):
                        param_avg_list.append(p.to_local().flatten())
                    else:
                        param_avg_list.append(p.flatten())
            param_avg = torch.cat(param_avg_list, 0)

            group["momentum"] = self._new_momentum(self.state["avg_grad"][local_pg_slice], group["momentum"], b1)

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                local_pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr,
                group["weight_decay"],
                group["clip_radius"],
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
                group["hess_init"],
            )

            pg_local_offset = 0
            for p in group["params"]:
                if p is not None:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        p_local.data.copy_(param_avg[pg_local_offset : pg_local_offset + p_local.numel()].view(p_local.shape))
                        pg_local_offset += p_local.numel()
                    else:
                        p.data.copy_(param_avg[pg_local_offset : pg_local_offset + p.numel()].view(p.shape))
                        pg_local_offset += p.numel()

            assert pg_local_offset == l_numel
            local_offset += l_numel

        assert local_offset == self._local_numel

    def _restore_noised_params(self, noise: Tensor):
        offset = 0
        if self._is_noised:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue
                if isinstance(p, DTensor):
                    p_local = p.to_local()
                    current_numel = p_local.numel()
                else:
                    current_numel = p.numel()
                noise_slice = slice(offset, offset + current_numel)
                restore_noise = noise[noise_slice]
                if isinstance(p, DTensor):
                    p_local.data.add_(restore_noise.view(p_local.shape))
                else:
                    p.data.add_(restore_noise.view(p.shape))
                offset += current_numel
        assert offset == self._local_numel, f"Offset {offset} does not match total number of parameters {self._local_numel}"
        self._is_noised = True

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == "price":
            return avg_nxg[pg_slice] * hess
        elif method == "gradsq":
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f"unknown hessian approx.: {method}")

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd) -> Tensor:
        f = IVON._get_nll_hess(method, hess + wd, avg_nxg, avg_gsq, pg_slice) * ess
        return beta2 * hess + (1.0 - beta2) * f + (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)

    @staticmethod
    def _new_param_averages(param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )
