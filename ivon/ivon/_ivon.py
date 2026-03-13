from contextlib import contextmanager
from math import pow
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.optim
from torch import Tensor
from torch.distributed.tensor import DTensor

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
            hess_init, numel = group["hess_init"], group["local_numel"]
            group["momentum"] = torch.zeros(numel, device=self._device, dtype=self._dtype)
            group["hess"] = torch.zeros(numel, device=self._device, dtype=self._dtype).add(torch.as_tensor(hess_init))

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
        offset = 0
        for group in self.param_groups:
            gnumel = group["local_numel"]
            noise_sample = torch.randn(gnumel, device=self._device, dtype=self._dtype) / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
            noise_samples.append(noise_sample)
            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                if isinstance(p, DTensor):
                    p_local = p.to_local()
                    p_numel = p_local.numel()
                    param_avgs.append(p_local.data.flatten().clone())
                    p_noise = noise_sample[goffset : goffset + p_numel]
                    p_local.data.add_(p_noise.view(p_local.shape))
                else:
                    p_numel = p.numel()
                    param_avgs.append(p.data.flatten().clone())
                    p_noise = noise_sample[goffset : goffset + p_numel]
                    p.data.add_(p_noise.view(p.shape))

                goffset += p_numel
                offset += p_numel
            assert goffset == gnumel
        assert offset == self._local_numel
        self._is_noised = True
        self.state["param_avgs"] = torch.cat(param_avgs, 0)
        self.state["noise"] = torch.cat(noise_samples, 0)
        return self.state["param_avgs"], self.state["noise"]

    def _update(self):
        self.current_step += 1
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["local_numel"])

            param_avg = torch.cat([p.to_local().flatten() if isinstance(p, DTensor) else p.flatten() for p in group["params"] if p is not None], 0)

            group["momentum"] = self._new_momentum(self.state["avg_grad"][pg_slice], group["momentum"], b1)

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                pg_slice,
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

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        p_local.data.copy_(param_avg[pg_offset : pg_offset + p_local.numel()].view(p_local.shape))
                        pg_offset += p_local.numel()
                    else:
                        p.data.copy_(param_avg[pg_offset : pg_offset + p.numel()].view(p.shape))
                        pg_offset += p.numel()

            assert pg_offset == group["local_numel"]
            offset += group["local_numel"]

        assert offset == self._local_numel

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
        return_val = beta2 * hess + (1.0 - beta2) * f + (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)
        # if (return_val < 0).any():
        #     print("Negative return_val detected")
        #     neg_mask = return_val < 0
        #     hess_neg = hess[neg_mask]
        #     print(f"{hess_neg=}")
        #     print(f"{hess_neg.shape=}")
        #     print(f"{wd=}")
        #     red_diff = (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd) - beta2 * hess + (1.0 - beta2) * f
        #     print("RED and NON-RED Difference")
        #     print(f"{red_diff.mean()=}")
        #     print(f"{red_diff.min()=}")
        #     print(f"{red_diff.max()=}")
        #     print(f"{(red_diff<0).sum()=}")

        return return_val

    @staticmethod
    def _new_param_averages(param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )
