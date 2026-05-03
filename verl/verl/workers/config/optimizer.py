# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from omegaconf import MISSING

from verl.base_config import BaseConfig

__all__ = ["OptimizerConfig", "FSDPOptimizerConfig", "McoreOptimizerConfig", "build_optimizer", "VeOmniOptimizerConfig"]


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """VeOmni optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adamw".
        lr (float): Learning rate.
        lr_min (float): Minimum learning rate.
        lr_start (float): Starting learning rate for warmup.
        lr_decay_ratio (float): LR decay ratio.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None
    optimizer_load_path: Optional[str] = None
    ivon_config: Optional[dict] = None

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn("`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2)
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        parameters: Model parameters to optimize
        config: FSDPOptimizerConfig with optimizer settings

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"
    """
    import importlib

    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }
    optimizer_name_lower = config.optimizer.lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas
    elif "ivon" in optimizer_name_lower:
        optimizer_args["beta1"] = config.betas[0]
        optimizer_args["beta2"] = config.betas[1]
        optimizer_args["ess"] = config.ivon_config.ess
        optimizer_args["hess_init"] = config.ivon_config.hess_init
        optimizer_args["hess_approx"] = config.ivon_config.hess_approx
        optimizer_args["clip_radius"] = config.ivon_config.clip_radius
        optimizer_args["sync"] = config.ivon_config.sync
        optimizer_args["debias"] = config.ivon_config.debias
        optimizer_args["rescale_lr"] = config.ivon_config.rescale_lr
        optimizer_args["mc_samples"] = config.ivon_config.mc_samples

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)
    try:
        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'.Available optimizers: {dir(module)}") from e

    optimizer = optimizer_cls(parameters, **optimizer_args)
    print(f"Before loading IVON: Hess sum={optimizer.param_groups[0]['hess'].sum()}, Hess min={optimizer.param_groups[0]['hess'].min()}")
    print(f"Before loading IVON: Hess momentum={optimizer.param_groups[0]['momentum'].sum()}, Hess min={optimizer.param_groups[0]['momentum'].min()}")
    if config.optimizer_load_path and "ivon" in optimizer_name_lower:
        print(f"Loading optimizer from {config.optimizer_load_path}")
        optimizer = _load_ivon_checkpoint(optimizer, config.optimizer_load_path)
    if "ivon" in optimizer_name_lower:
        for group in optimizer.param_groups:
            group["initial_lr"] = config.lr
            group["lr"] = config.lr
            group["weight_decay"] = config.weight_decay
            group["beta1"] = config.betas[0]
            group["beta2"] = config.betas[1]
            group["ess"] = config.ivon_config.ess
            group["clip_radius"] = config.ivon_config.clip_radius
    print(f"After loading IVON: Hess sum={optimizer.param_groups[0]['hess'].sum()}, Hess min={optimizer.param_groups[0]['hess'].min()}")
    print(f"After loading IVON: Hess momentum={optimizer.param_groups[0]['momentum'].sum()}, Hess min={optimizer.param_groups[0]['momentum'].min()}")
    return optimizer


def _load_optim_state_dict(path_or_repo, filename="optimizer.pt"):
    is_local = os.path.exists(path_or_repo)
    is_dist = dist.is_initialized()
    checkpoint_path = None
    if is_local:
        checkpoint_path = os.path.join(path_or_repo, filename) if os.path.isdir(path_or_repo) else path_or_repo
    else:
        if dist.get_rank() == 0:
            checkpoint_path = hf_hub_download(repo_id=path_or_repo, filename=filename)
        if is_dist:
            path_list = [checkpoint_path]
            dist.broadcast_object_list(path_list, src=0)
            checkpoint_path = path_list[0]
    if is_dist:
        dist.barrier()
    return torch.load(checkpoint_path, map_location="cpu", mmap=True)


def _load_ivon_checkpoint(optimizer, optim_load_path):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    optim_state_dict = _load_optim_state_dict(optim_load_path)
    for group in optim_state_dict["param_groups"]:
        optim_numel = group["numel"]
        if optim_numel % world_size:
            raise RuntimeError(f"Total elements {optim_numel} must be divisible by world_size {world_size}")
        slice_size = optim_numel // world_size
        start, end = rank * slice_size, (rank + 1) * slice_size
        for key in ["momentum", "hess"]:
            group[key] = group[key][start:end].to("cuda", non_blocking=True).clone()
        group["local_numel"] = slice_size

    optimizer.load_state_dict(optim_state_dict)
    return optimizer
