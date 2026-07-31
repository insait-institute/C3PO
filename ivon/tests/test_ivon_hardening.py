"""Phase-0 hardening tests for the IVON optimizer (single process, CPU).

Run with: python -m pytest ivon/tests/test_ivon_hardening.py -v
"""

import torch
import torch.nn as nn
import pytest

from ivon import IVON


def make_model(dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 4)).to(dtype)
    return model


def fake_step(model, opt, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(8, 16, dtype=next(model.parameters()).dtype)
    with opt.sampled_params(train=True):
        loss = model(x).square().mean()
        loss.backward()
    opt.step()
    opt.zero_grad()


class TestFp32State:
    def test_state_dtype_fp32_with_bf16_params(self):
        model = make_model(torch.bfloat16)
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, hess_init=0.1, noise_seed=0)
        for group in opt.param_groups:
            assert group["momentum"].dtype == torch.float32
            assert group["hess"].dtype == torch.float32
        fake_step(model, opt)
        for group in opt.param_groups:
            assert group["momentum"].dtype == torch.float32
            assert group["hess"].dtype == torch.float32
        assert opt.state["avg_grad"] is None or opt.state["avg_grad"].dtype == torch.float32

    def test_hess_updates_with_bf16_params(self):
        # With beta2 = 0.99999 the per-step hess update is ~1e-5 relative and
        # vanishes entirely in bf16 arithmetic; fp32 state must capture it.
        model = make_model(torch.bfloat16)
        opt = IVON(
            model.parameters(), lr=1e-3, ess=1e4, hess_init=0.1, beta2=0.99999, noise_seed=0
        )
        hess_before = opt.param_groups[0]["hess"].clone()
        for _ in range(3):
            fake_step(model, opt)
        delta = (opt.param_groups[0]["hess"] - hess_before).abs().max()
        assert delta > 0, "hess did not move at all"

    def test_regenerated_noise_fp32_and_no_stored_buffer(self):
        model = make_model(torch.bfloat16)
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, noise_seed=0)
        opt._sample_params()
        assert "noise" not in opt.state, "noise must not be stored"
        assert opt._regenerate_noise().dtype == torch.float32
        opt._restore_param_average(train=False)


class TestSampleRestoreRoundtrip:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_roundtrip_restores_params(self, dtype):
        model = make_model(dtype)
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, hess_init=0.1, noise_seed=0)
        before = [p.detach().clone() for p in model.parameters()]
        opt._sample_params()
        noised = [p.detach().clone() for p in model.parameters()]
        assert any((b != n).any() for b, n in zip(before, noised)), "no noise applied"
        opt._restore_param_average(train=False)
        after = [p.detach().clone() for p in model.parameters()]
        for b, a in zip(before, after):
            # add-then-subtract of the same (cast) noise loses at most 1 ulp
            atol = 1e-6 if dtype == torch.float32 else 2e-2
            assert torch.allclose(b.float(), a.float(), atol=atol)


class TestNoiseGenerator:
    def test_same_seed_same_noise(self):
        m1, m2 = make_model(seed=1), make_model(seed=1)
        o1 = IVON(m1.parameters(), lr=1e-3, ess=1e4, noise_seed=123)
        o2 = IVON(m2.parameters(), lr=1e-3, ess=1e4, noise_seed=123)
        o1._sample_params()
        o2._sample_params()
        assert torch.equal(o1._regenerate_noise(), o2._regenerate_noise())

    def test_different_seed_different_noise(self):
        m1, m2 = make_model(seed=1), make_model(seed=1)
        o1 = IVON(m1.parameters(), lr=1e-3, ess=1e4, noise_seed=123)
        o2 = IVON(m2.parameters(), lr=1e-3, ess=1e4, noise_seed=124)
        o1._sample_params()
        o2._sample_params()
        assert not torch.equal(o1._regenerate_noise(), o2._regenerate_noise())

    def test_none_seed_uses_entropy(self):
        m1, m2 = make_model(seed=1), make_model(seed=1)
        o1 = IVON(m1.parameters(), lr=1e-3, ess=1e4)
        o2 = IVON(m2.parameters(), lr=1e-3, ess=1e4)
        o1._sample_params()
        o2._sample_params()
        assert not torch.equal(o1._regenerate_noise(), o2._regenerate_noise())

    def test_independent_of_default_rng(self):
        # Identical torch.manual_seed on two "ranks" must not correlate IVON noise
        # when explicit noise_seed differs (the +rank offset in real use).
        torch.manual_seed(7)
        m1 = make_model(seed=1)
        o1 = IVON(m1.parameters(), lr=1e-3, ess=1e4, noise_seed=0)
        torch.manual_seed(7)
        m2 = make_model(seed=1)
        o2 = IVON(m2.parameters(), lr=1e-3, ess=1e4, noise_seed=1)
        o1._sample_params()
        o2._sample_params()
        assert not torch.equal(o1._regenerate_noise(), o2._regenerate_noise())


class TestMemoryLean:
    def test_regenerated_noise_matches_applied_perturbation(self):
        # The regenerated stream must reproduce exactly what was added to the
        # params: delta == noise elementwise (fp32 params, so only add rounding).
        model = make_model(torch.float32)
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, hess_init=0.1, noise_seed=3)
        before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        opt._sample_params()
        delta = torch.cat([p.detach().flatten().clone() for p in model.parameters()]) - before
        noise = opt._regenerate_noise()
        assert torch.allclose(delta, noise, atol=1e-6)
        # regeneration is repeatable and side-effect free
        assert torch.equal(noise, opt._regenerate_noise())
        opt._restore_param_average(train=False)

    def test_fast_path_keeps_no_gradient_buffers(self):
        model = make_model()
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, noise_seed=0)
        assert opt._single_sample
        x = torch.randn(8, 16)
        with opt.sampled_params(train=True):
            model(x).square().mean().backward()
        # after the train restore, before step: nothing buffered
        assert opt.state["avg_grad"] is None and opt.state["avg_nxg"] is None
        assert "noise" not in opt.state
        opt.step()
        assert opt.state["count"] == 0 and opt._gen_state is None

    @pytest.mark.parametrize("hess_approx", ["price", "gradsq"])
    def test_fast_path_equals_buffered_single_sample(self, hess_approx):
        # With exactly one MC sample the buffered Welford path and the fast
        # path must produce identical updates (same seed -> same noise stream).
        kw = dict(lr=1e-3, ess=1e4, hess_init=0.1, weight_decay=1e-3,
                  hess_approx=hess_approx, noise_seed=11)
        m_fast, m_buf = make_model(seed=2), make_model(seed=2)
        o_fast = IVON(m_fast.parameters(), train_mc_samples=1, **kw)
        o_buf = IVON(m_buf.parameters(), train_mc_samples=2, **kw)
        assert o_fast._single_sample and not o_buf._single_sample
        for step_seed in (100, 101):
            fake_step(m_fast, o_fast, seed=step_seed)
            fake_step(m_buf, o_buf, seed=step_seed)
        for pf, pb in zip(m_fast.parameters(), m_buf.parameters()):
            assert torch.allclose(pf, pb, atol=1e-7), "fast and buffered paths diverged"
        assert torch.allclose(o_fast.param_groups[0]["hess"], o_buf.param_groups[0]["hess"], atol=1e-7)
        assert torch.allclose(o_fast.param_groups[0]["momentum"], o_buf.param_groups[0]["momentum"], atol=1e-7)

    def test_buffered_welford_matches_mean(self):
        # Two MC samples: avg_grad must equal the mean of the two (cumulative,
        # since grads are not zeroed between samples) gradient snapshots.
        model = make_model()
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, noise_seed=0, train_mc_samples=2)
        snaps = []
        for seed in (5, 6):
            torch.manual_seed(seed)
            x = torch.randn(8, 16)
            with opt.sampled_params(train=True):
                model(x).square().mean().backward()
            snaps.append(torch.cat([p.grad.flatten().clone() for p in model.parameters()]).float())
        expected = (snaps[0] + snaps[1]) / 2
        assert torch.allclose(opt.state["avg_grad"], expected, atol=1e-6)
        opt.step()
        assert opt.state["avg_grad"] is None, "accumulators must be freed at step"

    def test_fast_path_abandons_pending_sample_on_resample(self):
        # Skipped-step flow: restore(train=True) happened but step() never ran
        # (non-finite grads); the next sample must drop the stale pending count.
        model = make_model()
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, noise_seed=0)
        x = torch.randn(8, 16)
        with opt.sampled_params(train=True):
            model(x).square().mean().backward()
        assert opt.state["count"] == 1
        opt.zero_grad()  # what dp_actor does on the skip
        with opt.sampled_params(train=True):
            model(x).square().mean().backward()
        assert opt.state["count"] == 1
        opt.step()  # must not raise


class TestOffloadHelpers:
    def test_ivon_buffers_covered_by_offload(self):
        from verl.utils.fsdp_utils import _move_optimizer_state

        model = make_model()
        opt = IVON(model.parameters(), lr=1e-3, ess=1e4, noise_seed=0)
        fake_step(model, opt)

        moved = {}
        orig_to = torch.Tensor.to

        def spy_to(self, *args, **kwargs):
            moved[id(self)] = True
            return orig_to(self, *args, **kwargs)

        # count how many IVON tensors the helper touches
        targets = [opt.param_groups[0]["momentum"], opt.param_groups[0]["hess"]]
        target_ids = {id(t) for t in targets}
        torch.Tensor.to = spy_to
        try:
            _move_optimizer_state(opt, "cpu")
        finally:
            torch.Tensor.to = orig_to
        assert target_ids <= set(moved), "offload helper missed IVON group-level buffers"

    def test_adamw_still_offloaded(self):
        from verl.utils.fsdp_utils import _move_optimizer_state

        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(4, 16)).sum().backward()
        opt.step()
        state_tensors = [
            v for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor)
        ]
        assert state_tensors
        _move_optimizer_state(opt, "cpu")  # must not raise, tensors stay on cpu
        for t in [v for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor)]:
            assert t.device.type == "cpu"


class TestIsotropicNoise:
    """Variant-B ablation: frozen Hessian + scaled-isotropic parameter noise."""

    def test_hess_frozen(self):
        # With isotropic_noise the Hessian must never move (curvature disabled).
        model = make_model()
        opt = IVON(
            model.parameters(), lr=1e-3, ess=1e4, hess_init=0.1, beta2=0.99999,
            noise_seed=0, isotropic_noise=True,
        )
        before = opt.param_groups[0]["hess"].clone()
        for _ in range(3):
            fake_step(model, opt)
        assert torch.equal(opt.param_groups[0]["hess"], before), "hess moved in isotropic mode"
        # curvature stats are never accumulated
        assert opt.state["avg_nxg"] is None and opt.state["avg_gsq"] is None

    def test_noise_scale_is_isotropic_and_ignores_hess(self):
        # The injected noise uses a single scalar derived from hess_init and must
        # ignore any per-element structure in group["hess"].
        import math

        model = make_model()
        ess, hess_init, wd = 1e4, 0.1, 0.0
        opt = IVON(
            model.parameters(), lr=1e-3, ess=ess, hess_init=hess_init,
            weight_decay=wd, noise_seed=0, isotropic_noise=True,
        )
        # Poke a wildly anisotropic per-element hess; isotropic noise must not care.
        opt.param_groups[0]["hess"].copy_(
            torch.rand_like(opt.param_groups[0]["hess"]) * 100.0 + 1.0
        )
        opt._sample_params()
        noise = opt._regenerate_noise()
        opt._restore_param_average(train=False)
        expected = 1.0 / math.sqrt(ess * (hess_init + wd))
        # empirical std over all elements tracks the scalar target
        assert abs(noise.std().item() - expected) / expected < 0.1
        # ...and the largest element is nowhere near a per-element-hess draw would give
        assert noise.abs().max().item() < 6 * expected

    def test_curvature_mode_still_learns_hess(self):
        # Sanity: default (isotropic_noise=False) still updates the Hessian.
        model = make_model()
        opt = IVON(
            model.parameters(), lr=1e-3, ess=1e4, hess_init=0.1, beta2=0.99999, noise_seed=0
        )
        before = opt.param_groups[0]["hess"].clone()
        for _ in range(3):
            fake_step(model, opt)
        assert (opt.param_groups[0]["hess"] - before).abs().max() > 0


class TestGuards:
    def test_sync_without_dtensor_ok(self):
        model = make_model()
        IVON(model.parameters(), lr=1e-3, ess=1e4, sync=True, noise_seed=0)

    def test_invalid_train_mc_samples(self):
        model = make_model()
        with pytest.raises(ValueError):
            IVON(model.parameters(), lr=1e-3, ess=1e4, train_mc_samples=0)

    def test_m3po_and_decoupled_mutually_exclusive(self):
        # verl must refuse to build an IVON optimizer with both MC variants on.
        from omegaconf import OmegaConf

        from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer

        model = make_model()
        cfg = FSDPOptimizerConfig(
            lr=1e-3,
            optimizer="IVON",
            optimizer_impl="ivon",
            ivon_config=OmegaConf.create(
                {
                    "ess": 1e4,
                    "hess_init": 0.1,
                    "hess_approx": "price",
                    "clip_radius": 1e3,
                    "sync": False,
                    "debias": True,
                    "rescale_lr": True,
                    "noise_seed": 0,
                    "m3po_m": 2,
                    "decoupled_mc_samples": 2,
                }
            ),
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_optimizer(model.parameters(), cfg)

    def test_train_mc_samples_derived_from_variant_knobs(self):
        from omegaconf import OmegaConf

        from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer

        for m3po_m, decoupled, expected in [(1, 1, 1), (4, 1, 4), (1, 3, 3)]:
            model = make_model()
            cfg = FSDPOptimizerConfig(
                lr=1e-3,
                optimizer="IVON",
                optimizer_impl="ivon",
                ivon_config=OmegaConf.create(
                    {
                        "ess": 1e4,
                        "hess_init": 0.1,
                        "hess_approx": "price",
                        "clip_radius": 1e3,
                        "sync": False,
                        "debias": True,
                        "rescale_lr": True,
                        "noise_seed": 0,
                        "m3po_m": m3po_m,
                        "decoupled_mc_samples": decoupled,
                    }
                ),
            )
            opt = build_optimizer(model.parameters(), cfg)
            assert opt.train_mc_samples == expected
            assert opt._single_sample == (expected == 1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
