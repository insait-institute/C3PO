# Adaptive-temperature baseline (AdamW entropy-collapse guard)

A simple, optimizer-agnostic baseline for the entropy-collapse problem that the
Bayesian methods (B3PO/M3PO/C3PO with IVON) address by injecting posterior noise.
Instead of a Bayesian posterior, this baseline uses vanilla **AdamW** and a
feedback controller on the **rollout sampling temperature**: when the policy
entropy collapses to a fraction of its initial value, the temperature is raised
to re-inject exploration.

## Knobs

| Symbol           | Config key                              | Meaning                                                        |
| ---------------- | --------------------------------------- | -------------------------------------------------------------- |
| `TEMP_HIGH`      | `actor_rollout_ref.rollout.temp_high`   | Sampling temperature to switch to once entropy has collapsed.  |
| `LOW_ENT_RATIO`  | `actor_rollout_ref.rollout.low_ent_ratio` | Collapse threshold as a fraction of the reference entropy `H_0`. |
| (enable)         | `actor_rollout_ref.rollout.adaptive_temperature` | Turns the guard on (`bool`).                          |
| `T_base`         | `actor_rollout_ref.rollout.temperature` | Base sampling temperature used before any collapse.            |

In `verl/recipes/run_rl.sh` these are exposed as the env vars `TEMP_HIGH`,
`LOW_ENT_RATIO`, `ADAPTIVE_TEMP` (and the existing `TEMPERATURE` for `T_base`),
or activated in one shot with `METHOD=grpo_adaptivetemp`.

## Math

Let $H_t$ be the mean token-level policy entropy measured at optimizer step $t$,

$$
H_t \;=\; \operatorname*{\mathbb{E}}_{s \sim \mathcal{D}_t}\!\left[
  -\sum_{a} \pi_{\theta_t}(a \mid s)\,\log \pi_{\theta_t}(a \mid s)
\right],
$$

aggregated over the response tokens with the actor's `loss_agg_mode`
(this is the `actor/entropy` metric).

Define the **reference entropy** as the first measured value,

$$
H_0 \;\triangleq\; H_{t_0}, \qquad t_0 = \text{first training step},
$$

and the **collapse indicator**

$$
c_t \;=\; \mathbb{1}\!\left[\, H_t < \texttt{LOW\_ENT\_RATIO}\cdot H_0 \,\right].
$$

The rollout temperature follows a **one-way latch** (hysteresis): it starts at
the base temperature and, the first time collapse is detected, jumps to
`TEMP_HIGH` and stays there:

$$
\tau_{t+1} \;=\;
\begin{cases}
\texttt{TEMP\_HIGH}, & \text{if } b_t = 1,\\[4pt]
T_{\text{base}},     & \text{otherwise,}
\end{cases}
\qquad
b_t \;=\; \max\!\big(b_{t-1},\, c_t\big),\quad b_{t_0-1}=0.
$$

Sampling at step $t{+}1$ then draws from the tempered policy

$$
\pi^{(\tau)}_{\theta}(a \mid s) \;=\;
\frac{\exp\!\big(z_\theta(a\mid s)/\tau_{t+1}\big)}
     {\sum_{a'} \exp\!\big(z_\theta(a'\mid s)/\tau_{t+1}\big)},
$$

where $z_\theta$ are the pre-softmax logits. The same $\tau_{t+1}$ is applied
when recomputing log-probs and in the actor update (logits are divided by
$\tau$), so importance-sampling weights and the entropy term all use a single,
consistent temperature.

### Notes on the latch

The one-way latch avoids the oscillation a memoryless rule
($\tau_t = \texttt{TEMP\_HIGH}\cdot c_t + T_{\text{base}}\cdot(1-c_t)$) would
produce: raising $\tau$ recovers entropy, which would clear $c_t$, drop $\tau$,
and let entropy collapse again. Latching on $b_t = \max(b_{t-1}, c_t)$ makes the
switch monotone. Because entropy is measured *after* the step-$t$ rollout, the
detection at step $t$ takes effect on the step-$(t{+}1)$ rollout (one-step
feedback delay).

## Where it lives in the code

- Config fields: `verl/verl/workers/config/rollout.py`
  (`adaptive_temperature`, `temp_high`, `low_ent_ratio`) and the mirror
  `verl/verl/trainer/config/rollout/rollout.yaml`.
- Controller state + update: `RayPPOTrainer.fit` in
  `verl/verl/trainer/ppo/ray_trainer.py` — `H_0` = `self._entropy_ref`,
  latch = `self._temp_bumped`, current temperature = `self._rollout_temperature`.
- Rollout honors the per-batch temperature in
  `verl/verl/experimental/agent_loop/agent_loop.py`.
- Recipe knobs: `verl/recipes/run_rl.sh`.

Logged metrics (when enabled): `actor/entropy_ref` ($H_0$),
`actor/rollout_temperature` ($\tau$), `actor/temp_high_active` ($b_t$).
