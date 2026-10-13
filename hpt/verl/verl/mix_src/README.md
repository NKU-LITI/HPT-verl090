# HPT 0.1 source islands

This directory deliberately keeps algorithm-critical verl 0.1 code visible
inside the 0.9 tree.  The files are executable, not dead reference copies.

| verl 0.1 source | verl 0.9 direct source | 0.9 integration adapter |
| --- | --- | --- |
| `trainer/ppo/core_algos.py::compute_policy_loss` | `legacy_policy_loss.py` | `experimental/hpt/loss.py` |
| `mix_src/mix_core_alg.py::compute_sft_pure_loss` | `legacy_policy_loss.py` | `experimental/hpt/loss.py` |
| `mix_src/mix_fsdp_worker.py` + `mix_actor.py::update_policy/_optimizer_step` | `legacy_actor_update.py` | `experimental/hpt/actor_update.py`, `trainer/ppo/ray_trainer.py`, `workers/engine/fsdp/transformer_impl.py` |
| `utils/torch_functional.py::get_constant_schedule_with_warmup` | `legacy_actor_update.py::legacy_constant_schedule_scale` | launchers use one explicit 0.9 warmup step |
| verl-0.7 `experimental/agent_loop/agent_loop.py` (8 workers, no request priority) | `legacy_rollout.py` | `experimental/agent_loop/agent_loop.py` |
| `mix_src/rl_dataset_with_target.py::ResumableRandomSampler` | `legacy_sampler.py` | `experimental/hpt/dataset.py` |
| `mix_src/entropy_math/` | unchanged algorithm body in `experimental/hpt/legacy_entropy_math.py` | `experimental/hpt/reward.py`, process-main-thread boundary in `experimental/hpt/reward_manager_origin.py` |

The full old trainer/worker/rollout files are not imported into 0.9 because
their Ray, DataProto, FSDP, and vLLM interfaces no longer exist.  The boundary
adaptation stays in `experimental/hpt/`, while the formulas and sampler state
machine stay here for line-by-line comparison.

The canonical 0.1 snapshot is available at
`reference_verl010/Unify-Post-Training/hpt/verl/verl/` from the repository
root.  In particular, compare `mix_actor.py` with the loss helper above and
`mix_trainer.py` with `experimental/hpt/algorithm.py`.

## Runtime boundary that cannot be copied

The 0.1 recipe used PyTorch 2.4, Transformers 4.46, flash-attn 2.7.3, and
vLLM 0.6 with `VLLM_ATTENTION_BACKEND=XFORMERS`.  The current 0.9 runtime uses
PyTorch 2.11, Transformers 5.9, flash-attn 2.8.3, and vLLM 0.24.  vLLM 0.24
removed XFORMERS, so rollout sampling cannot be made bitwise identical by a
source-level HPT adapter.  It also ignores the old environment variable.  The
launchers leave vLLM's backend on automatic selection: forcing FLASH_ATTN gave
a step-1 solved rate of 0.2637 versus roughly 0.28--0.29 on automatic selection,
while forcing FLASHINFER caused a much larger actor/rollout log-prob mismatch
(mean 0.178 versus roughly 0.011) and a solved rate of 0.0586.  Neither forced
backend is treated as a compatibility fix.

The actor-side entropy implementation is nevertheless source-compatible:
verl 0.1 compiled `entropy_from_logits` with `torch.compile(dynamic=True)`, so
the migrated launchers must leave `fsdp_config.use_torch_compile=True`.

The cross-runtime boundary is measured, not assumed.  On the same fixed batch,
the current 0.9 HPT engine running under the 0.7 runtime (PyTorch 2.8 and
Transformers 4.56) reproduces the 0.1 current log-probabilities exactly and its
loss components exactly.  Conversely, the direct 0.1 actor source running under
the current PyTorch 2.11 runtime reproduces the 0.9 forward discrepancy
(log-prob relative L2 about 2.00%).  Thus copying more HPT Python statements
cannot remove that remaining training-kernel difference.
