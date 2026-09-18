# Hybrid Post-Training on verl 0.9.0

This recipe ports the HPT switch algorithm from the original
`Unify-Post-Training/verl/mix_src` implementation without copying the verl
0.1.0 runtime into 0.9.0.

The training parquet must contain the standard verl fields (`prompt`,
`data_source`, and `reward_model.ground_truth`) plus `target`. `target` may be a
string, an assistant message dict, or the original one-element list of
assistant message dicts.

Run from the verl repository root:

```bash
export MODEL_PATH=/path/to/Qwen2.5-Math-7B
export TRAIN_FILE=/path/to/openr1.parquet
export VAL_FILES="['/path/to/aime24.parquet','/path/to/math500.parquet']"
bash examples/hpt_trainer/run_hpt.sh
```

The GRPO comparison baseline defaults to the three switches used by the HPT
experiment scripts: unclipped ratio loss, token-sum divided by the fixed
response horizon, and no advantage standard-deviation normalization:

```bash
bash examples/hpt_trainer/run_grpo.sh
```

They can be independently restored to stock verl behavior with
`LOSS_REMOVE_CLIP=false`, `LOSS_REMOVE_TOKEN_MEAN=false`, and
`NORM_ADV_BY_STD=true`.

The other two baselines shipped with the original repository use the same
migrated target batch path and select a different target objective:

```bash
bash examples/hpt_trainer/run_luffy.sh  # mixed on/off-policy objective
bash examples/hpt_trainer/run_srft.sh   # entropy-weighted SRFT objective
```

The migrated implementation currently supports text-only, single-turn,
critic-free GRPO. It intentionally rejects rollout-correction bypass mode and
multimodal inputs because demonstration trajectories do not have rollout-policy
metadata for those modes.
