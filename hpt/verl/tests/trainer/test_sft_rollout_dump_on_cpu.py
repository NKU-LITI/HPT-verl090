import json
from types import SimpleNamespace

import torch

from verl.trainer.sft_trainer import SFTTrainer


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _SingleRankEngine:
    def is_mp_src_rank_with_outputs(self):
        return True

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_group(self):
        return None

    def get_data_parallel_rank(self):
        return 0


def test_sft_trainer_dumps_supervised_batch_in_rollout_format(tmp_path):
    trainer = object.__new__(SFTTrainer)
    trainer.config = SimpleNamespace(trainer={"rollout_data_dir": str(tmp_path)})
    trainer.model_config = SimpleNamespace(tokenizer=_Tokenizer())
    trainer.engine = _SingleRankEngine()
    trainer.loss_type = "dft"
    data = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
    }

    trainer._dump_supervised_data(data, global_step=7)

    records = [json.loads(line) for line in (tmp_path / "7.jsonl").read_text().splitlines()]
    assert records == [
        {"input": "1 2", "output": "3 4", "gts": "3 4", "score": None, "step": 7, "loss_type": "dft"},
        {"input": "5", "output": "6 7", "gts": "6 7", "score": None, "step": 7, "loss_type": "dft"},
    ]
