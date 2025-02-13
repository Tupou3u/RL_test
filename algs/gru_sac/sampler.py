from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.samplers import Sampler
from typing import Any, Dict, Tuple, Union
import torch

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."

class RNNSampler(Sampler):
    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        index = storage._rand_seq_given_ndim(batch_size)
        return index, {}

    def _empty(self):
        pass

    def dumps(self, path):
        # no op
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return