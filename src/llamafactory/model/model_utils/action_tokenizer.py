"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

import json
from functools import partial
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pose_lower_bound: List[float],
        pose_upper_bound: List[float],
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        use_extra: bool = True,
        action_template: str = "<|action_%d|>",
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        :param use_extra: Use the extra tokens (not just the last ones), only implemented for Qwen2
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action
        self.pose_lower_bound = np.array(pose_lower_bound, dtype=np.float32)
        self.pose_upper_bound = np.array(pose_upper_bound, dtype=np.float32)

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError("Cannot use extra tokens for this tokenizer!")

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        assert action.min() >= self.min_action and action.max() <= self.max_action, f"Action out of bounds: {action.min()} <= {action.max()}"
        discretized_action = np.digitize(action, self.bins)
        
        assert discretized_action.min() >= 1 and discretized_action.max() <= self.n_bins, f"Discretized action out of bounds: {discretized_action.min()} <= {discretized_action.max()}"

        # Handle single element vs. batch
        if len(discretized_action.shape) <= 1:
            return self.tokenizer.decode(list(self.tokenizer_len - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer_len - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer_len - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]
    
    def action_to_str(self, action: np.ndarray) -> str:
        assert len(action.shape) == 1, f"Action must be a single element, got {action.shape}."
        
        action = (action - self.pose_lower_bound) / (self.pose_upper_bound - self.pose_lower_bound) * (self.max_action - self.min_action) + self.min_action
        assert action.min() >= -1 and action.max() <= 1, f"Action out of bounds: {action.min()} <= {action.max()}"
        
        return self(action)
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        assert len(action.shape) == 1, f"Action must be a single element, got {action.shape}."
        assert action.min() >= -1 and action.max() <= 1, f"Action out of bounds: {action.min()} <= {action.max()}"
        
        action = (action - self.min_action) / (self.max_action - self.min_action) * (self.pose_upper_bound - self.pose_lower_bound) + self.pose_lower_bound
        return action

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    @property
    def required_future_horizon(self) -> int:
        # the number of future action horizon elements
        return 0