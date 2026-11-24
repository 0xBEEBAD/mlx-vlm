from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import KVCache
from ....models.granite_docling.language import TransformerBlock
from ..mtp_base import AutoregressiveMTPDraftModel
from .config import GraniteDoclingMTPConfig


class GraniteDoclingMTPHead(TransformerBlock):
    """One Granite decoder layer with an input projection and an output norm."""

    def __init__(self, config):
        super().__init__(config)
        dim = config.hidden_size
        self.enorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
        self.post_norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)

    def predict(self, token_embed: mx.array, hidden: mx.array, cache) -> mx.array:
        x = self.eh_proj(
            mx.concatenate([self.enorm(token_embed), self.hnorm(hidden)], axis=-1)
        )
        mask = create_attention_mask(x, cache)
        return self.post_norm(self(x, mask, cache))


@dataclass
class _Sequence:
    """Head caches and chain rows of one sequence."""

    cache: List[KVCache]
    pending: List[Optional[mx.array]]  # Last row of the head before, for each head
    draft_appended: List[int]  # Draft positions in each head cache
    draft_depth: int  # Heads that draft in a commit
    draft_tokens: Optional[mx.array] = None


class GraniteDoclingMTPDraftModel(AutoregressiveMTPDraftModel):
    """Chained MTP heads of Granite for Docling.

    The target gives the normalized hidden state ``h[i]`` of each position and
    the token ``t[i + 1]`` that it predicts. Head ``k`` at position ``i`` reads
    the token ``t[i + k + 1]`` and the output of head ``k - 1`` at position
    ``i`` (``h[i]`` for the first head), and predicts ``t[i + k + 2]``. Each
    head attends to its own earlier positions.

    So for the verified tokens of a round every head reads the same tokens,
    but head ``k`` reads the rows of head ``k - 1`` one position later. Each
    head keeps the last row of the head before it as its pending row.

    Head ``k`` then drafts on ``k`` more positions, with the draft tokens of
    the heads before it. One call of a head does both, so the draft tokens of
    the next round are ready when a round is committed. The draft positions
    are removed from the head caches before the next commit.

    Each sequence of a batch has its own caches, so the sequences can accept
    different numbers of draft tokens.
    """

    requires_uniform_batch_acceptance = False
    supports_left_padded_prefill = True

    def __init__(self, config: GraniteDoclingMTPConfig):
        super().__init__(config)
        text_config = config.text_config
        self.layers = [
            GraniteDoclingMTPHead(text_config)
            for _ in range(text_config.num_mtp_layers)
        ]
        self._sequences: List[_Sequence] = []

    def validate_target_compatibility(self, target_model) -> None:
        lm = getattr(target_model, "language_model", target_model)
        target = getattr(lm, "config", None)
        if getattr(target, "model_type", None) != "granite_docling_text":
            raise ValueError("Granite Docling MTP requires a Granite Docling target.")
        for name in ("hidden_size", "vocab_size"):
            if getattr(target, name) != getattr(self.config.text_config, name):
                raise ValueError(f"Granite Docling MTP target {name} does not match.")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        lm = getattr(target_model, "language_model", target_model)
        self._input_embed = lm.model.embed_tokens
        # The head logits do not have the logits_scaling division of the target.
        # With the division the draft tokens are the same for greedy decoding,
        # but the distribution is too flat to sample from.
        self._lm_head_fn = lm.output_projection
        return self

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def _new_sequence(self) -> _Sequence:
        count = len(self.layers)
        return _Sequence(self.make_cache(), [None] * count, [0] * count, count)

    def _resize(self, batch_size: int) -> None:
        if len(self._sequences) != batch_size:
            self._sequences = [self._new_sequence() for _ in range(batch_size)]

    def reset(self, target_model) -> list:
        super().reset(target_model)
        self._sequences = [self._new_sequence()]
        return self._sequences[0].cache

    def draft_eval_state(self):
        return [
            [c.state for c in seq.cache] + [seq.pending, seq.draft_tokens]
            for seq in self._sequences
        ]

    def _sample(self, hidden: mx.array, sampler, greedy: bool) -> mx.array:
        logits = self._lm_head_fn(hidden)
        return mx.argmax(logits, axis=-1) if greedy else sampler(logits)

    @staticmethod
    def _trim_draft(seq: _Sequence) -> None:
        for cache, appended in zip(seq.cache, seq.draft_appended):
            if appended > 0:
                cache.trim(appended)
        seq.draft_appended = [0] * len(seq.cache)
        seq.draft_tokens = None

    def _commit(self, seq: _Sequence, tokens, hidden, sampler, greedy: bool):
        """Add verified tokens and draft: ``hidden[:, i]`` predicted ``tokens[:, i]``."""
        self._trim_draft(seq)
        token_embed = self._input_embed(tokens)
        rows = hidden.astype(token_embed.dtype)
        draft_tokens: List[mx.array] = []
        draft_rows = None  # Rows that the head before wrote on its draft positions
        for k, head in enumerate(self.layers):
            if k == 0:
                rows = head.predict(token_embed, rows, seq.cache[k])
                draft_tokens.append(self._sample(rows[:, -1:], sampler, greedy))
                continue
            if rows.shape[1] == 0:
                break  # The prompt is shorter than the chain.
            # Head k reads the rows of head k - 1 one position later.
            last = rows[:, -1:]
            if seq.pending[k] is None:
                rows, token_embed = rows[:, :-1], token_embed[:, 1:]
            else:
                rows = mx.concatenate([seq.pending[k], rows[:, :-1]], axis=1)
            seq.pending[k] = last
            if k >= seq.draft_depth:
                rows = head.predict(token_embed, rows, seq.cache[k])
                continue
            # Draft positions: the pending row, then the draft rows of head k - 1.
            parts = [rows, last] if draft_rows is None else [rows, last, draft_rows]
            draft_embed = self._input_embed(mx.concatenate(draft_tokens, axis=1))
            out = head.predict(
                mx.concatenate([token_embed, draft_embed], axis=1),
                mx.concatenate(parts, axis=1),
                seq.cache[k],
            )
            rows, draft_rows = out[:, : rows.shape[1]], out[:, rows.shape[1] :]
            seq.draft_appended[k] = k
            draft_tokens.append(self._sample(draft_rows[:, -1:], sampler, greedy))
        seq.draft_tokens = mx.concatenate(draft_tokens, axis=1)

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        left_padding: Optional[List[int]] = None,
    ) -> None:
        if input_ids.shape[1] == 0:
            return
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=token_dtype)
        else:
            bonus = bonus_token[:, None].astype(token_dtype)
        batch_size = input_ids.shape[0]
        self._resize(batch_size)
        # A prompt that continues a cached prefix has fewer rows than tokens.
        length = min(input_ids.shape[1], hidden.shape[1])
        input_ids = input_ids[:, input_ids.shape[1] - length :].astype(token_dtype)
        hidden = hidden[:, hidden.shape[1] - length :]
        for i, seq in enumerate(self._sequences):
            pad = min(int(left_padding[i]), length - 1) if left_padding else 0
            tokens = mx.concatenate(
                [input_ids[i : i + 1, pad + 1 :], bonus[i : i + 1]], axis=1
            )
            self._commit(seq, tokens, hidden[i : i + 1, pad:], sampler, greedy)

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        self.accept_verified_tokens_batch(
            verify_hidden,
            draft_tokens,
            [accepted],
            [new_tokens],
            sampler,
            token_dtype,
            greedy,
        )

    def accept_verified_tokens_batch(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: List[int],
        new_tokens: List[List[int]],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> None:
        for i, seq in enumerate(self._sequences):
            if not new_tokens[i]:
                # No token budget is left, so this sequence ends here.
                self._trim_draft(seq)
                continue
            count = int(accepted[i])
            bonus = mx.array([[int(new_tokens[i][-1])]], dtype=token_dtype)
            tokens = mx.concatenate(
                [draft_tokens[i : i + 1, :count].astype(token_dtype), bonus], axis=1
            )
            self._commit(
                seq, tokens, verify_hidden[i : i + 1, : count + 1], sampler, greedy
            )

    def filter_batch(self, keep) -> None:
        if isinstance(keep, mx.array):
            keep = keep.tolist()
        self._sequences = [self._sequences[int(i)] for i in keep]

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
    ) -> mx.array:
        del cache
        if self._input_embed is None or self._lm_head_fn is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block()."
            )
        if isinstance(last_bonus, int):
            bonus = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            bonus = last_bonus[:, None].astype(token_dtype)
        self._resize(bonus.shape[0])

        # The longest draft that every sequence has. It is shorter than the
        # chain only when a prompt is shorter than the chain.
        depth = min(max(block_size - 1, 1), len(self.layers))
        rows = []
        for i, seq in enumerate(self._sequences):
            if seq.draft_tokens is None:
                # No prefill: the heads start from the last target position.
                self._commit(
                    seq, bonus[i : i + 1], hidden[i : i + 1, -1:], sampler, greedy
                )
            depth = min(depth, seq.draft_tokens.shape[1])
            rows.append(seq.draft_tokens)
            seq.draft_tokens = None
            # A smaller block needs fewer draft tokens from the next commit.
            seq.draft_depth = min(max(block_size - 1, 1), len(self.layers))
        self._draft_round += 1
        return mx.concatenate([row[:, :depth] for row in rows]).astype(token_dtype)

    def sanitize(self, weights):
        return {key.removeprefix("mtp."): value for key, value in weights.items()}
