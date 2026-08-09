"""EnsemblePolicy — probability-averaging over N independently-seeded specialists.

Each member is a :class:`~ptcg_il.model.policy.Policy` trained on the same deck
archetype with a different ``--seed``, producing decorrelated policies whose
averaged probabilities outperform any single member.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model.policy import Policy, load_policy_state, policy_from_config

logger = logging.getLogger(__name__)


class EnsemblePolicy(nn.Module):
    """Average predictions from N independently-seeded specialist Policies.

    Members share vocab/archetypes/deck but differ in init and batch order
    (``--seed``), producing decorrelated policies.

    Parameters
    ----------
    members : nn.ModuleList[Policy]
        Pre-built Policy instances with weights already loaded.
    """

    def __init__(self, members: nn.ModuleList):
        super().__init__()
        self.members = members

    @classmethod
    def from_checkpoints(
        cls,
        paths: list[str],
        all_card_feat: torch.Tensor | None = None,
        all_attack_feat: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> EnsemblePolicy:
        """Build an EnsemblePolicy from a list of checkpoint paths.

        For each path:
        1. Load checkpoint, auto-detect EMA shadow vs raw state dict.
        2. Build Policy from the checkpoint's own ``config``.
        3. Load weights via ``load_policy_state``.
        4. Verify shared decklist across members (fatal on mismatch).
           Warn on vocab_sha1/archetypes_sha1 mismatch (non-fatal — card
           identity routes through static features, and the ensemble is
           greedy-only so archetype labels are not read).

        Heterogeneous architectures (different D/layers/heads/ff) are allowed —
        the encode path is per-member, so shapes never mix.
        """
        if len(paths) < 1:
            raise ValueError("Need at least one checkpoint path")

        members = nn.ModuleList()
        ref_deck = None
        ref_vocab = None
        ref_archetypes = None

        for i, path in enumerate(paths):
            ckpt = torch.load(path, map_location=device, weights_only=False)

            # Auto-detect EMA shadow vs raw state dict.
            # Mirrors build_model_weights (build_submission.py:847-855).
            ema = ckpt.get("ema_state_dict")
            if ema is not None and "shadow" in ema:
                state_dict = ema["shadow"]
                logger.info("Member %d: using EMA shadow weights from %s", i, path)
            else:
                state_dict = ckpt.get("model_state_dict")
                if state_dict is None:
                    raise KeyError(
                        f"Checkpoint {path} missing both 'ema_state_dict' and "
                        "'model_state_dict'"
                    )
                logger.info("Member %d: using raw model_state_dict from %s", i, path)

            # Build policy from checkpoint's own config
            cfg = ckpt.get("config") or {}
            member = policy_from_config(cfg, all_card_feat=all_card_feat,
                                        all_attack_feat=all_attack_feat)
            load_policy_state(member, state_dict)
            member.to(device)
            member.eval()

            # Assert shared artifacts across members.
            # Without this assertion, you can ensemble models trained against
            # different vocabularies and get a bundle that runs fine and loses
            # almost everything.
            deck_record = ckpt.get("deck") or {}
            vocab_sha1 = deck_record.get("vocab_sha1")
            arch_sha1 = deck_record.get("archetypes_sha1")
            decklist = deck_record.get("deck")

            if i == 0:
                ref_deck = decklist
                ref_vocab = vocab_sha1
                ref_archetypes = arch_sha1
            else:
                if decklist != ref_deck:
                    raise ValueError(
                        f"Member {i} decklist differs from member 0. "
                        f"Member 0: {ref_deck[:5] if ref_deck else None}..., "
                        f"member {i}: {decklist[:5] if decklist else None}... "
                        "Ensembling models trained on different decks is a silent "
                        "catastrophe — the policy sees cards it never trained on."
                    )
                if vocab_sha1 != ref_vocab:
                    logger.warning(
                        "Member %d vocab_sha1 (%s) differs from member 0 (%s). "
                        "This is evidence but not fatal — card identity routes "
                        "through static features, not vocab indices.",
                        i, vocab_sha1, ref_vocab,
                    )
                if arch_sha1 != ref_archetypes:
                    logger.warning(
                        "Member %d archetypes_sha1 (%s) differs from member 0 (%s). "
                        "Ensemble is greedy-only so archetypes are not read, but "
                        "this means the checkpoints came from different mining runs.",
                        i, arch_sha1, ref_archetypes,
                    )

            members.append(member)

        logger.info(
            "EnsemblePolicy: %d members loaded, vocab=%s, %d distinct cards",
            len(members), ref_vocab,
            len(ref_deck) if ref_deck else 0,
        )
        return cls(members)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        history_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensemble forward: average per-member probabilities, return as logits.

        Returns
        -------
        logits : Tensor[B, O_MAX]
            Log of mean probabilities (so argmax is unchanged).
        value : Tensor[B]
            Mean value across members.
        history_h : Tensor[B, D]
            Cross-turn hidden state from the first member only.
        """
        all_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        first_history: torch.Tensor | None = None
        opt_mask = x.get("opt_mask")

        for i, member in enumerate(self.members):
            logits_i, value_i, hist_i = member(x, history_h)
            # Mask padding options then softmax
            if opt_mask is not None:
                logits_i = logits_i.masked_fill(~opt_mask, -1e9)
            probs_i = F.softmax(logits_i, dim=-1)
            all_probs.append(probs_i)
            all_values.append(value_i)
            if i == 0:
                first_history = hist_i

        # Average probabilities → log (so argmax is unchanged)
        mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        logits = torch.log(mean_probs + 1e-10)

        # Mask padding options with -1e9 (softmax → mean → log leaves ~ -23
        # on padding, which is not the hard mask callers expect).
        opt_mask = x.get("opt_mask")
        if opt_mask is not None:
            logits = logits.masked_fill(~opt_mask, -1e9)

        # Average values
        value = torch.stack(all_values, dim=0).mean(dim=0)

        return logits, value, first_history

    def select_multi(
        self,
        x: dict[str, torch.Tensor],
        history_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ensemble greedy AR multi-select inference.

        Each member pre-encodes independently, then per step:
        1. Each member produces logits via its own pointer + msgru state.
        2. Probabilities are averaged → argmax.
        3. Each member's msgru advances with its own option representation
           at the ensemble-chosen index.

        Members condition on the joint decision — this is what makes it an
        ensemble rather than N independent agents.
        """
        from ptcg_il.model.policy import _select_multi_raw

        # Pre-encode each member
        pointers: list = []
        h_list: list[torch.Tensor] = []
        xs: list[dict] = []
        for member in self.members:
            x_i, h_i, _hist_i = member._encode(x, history_h)
            pointers.append(member.pointer)
            h_list.append(h_i)
            xs.append(x_i)
        # Members share one corpus and therefore one static table, so every
        # x_i is the same gather; the pointer call below takes the first.
        x = xs[0]

        stop_col = x.get("stop_column")
        return _select_multi_raw(
            self.members[0].pointer,
            h_list[0],
            x["tok_mask"],
            self.members[0].embed.card,
            x,
            minC=x["minCount"],
            maxC=x["maxCount"],
            stop_column=stop_col,
            pointers=pointers,
            h_list=h_list,
        )

    def belief_logits(
        self, x: dict[str, torch.Tensor], history_h: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Delegate to first member only. Belief is only consumed by MCTS;
        the ensemble is greedy-only."""
        return self.members[0].belief_logits(x, history_h)

    def forward_with_belief(
        self, x: dict[str, torch.Tensor], history_h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forward + belief, delegating belief to the first member."""
        logits, value, history_h_out = self(x, history_h)
        belief = self.members[0].belief_logits(x, history_h)
        return logits, value, history_h_out, belief
