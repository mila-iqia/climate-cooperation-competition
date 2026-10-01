"""Probabilistic Logic Shield (PLS) for club mitigation compliance.

Implements the policy-level shield of Yang et al. (2023, IJCAI), Definition 3.1:

    pi+(a|s) = P(safe|a,s) * pi(a|s) / P_pi(safe|s)

The safety "program" here is a single compliance predicate
``safe :- mitigation_rate >= club_min``, so its compiled circuit reduces to a
per-level weight vector computed directly in JAX (no ProbLog needed).

The shield reuses the env's action-mask channel: the env emits continuous
safety weights P(safe|a,s) in place of the 0/1 mitigation mask, and
:class:`ShieldedCategoricalLayer` applies ``logits + log(w)`` — which after the
softmax is exactly the renormalized shielded policy above. For 0/1 masks this
degenerates to standard action masking, so the layer is a drop-in replacement
for all discrete heads.
"""

from __future__ import annotations

from typing import Literal

import chex
import distrax
import jax
import jax.numpy as jnp

# Private module path: jaxnasium is a pinned git dependency.
from jaxnasium.algorithms.core._input_output import (
    CategoricalLayer,
    _forward_per_dimension,
)

# log(1e-8) ~ -18.4: weight-0 actions get ~1e-8 probability (soft equivalent
# of the stock -1e9 hard mask; keeps log-probs finite for PPO).
_MIN_WEIGHT = 1e-8

ShieldMode = Literal["constant", "graded"]


def mitigation_safety_weights(
    club_min: chex.Array,
    membership: chex.Array,
    num_levels: int,
    shield_strength: float = 0.8,
    mode: ShieldMode = "constant",
    kappa: float = 10.0,
) -> chex.Array:
    """Per-region safety weights P(safe|a) over mitigation levels, shape (NR, L).

    Non-members get all-ones (unshielded). For members, levels below the club
    floor are downweighted but never zeroed, so defection stays possible:

    * ``constant``: P(safe|a) = 1 if a/L >= club_min else 1 - shield_strength
    * ``graded``:   P(safe|a) = exp(-kappa * max(0, club_min - a/L))

    ``shield_strength=0`` (or ``kappa=0``) is the canonical null: all-ones.
    """
    levels = jnp.arange(num_levels) / num_levels  # level -> realized rate a/L
    shortfall = jnp.maximum(0.0, club_min - levels)  # (L,)
    if mode == "constant":
        member_w = jnp.where(shortfall > 0.0, 1.0 - shield_strength, 1.0)
    elif mode == "graded":
        member_w = jnp.exp(-kappa * shortfall)
    else:
        raise ValueError(f"Unknown shield mode: {mode}")

    membership = membership.astype(jnp.float32)
    ones = jnp.ones_like(member_w)
    return membership[:, None] * member_w[None, :] + (1.0 - membership)[:, None] * ones


class ShieldedCategoricalLayer(CategoricalLayer):
    """CategoricalLayer whose mask channel carries safety weights in (0, 1].

    ``logits + log(w)`` renormalizes to pi+ ∝ P(safe|a) * pi (PLS Def. 3.1);
    the gradient flows through the base logits only, so PPO on this layer is
    the on-policy shielded policy gradient (Yang et al. 2023, Eq. 5).
    """

    def __call__(self, x, action_mask=None, *, key=None):
        logits = _forward_per_dimension(self.layers, x, key=key)
        if action_mask is not None:
            logits = jax.tree.map(
                lambda l, w: l + jnp.log(jnp.clip(w, _MIN_WEIGHT, 1.0)),
                logits,
                action_mask,
            )
        return distrax.Categorical(logits=logits, dtype=self.dtype)
