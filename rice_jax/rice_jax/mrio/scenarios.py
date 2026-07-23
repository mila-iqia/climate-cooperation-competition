"""MRIO/CBAM endogenous-club scenario subclasses."""

from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxnasium import Discrete, MultiDiscrete

from .env import RiceMRIO
from ..utils import i_to_agent_str

class _MRIOClubBase(RiceMRIO):
    """Base class for single-club MRIO negotiation scenarios.

    A single open-accession club is anchored on ``eu_region_idx`` (always a
    member).  Each stage:

    - **propose**  : every region announces a candidate club mitigation rate
      in ``state["proposals"]`` (the anchor's entry is the club's offer).
    - **evaluate** : every region votes accept/reject on the anchor's offer;
      accepters join the club.  Members are bound to mitigate at least the
      club rate via ``minimum_mitigation_rate_all_regions``.
    - **climate**  : RiceMRIO step runs; ``_postprocess_cbam`` levies CBAM on
      non-members' exports *to members*, using the club rate as the reference
      MAC (differential-CBAM gap, Nordhaus 2017 MAC curve).

    Literature: Nordhaus (2015) climate-club accession dynamics; differential
    CBAM rate = (MAC_club − MAC_r)/MAC_club (EU CBAM Reg. 2023/956 Art. 9).
    """

    # negotiation must be on for the 3-stage propose/evaluate/climate cycle
    negotiation_on: bool = True
    # clubs need the MAC-gap CBAM and the additive (RCPO) penalty channel
    cbam_tariff_mode: str = eqx.field(static=True, default="differential")
    reward_mode: str = eqx.field(static=True, default="additive_cbam")
    cbam_lambda_init: float = eqx.field(static=True, default=1.0)

    # ------------------------------------------------------------------ state
    def _get_initial_state(self, key) -> dict:
        state = super()._get_initial_state(key)
        N = self.num_regions
        # Drop the unused promise/ask channels from base negotiation.
        state.pop("promised_mitigation_rate", None)
        state.pop("requested_mitigation_rate", None)
        # Candidate club rate announced by every region (anchor's entry is used).
        state["proposals"] = jnp.zeros(N, dtype=jnp.float32)
        # Boolean membership mask; anchor is always a member.
        state["club_membership"] = (
            jnp.zeros(N, dtype=jnp.bool_).at[self.eu_region_idx].set(True)
        )
        # Current binding club mitigation rate (scalar for single club).
        state["club_mitigation_rate"] = jnp.float32(0.0)
        return state

    # ------------------------------------------------------------------ stages
    def step_propose(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        state = state.copy()
        proposals = actions["proposal"]  # (N,) in [0, 1) after process_actions
        # Credible-anchor commitment: when an EU mitigation schedule is given
        # (EU Climate Law 2021/1119 net-zero ramp), pin the EU core's club offer
        # to the scheduled rate so the club's reference MAC cannot self-collapse
        # to a low-ambition equilibrium (the learned-proposal failure mode).
        # Capped at (D-1)/D so the binding mitigation floor stays expressible on
        # the discrete grid (level D does not exist, so a 1.0 floor is infeasible).
        if self.eu_mitigation_schedule is not None:
            D = self.num_discrete_action_levels
            sched = jnp.array(self.eu_mitigation_schedule, dtype=jnp.float32)
            t_idx = jnp.clip(
                jnp.int32(state["activity_timestep"]), 0, sched.shape[0] - 1
            )
            eu_rate = jnp.minimum(sched[t_idx], (D - 1.0) / D)
            proposals = proposals.at[self.eu_region_idx].set(eu_rate)
        state["proposals"] = proposals
        return state

    def step_evaluate_proposals(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        state = state.copy()
        proposals = state["proposals"]                              # (N,)
        club_rate = proposals[self.eu_region_idx]                   # scalar — club offer
        # decisions[decider, target]; we read the column voting on the anchor.
        decisions = actions["proposal_decisions"] > 0              # (N, N) bool
        joins = decisions[:, self.eu_region_idx]                   # (N,) bool
        membership = joins.at[self.eu_region_idx].set(True)        # anchor always in
        # Members are bound to the club rate; non-members keep floor 0.
        mmr = jnp.where(membership, club_rate, 0.0)                # (N,)
        state["club_membership"] = membership
        state["club_mitigation_rate"] = club_rate
        state["minimum_mitigation_rate_all_regions"] = mmr
        return state

    # ------------------------------------------------------------------ CBAM
    def _postprocess_cbam(
        self, state, trade_flows, gross_imports_mrio, mitigation_rates,
        cbam_tariff_matrix, cbam_revenue, cbam_cost_raw,
    ):
        """Levy CBAM on non-members' exports to club members.

        Singleton club {anchor} with club_rate = μ_anchor reproduces the base
        RiceMRIO single-EU differential CBAM exactly (canonical null condition).
        """
        N = self.num_regions
        t = state["activity_timestep"]
        mu = (
            mitigation_rates
            if mitigation_rates is not None
            else state["mitigation_rates_all_regions"]
        )
        membership = state["club_membership"]                       # (N,) bool
        club_rate = state["club_mitigation_rate"]                   # scalar
        member_f = membership.astype(jnp.float32)                   # (N,)

        # Reference MAC = club's MAC evaluated at the club rate.
        mac_ref = self._mac(jnp.full(N, club_rate), t)[self.eu_region_idx]
        mac_r = self._mac(mu, t)                                     # (N,)
        tau = jnp.clip((mac_ref - mac_r) / jnp.maximum(mac_ref, 1e-8), 0.0, 1.0)
        tau = jnp.where(membership, 0.0, tau)                       # members exempt

        # Exports from each region to club members, by sector.
        club_exports = jnp.einsum("rds,d->rs", trade_flows, member_f)  # (N, NS)

        intensity = jnp.array(self.emissions_intensity)             # (N, NS)
        coverage = state.get("sector_coverage", None)               # (NS,) or None
        if coverage is not None:
            intensity = intensity * coverage.astype(jnp.float32)[None, :]

        cbam_cost = (club_exports * intensity * tau[:, None]).sum(axis=1)  # (N,)

        n_members = jnp.maximum(member_f.sum(), 1.0)
        cbam_revenue = member_f * (cbam_cost.sum() / n_members)     # equal split
        cbam_tariff_matrix = member_f[:, None] * tau[None, :]       # (N, N) logging
        return cbam_tariff_matrix, cbam_revenue, cbam_cost

    # ------------------------------------------------------------------ reward
    def generate_rewards(self, new_state: dict, old_state: dict) -> dict:
        rewards = super().generate_rewards(new_state, old_state)
        if self.reward_mode != "additive_cbam":
            return rewards
        # The additive −λ·cost penalty must apply once per 3-stage cycle, on the
        # climate stage only.  On propose/evaluate stages ΔU=0 but the persisted
        # cbam_cost would otherwise be double/triple-counted; add it back there.
        is_climate = (new_state["current_timestep"] % 3) == 0
        lam = new_state["cbam_lambda"]
        cbam_cost = new_state["cbam_cost_all_regions"]              # (N,)
        add_back = jnp.where(is_climate, 0.0, lam * cbam_cost)      # (N,)
        return {
            i_to_agent_str(i): rewards[i_to_agent_str(i)] + add_back[i]
            for i in range(self.num_regions)
        }

    # ------------------------------------------------------------------ action space
    @property
    def action_space(self) -> dict:
        spaces = super().action_space  # RiceMRIO mrio actions per agent
        N = self.num_regions
        D = self.num_discrete_action_levels
        for agent_id in range(N):
            astr = i_to_agent_str(agent_id)
            spaces[astr]["proposal"] = Discrete(D)                 # candidate club rate
            spaces[astr]["proposal_decisions"] = MultiDiscrete([2] * N)  # accept/reject
        return spaces

    def generate_action_masks(self, state: dict) -> dict:
        mask = super().generate_action_masks(state)
        N = self.num_regions
        D = self.num_discrete_action_levels
        for agent_id in range(N):
            astr = i_to_agent_str(agent_id)
            mask[astr]["proposal"] = np.ones(D, dtype=np.float32)
            mask[astr]["proposal_decisions"] = np.ones((N, 2), dtype=np.float32)
        return mask

    # ------------------------------------------------------------------ observation
    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        obs = super().generate_observation(state)  # RiceMRIO compact obs
        stage = jnp.float32(state["current_timestep"] % 3)
        for agent_id in range(self.num_regions):
            astr = i_to_agent_str(agent_id)
            obs[astr]["negotiation_stage"] = stage
            obs[astr]["proposals"] = state["proposals"]
            obs[astr]["club_membership"] = state["club_membership"].astype(jnp.float32)
            obs[astr]["club_mitigation_rate"] = state["club_mitigation_rate"]
            obs[astr]["own_min_mitigation"] = (
                state["minimum_mitigation_rate_all_regions"][agent_id]
            )
        return obs


class MRIOClubCBAM(_MRIOClubBase):
    """B1 — Open-accession single CBAM club.

    The plain single-club design: one club anchored on ``eu_region_idx`` that
    any region may join by accepting the club's mitigation rate.  Members face
    no CBAM; non-members pay a MAC-gap CBAM on their exports to members.
    A singleton club {anchor} with club_rate = μ_anchor is bit-identical to the
    base RiceMRIO single-EU differential CBAM (canonical null condition).
    """

    pass


class MRIOSectoralClub(_MRIOClubBase):
    """B2 — Single club with negotiated sectoral CBAM coverage.

    Extends the open-accession club with a per-sector coverage vote: the
    anchor decides which sectors the CBAM applies to (``state["sector_coverage"]``).
    Uncovered sectors are exempt from the border levy, modelling the staged
    sectoral rollout of the EU CBAM (Reg. 2023/956 Annex I — initially iron &
    steel, cement, aluminium, fertilisers, electricity, hydrogen).
    """

    def _get_initial_state(self, key) -> dict:
        state = super()._get_initial_state(key)
        # All sectors covered by default (anchor can switch them off each round).
        state["sector_coverage"] = jnp.ones(self.num_sectors, dtype=jnp.bool_)
        return state

    @property
    def action_space(self) -> dict:
        spaces = super().action_space
        NS = self.num_sectors
        for agent_id in range(self.num_regions):
            astr = i_to_agent_str(agent_id)
            spaces[astr]["coverage_vote"] = MultiDiscrete([2] * NS)  # per-sector on/off
        return spaces

    def generate_action_masks(self, state: dict) -> dict:
        mask = super().generate_action_masks(state)
        NS = self.num_sectors
        for agent_id in range(self.num_regions):
            astr = i_to_agent_str(agent_id)
            mask[astr]["coverage_vote"] = np.ones((NS, 2), dtype=np.float32)
        return mask

    def step_propose(self, state: dict, actions: dict) -> dict:
        state = super().step_propose(state, actions)
        # The anchor sets sectoral coverage; >0 vote ⇒ sector is covered.
        state["sector_coverage"] = actions["coverage_vote"][self.eu_region_idx] > 0
        return state

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        obs = super().generate_observation(state)
        for agent_id in range(self.num_regions):
            astr = i_to_agent_str(agent_id)
            obs[astr]["sector_coverage"] = state["sector_coverage"].astype(jnp.float32)
        return obs


class MRIOMultiClub(_MRIOClubBase):
    """B3 — Competing CBAM clubs.

    Multiple clubs, each anchored on a fixed *core* region (``club_cores``).
    Every region may join at most one club (the accepted core offering the
    highest mitigation rate).  An exporter pays each club it is *not* a member
    of, on its exports to that club's members, at that club's MAC-gap rate.

    This models overlapping/competing carbon clubs and accession competition
    (cf. Hagen & Schneider 2021; Farrokhi & Lashkaripour 2024 on coalition
    structure under carbon tariffs).  With a single core equal to
    ``eu_region_idx`` it reduces to :class:`MRIOClubCBAM`.
    """

    # Fixed anchor regions for the competing clubs. Empty ⇒ fall back to the
    # single ``eu_region_idx`` core (degenerate single-club case).
    club_cores: tuple = eqx.field(static=True, default=())

    def _cores(self) -> list:
        return list(self.club_cores) if self.club_cores else [self.eu_region_idx]

    def _get_initial_state(self, key) -> dict:
        state = super()._get_initial_state(key)
        N = self.num_regions
        # club_id[r] = core index of r's club, or -1 if unaffiliated. Cores
        # belong to their own clubs from the start.
        club_id = jnp.full(N, -1, dtype=jnp.int32)
        for c in self._cores():
            club_id = club_id.at[c].set(c)
        state["club_id"] = club_id
        state["club_membership"] = club_id >= 0
        state["club_mitigation_rate"] = jnp.zeros(N, dtype=jnp.float32)
        return state

    def step_evaluate_proposals(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        state = state.copy()
        N = self.num_regions
        proposals = state["proposals"]                     # (N,)
        decisions = actions["proposal_decisions"] > 0      # (N, N) decider×target
        club_id = jnp.full(N, -1, dtype=jnp.int32)
        best_rate = jnp.full(N, -1.0, dtype=jnp.float32)
        for c in self._cores():
            accept_c = decisions[:, c]                      # (N,) who accepts core c
            rate_c = proposals[c]                           # scalar
            take = accept_c & (rate_c > best_rate)          # join the highest-rate club
            club_id = jnp.where(take, c, club_id)
            best_rate = jnp.where(take, rate_c, best_rate)
        for c in self._cores():
            club_id = club_id.at[c].set(c)                  # cores stay in own club
        membership = club_id >= 0
        # Per-region binding rate = the rate of the club it belongs to.
        safe_id = jnp.clip(club_id, 0, N - 1)
        club_rate_per_region = jnp.where(membership, proposals[safe_id], 0.0)
        state["club_id"] = club_id
        state["club_membership"] = membership
        state["club_mitigation_rate"] = club_rate_per_region
        state["minimum_mitigation_rate_all_regions"] = club_rate_per_region
        return state

    def _postprocess_cbam(
        self, state, trade_flows, gross_imports_mrio, mitigation_rates,
        cbam_tariff_matrix, cbam_revenue, cbam_cost_raw,
    ):
        N = self.num_regions
        t = state["activity_timestep"]
        mu = (
            mitigation_rates
            if mitigation_rates is not None
            else state["mitigation_rates_all_regions"]
        )
        proposals = state["proposals"]
        club_id = state["club_id"]
        mac_r = self._mac(mu, t)                            # (N,)
        intensity = jnp.array(self.emissions_intensity)    # (N, NS)
        total_cost = jnp.zeros(N, dtype=jnp.float32)
        revenue = jnp.zeros(N, dtype=jnp.float32)
        matrix = jnp.zeros((N, N), dtype=jnp.float32)
        for c in self._cores():
            members = club_id == c                          # (N,) bool, incl. core c
            member_f = members.astype(jnp.float32)
            rate_c = proposals[c]
            mac_ref = self._mac(jnp.full(N, rate_c), t)[c]  # scalar
            tau = jnp.clip((mac_ref - mac_r) / jnp.maximum(mac_ref, 1e-8), 0.0, 1.0)
            tau = jnp.where(members, 0.0, tau)              # members exempt from club c
            club_exports = jnp.einsum("rds,d->rs", trade_flows, member_f)  # (N, NS)
            cost_c = (club_exports * intensity * tau[:, None]).sum(axis=1)  # (N,)
            total_cost = total_cost + cost_c
            n_members = jnp.maximum(member_f.sum(), 1.0)
            revenue = revenue + member_f * (cost_c.sum() / n_members)
            matrix = matrix + member_f[:, None] * tau[None, :]
        return matrix, revenue, total_cost

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        obs = super().generate_observation(state)
        for agent_id in range(self.num_regions):
            astr = i_to_agent_str(agent_id)
            obs[astr]["club_id"] = state["club_id"].astype(jnp.float32)
        return obs
    