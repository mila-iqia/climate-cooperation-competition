from typing import Any
import jaxnasium as jym

import chex
import jax.numpy as jnp
import numpy as np
from jaxnasium import Discrete, MultiDiscrete
import equinox as eqx
import jax
import optax

from rice_jax import Rice, RiceMRIO
from rice_jax.utils import i_to_agent_str


class OptimalMitigation(Rice):
    """Sets a mimum and maximum mitigation rate for all agents through the action mask

    env = OptimalMitigation(minimum_mitigation_rate=y, maximum_mitigation_rate=x)
    """

    # Both are inclusive:
    minimum_mitigation_rate: int = 9
    maximum_mitigation_rate: int = 9

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        action_mask = super().generate_action_masks(state)  # get default

        actions = jnp.arange(self.num_discrete_action_levels)
        min_mask = actions >= self.minimum_mitigation_rate
        max_mask = actions <= self.maximum_mitigation_rate
        min_max_mask = min_mask * max_mask

        for agent_id in range(self.num_regions):
            mitigation_mask = action_mask[i_to_agent_str(agent_id)]["mitigation_rate"]
            mitigation_mask = mitigation_mask * min_max_mask

        return action_mask


class BasicClub(Rice):
    club_mitigation_rate: int = 8
    promote_free_trade_among_club_members: bool = True
    # NOTE: this will be updated later with more targeted region_ids
    club_members_ = [0, 2, 4, 5, 6, 7, 15, 8, 12]

    @property
    def club_members(self) -> np.ndarray:
        return np.array(
            [
                region_id
                for region_id in range(self.num_regions)
                if region_id in self.club_members_
            ]
        )

    @property
    def non_club_members(self) -> list:
        return np.array(
            [
                region_id
                for region_id in range(self.num_regions)
                if region_id not in self.club_members
            ]
        )

    def generate_action_masks(self, state: dict) -> chex.Array:
        action_mask = super().generate_action_masks(state)  # get default

        # Update action mask for each club member
        for agent_id in self.club_members:
            agent_str = i_to_agent_str(agent_id)

            # Force club members to mitigate a minimum of self.club_mitigation_rate
            club_member_mitigation_mask = (
                jnp.arange(self.num_discrete_action_levels) >= self.club_mitigation_rate
            )
            action_mask[agent_str]["mitigation_rate"] = club_member_mitigation_mask

            # Now we put a minimum tariff on everyone below the club mitigation rate
            # (for club members the minimum should be 0 since they always mitigate the club rate)
            min_tariff_amount_per_region = (
                self.club_mitigation_rate
                - (
                    state["mitigation_rates_all_regions"]
                    * self.num_discrete_action_levels
                )
            ).clip(min=0)
            min_tariff_amount_per_region_mask = (
                jnp.arange(self.num_discrete_action_levels)
                >= min_tariff_amount_per_region[:, None]
            )
            action_mask[agent_str]["import_tariff"] = min_tariff_amount_per_region_mask

            # Optional: promote free trade among club members
            # Only allow "no-tariff" among club members
            if self.promote_free_trade_among_club_members:
                action_mask[agent_str]["import_tariff"].at[self.club_members].set(
                    jnp.arange(self.num_discrete_action_levels) == 0
                )

        return action_mask

    def generate_observation(self, state: dict) -> chex.Array:
        """Add a club membership indicator to the observation"""
        obs = super().generate_observation(state)
        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)
            is_club_member = agent_id in self.club_members
            # Simply add a feature to each agent observation
            obs[agent_str]["is_club_member"] = is_club_member

        return obs


class MaxExport(Rice):
    """Scenario where each region is forced to choose the maximum export limit permitted.

    This builds on the base export mask that already constrains export_ratio by
    physical capacity. After computing the default mask we reduce the
    export_limit action space to a single choice: the highest-discretization
    level still allowed by that mask.
    """

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        mask = super().generate_action_masks(state)

        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)
            if "export_limit" in mask[agent_str]:
                export_mask = mask[agent_str]["export_limit"]
                # find largest allowed index in a JIT‑friendly way
                # reverse mask and take argmax (returns 0 if no True values)
                rev_idx = jnp.argmax(export_mask[::-1])
                # compute corresponding forward index
                max_idx = export_mask.shape[0] - 1 - rev_idx
                # ensure the mask actually contained a True
                has_allowed = jnp.any(export_mask)
                # build one-hot: if none allowed, leave mask unchanged
                new_mask = jnp.where(
                    has_allowed,
                    jnp.eye(export_mask.shape[0], dtype=export_mask.dtype)[max_idx],
                    export_mask,
                )
                mask[agent_str]["export_limit"] = new_mask
        return mask


class MaxExportFixedSavings(MaxExport):
    """Max-export scenario with a constant savings rate for all regions.

    The savings rate action is ignored and replaced with a fixed continuous
    value (default 0.2) after the parent class has processed the actions. This
    allows running the environment without training or specifying savings_rate
    actions while still enforcing the maximum export behavior from
    :class:`MaxExport`.
    """

    fixed_savings_rate: float = 0.2

    def process_actions(self, actions: dict[str, Any], state: dict[str, chex.Array]):
        # call parent processing (including masking, scaling, stacking)
        actions = super().process_actions(actions, state)
        # override the savings_rate component with a constant value between 0
        # and 1. The actions coming in have already been divided by
        # ``num_discrete_action_levels`` so 0.2 is treated as the continuous rate.
        if "savings_rate" in actions:
            actions["savings_rate"] = jnp.ones_like(actions["savings_rate"]) * self.fixed_savings_rate
        return actions


class BasicClubTariffAmbition(Rice):
    @property
    def action_space(self):
        N_REGIONS = self.num_regions
        N_DISCRETIZATION = self.num_discrete_action_levels

        action_space = super().action_space

        for agent_id in range(self.num_regions):
            agent_key = i_to_agent_str(agent_id)
            # remove standard propose
            action_space[agent_key].pop("proposal_ask", None)
            action_space[agent_key].pop("proposal_promise", None)
            # add new propose
            action_space[agent_key]["proposal"] = Discrete(N_DISCRETIZATION)

        return action_space


    def step_propose(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        proposals = actions["proposal"]

        state["proposals"] = proposals
        return state

    def step_evaluate_proposals(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")

        proposals = state["proposals"]
        proposal_decisions = actions["proposal_decisions"].T

        proposal_decisions_ = (proposal_decisions > 0).astype(jnp.bool_)

        accepted_mitigation_rates = proposals * proposal_decisions_.astype(jnp.int_)

        lower_bound_mitigation_rates = jnp.max(accepted_mitigation_rates, axis=1)
        state["proposal_decisions"] = proposal_decisions_
        state["minimum_mitigation_rate_all_regions"] = lower_bound_mitigation_rates
        return state

    def _get_initial_state(self, key):
        # this fn gets called in reset_env of parent class before generate_observation_and_action_mask
        # so we can do scenario-specific state alterations here

        # Get default initial state
        state = super()._get_initial_state(key)

        # Scenario alterations:
        state.pop("promised_mitigation_rate")
        state.pop("requested_mitigation_rate")

        state["proposals"] = jnp.zeros((self.num_regions))

        return state

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        """Format observations for each agent by concatenating global, public and private features."""

        global_features = [
            # World values observed by all regions
            "activity_timestep",
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "global_temperature_boxes",
            "global_carbon_reservoirs",
            "global_cumulative_emissions",
            "global_cumulative_land_emissions",
            "global_alpha",
            "global_emissions",
            "global_acc_pert_carb_stock",
        ]

        public_features = [  # NOTE: commented some things out here and (partly) moved to private
            # Per agent values observed by all regions
            "mitigation_rates_all_regions",
            # "capital_all_regions",
            # "capital_depreciation_all_regions",
            # "labor_all_regions",
            # "gross_output_all_regions",
            # "investment_all_regions",
            # "aggregate_consumption",
            # "savings_all_regions",
            # "export_limit_all_regions",
            # "current_balance_all_regions",
            # "import_tariffs",
            "proposals",
        ]

        private_features = [
            # Per agent values whereby each agent only observes its own values
            "production_factor_all_regions",
            "intensity_all_regions",
            # "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            # "social_welfare_all_regions",
            # "reward_all_regions",
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "aggregate_consumption",
            # "savings_all_regions",
            # "export_limit_all_regions",
            # "current_balance_all_regions",
            # "import_tariffs",
        ]
        bilateral_features = []  # Bilateral features are only observed by two regions
        if self.negotiation_on:
            global_features.append("negotiation_stage")
            private_features.append("minimum_mitigation_rate_all_regions")
            bilateral_features = ["proposal_decisions"]

        obs = {
            i_to_agent_str(agent_id): {
                **{feature: state[feature] for feature in global_features},
                **{feature: state[feature] for feature in public_features},
                **{feature: state[feature][agent_id] for feature in private_features},
                **{
                    feature: jnp.concatenate(
                        [state[feature][agent_id], state[feature].T[agent_id]]
                    )
                    for feature in bilateral_features
                },
            }
            for agent_id in range(self.num_regions)
        }

        # We flatten and concat everything (per agent) in the `generate_observation_and_action_mask` function
        return obs

    def generate_action_masks(self, state: dict) -> chex.Array:
        action_mask = super().generate_action_masks(state)  # get default

        # Update action masks
        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)

            # Minimum mitigation mask for own proposals is set via minimum mitigation rate in the state
            # and therefore already handled in parent class

            # Now we put a minimum tariff on everyone below the club mitigation rate
            # (for club members the minimum should be 0 since they always mitigate the club rate)
            min_tariff_amount_per_region = (
                (
                    state["minimum_mitigation_rate_all_regions"][agent_id]
                    - (state["mitigation_rates_all_regions"])
                )
                * self.num_discrete_action_levels
            ).clip(min=0)
            min_tariff_amount_per_region_mask = (
                jnp.arange(self.num_discrete_action_levels)
                >= min_tariff_amount_per_region[:, None]
            )

            # When other regions are above own MMR, then the min_tariff should now all be 1s
            # Those we set to "only allow 0 tariff"
            min_tariff_amount_per_region_mask = jnp.where(
                jnp.all(min_tariff_amount_per_region_mask, axis=1)[:, None],
                (jnp.arange(self.num_discrete_action_levels) == 0)[None, :],
                min_tariff_amount_per_region_mask,
            )
            action_mask[agent_str]["import_tariff"] = min_tariff_amount_per_region_mask

        return action_mask
    
class BasicClubTariffAmbitionFixedSavings(BasicClubTariffAmbition):
    """
    This scenario is identical to BasicClubTariffAmbition, but the savings
    rate action is removed from the agent's action space and fixed to a 
    constant value of 0.2 for all regions.
    """

    fixed_savings_rate: float = 0.2

    @property
    def action_space(self):
        # Get the action space from the parent class (BasicClubTariffAmbition)
        # by explicitly calling the property's getter function.
        action_space = BasicClubTariffAmbition.action_space.fget(self)

        # Remove the 'savings_rate' action for all agents, as it is now fixed
        for agent_id in range(self.num_regions):
            agent_key = i_to_agent_str(agent_id)
            if "savings_rate" in action_space[agent_key]:
                action_space[agent_key].pop("savings_rate")

        return action_space

    def process_actions(self, actions: dict, state: dict):
        # To call the overridden parent method, we must be explicit, as super()
        # does not work with a regular method if the class has property overrides.
        processed_actions = BasicClubTariffAmbition.process_actions(self, actions, state)

        # Manually add the fixed savings rate to the processed actions dictionary.
        # This creates an array of shape (num_regions,) with the fixed value.
        processed_actions["savings_rate"] = jnp.full(
            (self.num_regions,), self.fixed_savings_rate
        )

        return processed_actions
    
    def generate_action_masks_base(self, state: dict[str, Any]) -> dict[str, Any]:
        """Should output the same structure as `self.action_space`
        1: Allowed action, 0: disallowed action
        """

        NUM_REGIONS = self.num_regions
        DISCRETE_ACTION_LEVELS = self.num_discrete_action_levels

        def allow_all_actions_in_action_space(action_space):
            if isinstance(action_space, MultiDiscrete):
                num_actions = np.array(action_space.nvec).shape[0]
                num_discrete_actions = np.array(action_space.nvec)[0]
                return np.ones((num_actions, num_discrete_actions))
            elif isinstance(action_space, Discrete):
                num_discrete_actions = int(action_space.n)
                return np.ones((num_discrete_actions,))
            else:
                raise ValueError(f"Unknown action space: {action_space}")

        # Allow each action as a base
        mask = jax.tree.map(allow_all_actions_in_action_space, self.action_space)

        # Disallow actions on own region "self"
        for a_id in range(NUM_REGIONS):
            agent_str = i_to_agent_str(a_id)
            # Set diagonal elements to 0 for import actions (except first element)
            mask[agent_str]["import_bid"][a_id][1:] = 0
            mask[agent_str]["import_tariff"][a_id][1:] = 0
            if self.negotiation_on and "proposal_ask" in mask[agent_str]:
                mask[agent_str]["proposal_ask"][a_id][1:] = 0
                mask[agent_str]["proposal_promise"][a_id][1:] = 0

        # Minimum mitigation rate masking
        minimum_mitigation_rate_all = state["minimum_mitigation_rate_all_regions"]
        for agent_id in range(self.num_regions):
            min_mitigation_rate_agent = (
                minimum_mitigation_rate_all[agent_id] * self.num_discrete_action_levels
            )
            mask[i_to_agent_str(agent_id)]["mitigation_rate"] = (
                jnp.arange(self.num_discrete_action_levels) >= min_mitigation_rate_agent
            )

        if self.action_window_size > 0:

            def create_windowed_mask(prev_actions):
                MAX_DIFF = self.action_window_size
                POSSIBLE_ACTIONS = jnp.arange(DISCRETE_ACTION_LEVELS)
                return jnp.abs(POSSIBLE_ACTIONS - prev_actions) <= MAX_DIFF

            # Only allow actions around the previous action for `savings` and `mitigation` rate actions
            # prev_savings_actions = jnp.round(
            #     state["savings_all_regions"] * DISCRETE_ACTION_LEVELS
            # )
            prev_mitigation_actions = jnp.round(
                state["mitigation_rates_all_regions"] * DISCRETE_ACTION_LEVELS
            )
            for agent_id in range(NUM_REGIONS):
                # _savings_mask = create_windowed_mask(prev_savings_actions[agent_id])
                # mask[i_to_agent_str(agent_id)]["savings_rate"] = (
                #     mask[i_to_agent_str(agent_id)]["savings_rate"] * _savings_mask
                # )  # Multiply with existing mask to not overwrite

                _mitigation_mask = create_windowed_mask(
                    prev_mitigation_actions[agent_id]
                )
                agent_mitigation_mask = (
                    mask[i_to_agent_str(agent_id)]["mitigation_rate"] * _mitigation_mask
                )  # Multiply with existing mask to not overwrite

                # If the mitigation rate mask is now all 0s, that means the minimum mitigation rate is outside the window
                # in that case, the mitigation rate should still MOVE to the minimum mitigation rate
                # i.e. the upper half of the _mitigation_mask should be
                is_action_available = jnp.any(agent_mitigation_mask)
                move_to_minimum_within_window = (
                    prev_mitigation_actions[agent_id]
                    < jnp.arange(DISCRETE_ACTION_LEVELS)
                ) * _mitigation_mask
                mask[i_to_agent_str(agent_id)]["mitigation_rate"] = jax.lax.select(
                    is_action_available,
                    agent_mitigation_mask,
                    move_to_minimum_within_window,
                )

        return mask
    
    def generate_action_masks(self, state: dict) -> chex.Array:
        action_mask = self.generate_action_masks_base(state)  # get default

        # Update action masks
        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)

            # Minimum mitigation mask for own proposals is set via minimum mitigation rate in the state
            # and therefore already handled in parent class

            # Now we put a minimum tariff on everyone below the club mitigation rate
            # (for club members the minimum should be 0 since they always mitigate the club rate)
            min_tariff_amount_per_region = (
                (
                    state["minimum_mitigation_rate_all_regions"][agent_id]
                    - (state["mitigation_rates_all_regions"])
                )
                * self.num_discrete_action_levels
            ).clip(min=0)
            min_tariff_amount_per_region_mask = (
                jnp.arange(self.num_discrete_action_levels)
                >= min_tariff_amount_per_region[:, None]
            )

            # When other regions are above own MMR, then the min_tariff should now all be 1s
            # Those we set to "only allow 0 tariff"
            min_tariff_amount_per_region_mask = jnp.where(
                jnp.all(min_tariff_amount_per_region_mask, axis=1)[:, None],
                (jnp.arange(self.num_discrete_action_levels) == 0)[None, :],
                min_tariff_amount_per_region_mask,
            )
            action_mask[agent_str]["import_tariff"] = min_tariff_amount_per_region_mask

        return action_mask


# ======================================================================
#  MRIO endogenous-club scenarios  (Phase 2E)
# ======================================================================
#
#  Three negotiation variants layered on top of RiceMRIO.  The RICE-N
#  "club stick" — per-agent import tariffs on laggards — does not exist
#  in RiceMRIO (it deletes import_bid / import_tariff / export_limit
#  actions).  The stick is therefore a *collectively levied* CBAM,
#  collected by the club on imports from non-members and rebated to
#  members.  This is the carbon-club design of Nordhaus (2015,
#  "Climate Clubs", American Economic Review 105(4)) operationalised
#  through the differential-CBAM machinery (EU CBAM Reg. 2023/956 Art. 9).
#
#  Negotiation runs on the base 3-stage cycle (propose → evaluate →
#  climate).  All club state lives in the `state` dict (initialised in
#  `_get_initial_state`) so the frozen equinox module is never mutated
#  and `jax.lax.switch` sees an invariant pytree structure across stages.
#
#  Club CBAM is routed through the `_postprocess_cbam` hook added to
#  RiceMRIO; only `cbam_cost_all_regions` feeds the reward, so clubs use
#  `reward_mode="additive_cbam"` (RCPO penalty −λ·cost).
# ======================================================================


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
    