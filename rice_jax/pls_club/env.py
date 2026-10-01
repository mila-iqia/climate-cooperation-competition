"""PLS climate club environments on JAX RICE-N (core Rice, not MRIO).

The probabilistic shield replaces the hard mitigation action mask: club members'
mitigation policies are reweighted toward the club floor (via the action-mask
channel + :class:`~pls_club.shield.ShieldedCategoricalLayer`), but non-compliant
levels keep nonzero probability — members CAN defect. Enforcement is economic:
complying members impose a Nordhaus (2015) penalty tariff on non-members AND on
defectors, not a constraint.

Two variants:
  * :class:`PLSNaiveClubRice` — naive: fixed exogenous club terms, regions
    choose ``club_join`` every step, no mediator, single-stage steps.
  * :class:`PLSClubMediator` — mirrors ``club.env.RiceClubMediator`` (learned
    mediator proposes terms, regions vote) with the mask swapped for the shield.

Literature:
  * Yang et al. (2023, IJCAI) "Safe RL via Probabilistic Logic Shields" — shield.
  * Nordhaus (2015, AER 105(4)) "Climate Clubs" — penalty tariff.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jaxnasium as jym
from jaxnasium import Discrete

from club.env import _MED, RiceClubMediator
from rice_jax.core.env import Rice
from rice_jax.core.scenarios import BasicClubTariffAmbition
from rice_jax.utils import i_to_agent_str

from .shield import ShieldMode, mitigation_safety_weights


class PLSNaiveClubRice(Rice):
    """Naive PLS club: fixed terms, per-step opt-in, shield instead of mask.

    Timing: joining at step t shields (and binds) the agent from step t+1 —
    defection is judged against the membership committed on the previous step,
    i.e. the membership the shield actually acted on.
    """

    club_min_rate: float = 0.5  # required mitigation rate for members, in [0, 1]
    club_tariff: float = 0.5  # penalty tariff on non-members/defectors, in [0, 1]
    shield_strength: float = 0.8  # P(safe|defect) = 1 - strength; 0 = null shield
    shield_mode: ShieldMode = "constant"
    shield_kappa: float = 10.0  # only used for mode="graded"

    # ------------------------------------------------------------------ spaces

    @property
    def action_space(self) -> dict[str, jym.Space]:
        spaces = super().action_space
        return {
            agent: {**actions, "club_join": Discrete(2)}
            for agent, actions in spaces.items()
        }

    # ------------------------------------------------------------------- state

    def _get_initial_state(self, key):
        state = super()._get_initial_state(key)
        state["club_membership"] = jnp.zeros(self.num_regions)
        state["club_defectors"] = jnp.zeros(self.num_regions)
        return state

    # -------------------------------------------------------------------- step

    def step_climate_and_economy(self, state: dict[str, Any], actions: dict[str, Any]):
        state = state.copy()

        # Defection: committed member whose realized mitigation is below the floor.
        prev_membership = state["club_membership"]
        defectors = prev_membership * (
            actions["mitigation_rate"] < self.club_min_rate
        ).astype(jnp.float32)
        complying = prev_membership - defectors

        # Nordhaus (2015) penalty: complying members floor their tariff on
        # everyone outside the complying set (non-members + defectors).
        # Diagonal is 0 since complying and (1 - complying) are complementary.
        tariff_floor = (
            complying[:, None] * (1.0 - complying)[None, :] * self.club_tariff
        )
        effective_tariff = jnp.maximum(actions["import_tariff"], tariff_floor)
        actions = {**actions, "import_tariff": effective_tariff}

        # club_join arrives scaled by 1/L from process_actions; >0 means join
        state["club_membership"] = (actions["club_join"] > 0).astype(jnp.float32)
        state["club_defectors"] = defectors
        # Base Rice never populates "import_tariffs", leaving the welfloss
        # channel inert; sanctions need it to bite.
        state["import_tariffs"] = effective_tariff
        return super().step_climate_and_economy(state, actions)

    def calc_global_carbon_mass(
        self, state: dict, productions, mitigation_rates
    ) -> tuple:
        global_carbon_mass, carbon_updates = super().calc_global_carbon_mass(
            state, productions, mitigation_rates
        )
        if "global_emissions" not in carbon_updates:
            # carbon_model="base" freezes global_emissions at its initial value;
            # recompute it exactly as the base branch does internally
            land_emissions = (
                self.region_params.xE_L0
                * (1 - self.region_params.xdelta_EL)
                ** (state["activity_timestep"] - 1)
                / self.num_regions
            )
            aux_m_all_regions = (
                state["intensity_all_regions"] * (1 - mitigation_rates) * productions
                + land_emissions
            )
            carbon_updates["global_emissions"] = jnp.sum(aux_m_all_regions)
        return global_carbon_mass, carbon_updates

    # ------------------------------------------------------------- obs & shield

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        obs = super().generate_observation(state)
        club_terms = jnp.array([self.club_min_rate, self.club_tariff], jnp.float32)
        for agent_id in range(self.num_regions):
            agent = obs[i_to_agent_str(agent_id)]
            agent["club_membership"] = state["club_membership"]
            agent["club_defectors"] = state["club_defectors"]
            agent["own_membership"] = state["club_membership"][agent_id]
            agent["club_terms"] = club_terms
        return obs

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        mask = super().generate_action_masks(state)
        weights = mitigation_safety_weights(
            jnp.float32(self.club_min_rate),
            state["club_membership"],
            self.num_discrete_action_levels,
            self.shield_strength,
            self.shield_mode,
            self.shield_kappa,
        )
        for agent_id in range(self.num_regions):
            m = mask[i_to_agent_str(agent_id)]
            # Shield replaces the hard mitigation mask (base mask is all-ones
            # since minimum_mitigation_rate_all_regions stays 0 here).
            m["mitigation_rate"] = m["mitigation_rate"] * weights[agent_id]
        return mask


class PLSClubTariffAmbition(BasicClubTariffAmbition):
    """BasicClubTariffAmbition with the hard mitigation mask swapped for a PLS shield.

    The accepted-proposal floor (``minimum_mitigation_rate_all_regions``) and the
    tariff-ambition mask are inherited unchanged; only the mitigation head's hard
    floor mask is replaced by soft safety weights, so regions CAN sample below
    their committed floor — and are then tariff-sanctioned by the inherited
    tariff-ambition mask (which keys on realized mitigation).
    """

    shield_strength: float = 0.8
    shield_mode: ShieldMode = "constant"
    shield_kappa: float = 10.0

    def generate_action_masks(self, state: dict) -> dict:
        mask = super().generate_action_masks(state)
        floors = state["minimum_mitigation_rate_all_regions"]  # per-region floors
        weights = jax.vmap(
            lambda floor: mitigation_safety_weights(
                floor,
                jnp.ones(1),
                self.num_discrete_action_levels,
                self.shield_strength,
                self.shield_mode,
                self.shield_kappa,
            )[0]
        )(floors)
        for agent_id in range(self.num_regions):
            # overwrite (not multiply): parent set the hard floor mask here
            mask[i_to_agent_str(agent_id)]["mitigation_rate"] = weights[agent_id]
        return mask


class PLSClubMediator(RiceClubMediator):
    """Mediator club with the hard mitigation mask replaced by a PLS shield.

    Members whose sampled mitigation falls below the mediator's floor are
    defectors: they are tariff-sanctioned like non-members that climate step.
    """

    shield_strength: float = 0.8
    shield_mode: ShieldMode = "constant"
    shield_kappa: float = 10.0

    def _get_initial_state(self, key):
        state = super()._get_initial_state(key)
        state["club_defectors"] = jnp.zeros(self.num_regions)
        return state

    def step_evaluate_proposals(self, state: dict, actions: dict) -> dict:
        # copy: lax.switch branches share the closed-over state dict
        state = state.copy()
        joins = actions["club_join"][_MED + 1 :]  # region rows only
        state["club_membership"] = (joins > 0).astype(jnp.float32)
        # Unlike RiceClubMediator, minimum_mitigation_rate_all_regions stays 0:
        # the shield replaces the hard mask.
        return state

    def step_climate_and_economy(self, state: dict[str, Any], actions: dict[str, Any]):
        region_actions = {k: v[_MED + 1 :] for k, v in actions.items()}

        membership = state["club_membership"]
        defectors = membership * (
            region_actions["mitigation_rate"] < state["club_min_mitigation"]
        ).astype(jnp.float32)
        complying = membership - defectors

        tariff_floor = (
            complying[:, None]
            * (1.0 - complying)[None, :]
            * state["club_tariff_rate"]
        )
        effective_tariff = jnp.maximum(region_actions["import_tariff"], tariff_floor)
        region_actions["import_tariff"] = effective_tariff

        state = state.copy()
        state["club_defectors"] = defectors
        state["import_tariffs"] = effective_tariff
        # Skip RiceClubMediator's climate step: its member/non-member floor is
        # replaced by the defector-aware floor above.
        return Rice.step_climate_and_economy(self, state, region_actions)

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        obs = super().generate_observation(state)
        for agent in obs.values():
            agent["club_defectors"] = state["club_defectors"]
        return obs

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        mask = super().generate_action_masks(state)
        weights = mitigation_safety_weights(
            state["club_min_mitigation"],
            state["club_membership"],
            self.num_discrete_action_levels,
            self.shield_strength,
            self.shield_mode,
            self.shield_kappa,
        )
        for agent_id in range(self.num_regions):
            m = mask[i_to_agent_str(agent_id)]
            m["mitigation_rate"] = m["mitigation_rate"] * weights[agent_id]
        return mask
