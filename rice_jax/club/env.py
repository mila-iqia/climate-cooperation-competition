"""Two-layer club mediator environment on JAX RICE-N.

A mediator agent proposes club terms once per negotiation cycle:
  * ``club_min_rate``  — minimum mitigation rate required for membership
  * ``club_tariff``    — uniform tariff club members impose on non-member imports

Regions no longer propose; they only accept/reject via ``club_join``.
Stage cycle (current_timestep % 3): 0 = climate/economy, 1 = mediator proposes,
2 = regions vote. Membership is re-decided every cycle.

Literature:
  * Nordhaus (2015, AER 105(4)) "Climate Clubs" — uniform penalty tariff on non-members.
  * Zhang et al. (2022) RICE-N — base negotiation protocol this replaces.
  * Ivanov et al. (2023) "Mediated Multi-Agent RL" — learned-mediator concept.
"""

from typing import Any, Literal

import jax
import jax.numpy as jnp
import jaxnasium as jym
from jaxnasium import Discrete, MultiDiscrete

from rice_jax.core.env import Rice
from rice_jax.utils import i_to_agent_str

MEDIATOR = "mediator"
# "mediator" < "region-XX" in sorted pytree key order, so the mediator is
# row 0 in the stacked per-action arrays produced by `process_actions`.
_MED = 0


class RiceClubMediator(Rice):
    """Rice with a mediator agent that defines a Nordhaus-style climate club."""

    negotiation_on: bool = True  # required: reuses the 3-stage step cycle

    mediator_reward_mode: Literal["emissions", "members"] = "emissions"
    mediator_reward_scale: float = 0.01  # scales -global_emissions to ~region reward magnitude
    # (min_mitigation, tariff) in [0, 1]; bypasses the mediator policy when set
    fixed_club_params: tuple[float, float] | None = None

    def __check_init__(self):
        if not self.negotiation_on:
            raise ValueError("RiceClubMediator requires negotiation_on=True")

    @property
    def agent_keys(self) -> list[str]:
        return [i_to_agent_str(i) for i in range(self.num_regions)] + [MEDIATOR]

    # ------------------------------------------------------------------ spaces

    @property
    def action_space(self) -> dict[str, jym.Space]:
        NR = self.num_regions
        L = self.num_discrete_action_levels
        # Identical structure for every agent (required by `process_actions`
        # and the shared-policy PPO); masks pin the role-irrelevant slots.
        actions = {
            "import_bid": MultiDiscrete([L] * NR),
            "import_tariff": MultiDiscrete([L] * NR),
            "savings_rate": Discrete(L),
            "mitigation_rate": Discrete(L),
            "export_limit": Discrete(L),
            "club_join": Discrete(2),  # regions only
            "club_min_rate": Discrete(L),  # mediator only
            "club_tariff": Discrete(L),  # mediator only
        }
        return {agent: actions for agent in self.agent_keys}

    @property
    def observation_space(self) -> dict[str, jym.AgentObservation]:
        obs, _ = self.reset(jax.random.PRNGKey(0))

        def agent_space(agent_obs: jym.AgentObservation) -> jym.AgentObservation:
            return jym.AgentObservation(
                observation=jym.Box(
                    low=-9999,
                    high=9999,
                    shape=agent_obs.observation.shape,
                    dtype=agent_obs.observation.dtype,
                ),
                action_mask=jax.tree.map(
                    lambda m: jym.Box(low=0, high=1, shape=m.shape, dtype=m.dtype),
                    agent_obs.action_mask,
                ),
            )

        return {agent: agent_space(obs[agent]) for agent in self.agent_keys}

    # ------------------------------------------------------------------- state

    def _get_initial_state(self, key):
        state = super()._get_initial_state(key)
        state["club_min_mitigation"] = jnp.float32(0.0)
        state["club_tariff_rate"] = jnp.float32(0.0)
        state["club_membership"] = jnp.zeros(self.num_regions)
        return state

    # ------------------------------------------------------------------ stages

    def step_propose(self, state: dict, actions: dict) -> dict:
        # copy: lax.switch branches share the closed-over state dict
        state = state.copy()
        # process_actions already divided levels by num_discrete_action_levels
        club_min = actions["club_min_rate"][_MED].astype(jnp.float32)
        club_tariff = actions["club_tariff"][_MED].astype(jnp.float32)
        if self.fixed_club_params is not None:
            # derive from traced values: fresh scalar consts become jaxpr
            # Literals, which break eager lax.switch in step_env
            club_min = club_min * 0.0 + self.fixed_club_params[0]
            club_tariff = club_tariff * 0.0 + self.fixed_club_params[1]
        state["club_min_mitigation"] = club_min
        state["club_tariff_rate"] = club_tariff
        return state

    def step_evaluate_proposals(self, state: dict, actions: dict) -> dict:
        state = state.copy()
        joins = actions["club_join"][_MED + 1 :]  # region rows only
        membership = (joins > 0).astype(jnp.float32)
        state["club_membership"] = membership
        # Existing mask machinery enforces the mitigation floor for members
        state["minimum_mitigation_rate_all_regions"] = (
            state["club_min_mitigation"] * membership
        )
        return state

    def step_climate_and_economy(self, state: dict[str, Any], actions: dict[str, Any]):
        region_actions = {k: v[_MED + 1 :] for k, v in actions.items()}

        # Nordhaus (2015) penalty tariff: member importer i floors its tariff on
        # non-member exporter j at the club rate. Diagonal stays 0 by construction.
        membership = state["club_membership"]
        tariff_floor = (
            membership[:, None]
            * (1.0 - membership)[None, :]
            * state["club_tariff_rate"]
        )
        effective_tariff = jnp.maximum(region_actions["import_tariff"], tariff_floor)
        region_actions["import_tariff"] = effective_tariff

        state = state.copy()
        # Base Rice never populates "import_tariffs", leaving calc_welfloss_multiplier
        # inert; the club needs this channel so tariffed exporters lose welfare.
        state["import_tariffs"] = effective_tariff
        return super().step_climate_and_economy(state, region_actions)

    def calc_global_carbon_mass(
        self, state: dict, productions, mitigation_rates
    ) -> tuple:
        global_carbon_mass, carbon_updates = super().calc_global_carbon_mass(
            state, productions, mitigation_rates
        )
        if "global_emissions" not in carbon_updates:
            # carbon_model="base" never updates global_emissions (frozen at its
            # initial value), which would make the mediator emissions reward a
            # constant; recompute it exactly as the base branch does internally
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

    # ------------------------------------------------------------- obs & masks

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        global_features = [
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
        public_features = ["mitigation_rates_all_regions"]
        private_features = [
            "production_factor_all_regions",
            "intensity_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "aggregate_consumption",
            "minimum_mitigation_rate_all_regions",
        ]
        club_features = {
            "club_min_mitigation": state["club_min_mitigation"],
            "club_tariff_rate": state["club_tariff_rate"],
            "club_membership": state["club_membership"],
            # stage of the NEXT step, which this observation informs
            "next_stage": jax.nn.one_hot((state["current_timestep"] + 1) % 3, 3),
        }

        shared = {
            **{f: state[f] for f in global_features},
            **{f: state[f] for f in public_features},
            **club_features,
        }
        obs = {
            i_to_agent_str(agent_id): {
                **shared,
                **{f: state[f][agent_id] for f in private_features},
                "role_flag": jnp.float32(0.0),
                "own_membership": state["club_membership"][agent_id],
            }
            for agent_id in range(self.num_regions)
        }
        # Same feature keys/shapes as regions so flattened obs lengths match
        obs[MEDIATOR] = {
            **shared,
            **{f: jnp.zeros_like(state[f][0]) for f in private_features},
            "role_flag": jnp.float32(1.0),
            "own_membership": jnp.float32(0.0),
        }
        return obs

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        mask = super().generate_action_masks(state)
        L = self.num_discrete_action_levels
        pin0 = jnp.arange(L) == 0  # only level 0 allowed

        for agent_id in range(self.num_regions):
            m = mask[i_to_agent_str(agent_id)]
            m["club_min_rate"] = m["club_min_rate"] * pin0
            m["club_tariff"] = m["club_tariff"] * pin0

        med = mask[MEDIATOR]
        med["savings_rate"] = med["savings_rate"] * pin0
        med["mitigation_rate"] = med["mitigation_rate"] * pin0
        med["export_limit"] = med["export_limit"] * pin0
        med["import_bid"] = med["import_bid"] * pin0
        med["import_tariff"] = med["import_tariff"] * pin0
        med["club_join"] = med["club_join"] * jnp.array([1.0, 0.0])
        return mask

    # ----------------------------------------------------------------- rewards

    def generate_rewards(self, new_state: dict, old_state: dict) -> dict[str, float]:
        rewards = super().generate_rewards(new_state, old_state)
        if self.mediator_reward_mode == "members":
            mediator_reward = jnp.mean(new_state["club_membership"])
        elif self.mediator_reward_mode == "emissions":
            # Absolute (not diff) reward; a worst-case baseline offset would be
            # action-independent and leave the policy gradient unchanged.
            mediator_reward = (
                -new_state["global_emissions"] * self.mediator_reward_scale
            )
        else:
            raise ValueError(
                f"Unknown mediator_reward_mode: {self.mediator_reward_mode}"
            )
        rewards[MEDIATOR] = mediator_reward
        return rewards
