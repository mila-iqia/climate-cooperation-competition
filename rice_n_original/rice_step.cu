// Copyright (c) 2022, salesforce.com, inc and MILA.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause


__constant__ float kSmallNum = 1.0e-0;

extern "C"
{

    // Device helper functions for the environment dynamics
    __device__ float get_mitigation_cost(
        float p_b,
        float theta_2,
        float delta_pb,
        int timestep,
        float intensity)
    {
        return p_b /
               (1000 * theta_2) *
               pow(1 - delta_pb, timestep - 1) *
               intensity;
    }

    __device__ float get_exogenous_emissions(
        float f_0,
        float f_1,
        int t_f,
        int timestep)
    {
        return f_0 + min(f_1 - f_0, (f_1 - f_0) / t_f * (timestep - 1));
    }

    __device__ float get_land_emissions(
        float e_l0,
        float delta_el,
        int timestep,
        int kNumAgents)
    {
        return e_l0 * pow(1 - delta_el, timestep - 1) / float(kNumAgents);
    }

    __device__ float get_production(
        float production_factor_all_regions,
        float capital_all_regions,
        float labor_all_regions,
        float gamma)
    {
        return production_factor_all_regions *
               pow(capital_all_regions, gamma) *
               pow(labor_all_regions / 1000.0, 1 - gamma);
    }

    __device__ float get_damages(
        float t_at,
        float a_1,
        float a_2,
        int a_3)
    {
        return 1 / (1 + a_1 * t_at + a_2 * pow(t_at, a_3));
    }

    __device__ float get_abatement_cost(
        float mitigation_rate,
        float mitigation_cost,
        float theta_2)
    {
        return mitigation_cost * pow(mitigation_rate, theta_2);
    }

    __device__ float get_gross_output(
        float damages,
        float abatement_cost,
        float production)
    {
        return damages * (1 - abatement_cost) * production;
    }

    __device__ float get_investment(
        float savings,
        float gross_output)
    {
        return savings * gross_output;
    }

    __device__ float get_consumption(
        float savings,
        float gross_output,
        float *exports,
        const int kAgentId,
        const int kNumAgents)
    {
        float exports_total = 0.0;
        for (int region_id = 0; region_id < kNumAgents; region_id++)
        {
            exports_total += exports[kAgentId + region_id * kNumAgents];
        }
        float consumption = gross_output * (1 - savings) - exports_total;

        if (consumption >= 0.0)
        {
            return consumption;
        }
        else
        {
            return 0.0;
        }
    }

    __device__ float get_max_potential_exports(
        float x_max,
        float gross_output,
        float investment)
    {
        if (x_max * gross_output <= gross_output - investment)
        {
            return x_max * gross_output;
        }
        else
        {
            return gross_output - investment;
        }
    }

    __device__ void update_global_temperature(
        const float *phi_t,
        float *global_temperature,
        const float *b_t,
        float f_2x,
        float m_at,
        int m_at_1750,
        float global_exogenous_emissions,
        const int global_temperture_len,
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents)
    {
        // Update global_temperature in place.
        // Shapes:
        // phi_t: (global_temperture_len, global_temperture_len)
        // global_temperature: (num_envs, global_temperture_len)
        // b_t: (global_temperture_len,)
        // f_i: (1,)

        const float f_i = f_2x *
                              logf(m_at / m_at_1750) / logf(2) +
                          global_exogenous_emissions;

        for (int i_idx = 0; i_idx < global_temperture_len; i_idx++)
        {
            float intermediate_dot_prod = 0.0;
            for (int j_idx = 0; j_idx < global_temperture_len; j_idx++)
            {
                intermediate_dot_prod +=
                    phi_t[i_idx * global_temperture_len + j_idx] *
                    global_temperature[kEnvId * global_temperture_len + j_idx];
            }
            const int global_temperature_idx =
                kEnvId * global_temperture_len + i_idx;
            global_temperature[global_temperature_idx] =
                intermediate_dot_prod + b_t[i_idx] * f_i;
        }
    }

    __device__ float get_aux_m(
        float intensity,
        float mitigation_rate,
        float production,
        float land_emissions)
    {
        return intensity *
                   (1 - mitigation_rate) *
                   production +
               land_emissions;
    }

    __device__ void update_global_carbon_mass(
        const float *phi_m,
        float *global_carbon_mass,
        const float *b_m,
        float sum_aux_ms,
        const int carbon_mass_array_len,
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents)
    {
        // Update global_carbon_mass in-place.
        // Shapes:
        // phi_m:  carbon_mass_array_len, carbon_mass_array_len)
        // global_carbon_mass: (num_envs, num_agents, carbon_mass_array_len)
        // b_t:  carbon_mass_array_len,)
        // sum_aux_ms: (1,)
        // global_carbon_mass: (num_envs, num_agents, carbon_mass_array_len)
        for (int i_idx = 0; i_idx < carbon_mass_array_len; i_idx++)
        {
            float intermediate_dot_prod = 0.0;
            for (int j_idx = 0; j_idx < carbon_mass_array_len; j_idx++)
            {
                intermediate_dot_prod +=
                    phi_m[i_idx * carbon_mass_array_len + j_idx] *
                    global_carbon_mass[kEnvId * carbon_mass_array_len + j_idx];
            }
            const int global_carbon_mass_idx =
                kEnvId * carbon_mass_array_len + i_idx;
            global_carbon_mass[global_carbon_mass_idx] =
                intermediate_dot_prod +
                b_m[i_idx] * sum_aux_ms;
        }
    }

    __device__ float get_capital(
        float capital_depreciation,
        float capital,
        float delta,
        float investment)
    {
        return capital_depreciation * capital +
               delta * investment;
    }

    __device__ float get_labor(
        float labor,
        float l_a,
        float l_g)
    {
        return labor * pow((1 + l_a) / (1 + labor), l_g);
    }

    __device__ float get_production_factor(
        float production_factor,
        float g_a,
        float delta_a,
        float delta,
        int timestep)
    {
        return production_factor * (exp(0.0033) +
                                    g_a * exp(-delta_a * delta * (timestep - 1)));
    }

    __device__ float get_carbon_intensity(
        float intensity,
        float g_sigma,
        float delta_sigma,
        float delta,
        int timestep)
    {
        return intensity *
               exp(-g_sigma * pow(1 - delta_sigma, delta * (timestep - 1)) * delta);
    }

    __device__ float get_utility(
        float labor,
        float consumption,
        float alpha)
    {
        return (labor / 1000.0) * (pow(consumption / (labor / 1000.0) + kSmallNum, 1 - alpha) - 1) /
               (1 - alpha);
    }

    __device__ float get_social_welfare(
        float utility,
        float rho,
        float delta,
        int timestep)
    {
        return utility / pow(1 + rho, delta * timestep);
    }

    // Consumption aggregation for trade
    __device__ float get_armington_agg(
        float c_dom,
        float *c_for,
        const float *sub_rate,
        const float *dom_pref,
        const float *for_pref,
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents)
    {
        // assert(0 < sub_rate[0] && sub_rate[0] < 1);
        // assert(0 <= dom_pref[0] && dom_pref[0] <= 1);
        int c_for_index_offset = kEnvId * kNumAgents * kNumAgents +
                                 kAgentId * kNumAgents;
        float c_dom_pref = dom_pref[0] * pow(c_dom, sub_rate[0]);

        float c_for_pref = 0.0;
        for (int region_id = 0; region_id < kNumAgents; region_id++)
        {
            c_for_pref += for_pref[region_id] *
                          pow(
                              c_for[c_for_index_offset + region_id],
                              sub_rate[0]);
        }
        float c_agg = pow(c_dom_pref + c_for_pref, 1.0 / sub_rate[0]);
        return c_agg;
    }

    // // Generate action masks
    __device__ void CudaGenerateActionMasks(
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents,
        const int kAgentArrayIdx,
        const int kActionLen,
        const int kNumSavingsactions,
        const int kNumDiscreteActionLevels,
        const bool kNegotiationOn,
        int *action_masks_arr,
        float *minimum_mitigation_rate_all_regions)
    {
        if (kNegotiationOn)
        {
            // Currently, the masks are only updated
            // when negotiation is turned on.
            // Otherwise, we use the default (all-ones) mask.
            const int mask_arr_idx_offset = kEnvId * kNumAgents * kActionLen +
                                            kAgentId * kActionLen +
                                            kNumSavingsactions *
                                                kNumDiscreteActionLevels;
            int minimum_mitigation_rate =
                lroundf(minimum_mitigation_rate_all_regions[kAgentArrayIdx] * kNumDiscreteActionLevels);
            for (int idx = 0; idx < kNumDiscreteActionLevels; idx++)
            {
                action_masks_arr[mask_arr_idx_offset + idx] = (idx < minimum_mitigation_rate) ? 0 : 1;
            }
        }
    }

    // Generate observation
    __device__ void CudaGenerateObservation(
        float *abatement_cost_all_regions,
        int *activity_timestep,
        float *capital_all_regions,
        float *capital_depreciation_all_regions,
        float *consumption_all_regions,
        float *current_balance_all_regions,
        float *damages_all_regions,
        float *global_carbon_mass,
        float *global_exogenous_emissions,
        float *global_land_emissions,
        float *global_temperature,
        float *gross_output_all_regions,
        float *intensity_all_regions,
        float *investment_all_regions,
        float *labor_all_regions,
        float *max_export_limit_all_regions,
        float *minimum_mitigation_rate_all_regions,
        float *mitigation_cost_all_regions,
        float *mitigation_rate_all_regions,
        float *production_all_regions,
        float *production_factor_all_regions,
        float *promised_mitigation_rate,
        float *requested_mitigation_rate,
        float *proposal_decisions,
        float *reward_all_regions,
        float *savings_all_regions,
        float *social_welfare_all_regions,
        int *stage,
        int *timestep,
        float *tariffs,
        float *utility_all_regions,
        float abatement_cost_all_regions_norm,
        float activity_timestep_norm,
        float capital_all_regions_norm,
        float capital_depreciation_all_regions_norm,
        float consumption_all_regions_norm,
        float current_balance_all_regions_norm,
        float damages_all_regions_norm,
        float global_carbon_mass_norm,
        float global_exogenous_emissions_norm,
        float global_land_emissions_norm,
        float global_temperature_norm,
        float gross_output_all_regions_norm,
        float intensity_all_regions_norm,
        float investment_all_regions_norm,
        float labor_all_regions_norm,
        float max_export_limit_all_regions_norm,
        float minimum_mitigation_rate_all_regions_norm,
        float mitigation_cost_all_regions_norm,
        float mitigation_rate_all_regions_norm,
        float production_all_regions_norm,
        float production_factor_all_regions_norm,
        float promised_mitigation_rate_norm,
        float proposal_decisions_norm,
        float requested_mitigation_rate_norm,
        float reward_all_regions_norm,
        float savings_all_regions_norm,
        float social_welfare_all_regions_norm,
        float stage_norm,
        float tariffs_norm,
        float timestep_norm,
        float utility_all_regions_norm,
        const int kNumSavingsActions,
        const int kNumDiscreteActionLevels,
        const bool kNegotiationOn,
        float *observations_arr,
        int *action_masks_arr,
        const int global_temperature_len,
        const int global_carbon_mass_len,
        const int global_exogenous_emissions_len,
        const int global_land_emissions_len,
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents,
        const int kAgentArrayIdx,
        const int kNumFeatures,
        const int kActionLen)
    {
        int obs_arr_idx_offset = kEnvId * kNumAgents * kNumFeatures +
                                 kAgentId * kNumFeatures;

        // region indicator
        observations_arr[obs_arr_idx_offset + kAgentId] = 1.0;
        obs_arr_idx_offset += kNumAgents;

        // global features
        for (int idx = 0; idx < global_temperature_len; idx++)
        {
            observations_arr[obs_arr_idx_offset + idx] =
                global_temperature[kEnvId * global_temperature_len + idx] /
                global_temperature_norm;
        }
        obs_arr_idx_offset += global_temperature_len;

        for (int idx = 0; idx < global_carbon_mass_len; idx++)
        {
            observations_arr[obs_arr_idx_offset + idx] =
                global_carbon_mass[kEnvId * global_carbon_mass_len + idx] /
                global_carbon_mass_norm;
        }
        obs_arr_idx_offset += global_carbon_mass_len;

        for (int idx = 0; idx < global_exogenous_emissions_len; idx++)
        {
            observations_arr[obs_arr_idx_offset + idx] =
                global_exogenous_emissions[kEnvId * global_exogenous_emissions_len + idx] /
                global_exogenous_emissions_norm;
        }
        obs_arr_idx_offset += global_exogenous_emissions_len;

        for (int idx = 0; idx < global_land_emissions_len; idx++)
        {
            observations_arr[obs_arr_idx_offset + idx] =
                global_land_emissions[kEnvId * global_land_emissions_len + idx] /
                global_land_emissions_norm;
        }
        obs_arr_idx_offset += global_land_emissions_len;

        observations_arr[obs_arr_idx_offset] = timestep[kEnvId] /
                                               timestep_norm;
        obs_arr_idx_offset += 1;

        if (kNegotiationOn)
        {
            observations_arr[obs_arr_idx_offset] = stage[kEnvId] / stage_norm;
            obs_arr_idx_offset += 1;
        }

        // public features
        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                capital_all_regions[feature_idx] / capital_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                capital_depreciation_all_regions[feature_idx] /
                capital_depreciation_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                labor_all_regions[feature_idx] / labor_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                gross_output_all_regions[feature_idx] /
                gross_output_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                investment_all_regions[feature_idx] /
                investment_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                consumption_all_regions[feature_idx] /
                consumption_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                savings_all_regions[feature_idx] /
                savings_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                mitigation_rate_all_regions[feature_idx] /
                mitigation_rate_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                max_export_limit_all_regions[feature_idx] /
                max_export_limit_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int idx = 0; idx < kNumAgents; idx++)
        {
            int feature_idx = kEnvId * kNumAgents + idx;
            observations_arr[obs_arr_idx_offset + idx] =
                current_balance_all_regions[feature_idx] /
                current_balance_all_regions_norm;
        }
        obs_arr_idx_offset += kNumAgents;

        for (int i = 0; i < kNumAgents; i++)
        {
            for (int j = 0; j < kNumAgents; j++)
            {
                int tariff_idx = i * kNumAgents + j;
                observations_arr[obs_arr_idx_offset + tariff_idx] =
                    tariffs[kEnvId * kNumAgents * kNumAgents + tariff_idx] /
                    tariffs_norm;
            }
        }
        obs_arr_idx_offset += kNumAgents * kNumAgents;

        // private features
        observations_arr[obs_arr_idx_offset] =
            production_factor_all_regions[kAgentArrayIdx] /
            production_factor_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            intensity_all_regions[kAgentArrayIdx] /
            intensity_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            mitigation_cost_all_regions[kAgentArrayIdx] /
            mitigation_cost_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            damages_all_regions[kAgentArrayIdx] /
            damages_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            abatement_cost_all_regions[kAgentArrayIdx] /
            abatement_cost_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            production_all_regions[kAgentArrayIdx] /
            production_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            utility_all_regions[kAgentArrayIdx] /
            utility_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            social_welfare_all_regions[kAgentArrayIdx] /
            social_welfare_all_regions_norm;
        obs_arr_idx_offset += 1;

        observations_arr[obs_arr_idx_offset] =
            reward_all_regions[kAgentArrayIdx] /
            reward_all_regions_norm;
        obs_arr_idx_offset += 1;

        if (kNegotiationOn)
        {
            // private features
            observations_arr[obs_arr_idx_offset] =
                minimum_mitigation_rate_all_regions[kAgentArrayIdx] /
                minimum_mitigation_rate_all_regions_norm;
            obs_arr_idx_offset += 1;

            // bilateral features
            for (int idx = 0; idx < kNumAgents; idx++)
            {
                int feature_idx = kEnvId * kNumAgents * kNumAgents +
                                  kAgentId * kNumAgents + idx;
                observations_arr[obs_arr_idx_offset + idx] =
                    promised_mitigation_rate[feature_idx] /
                    promised_mitigation_rate_norm;
            }
            obs_arr_idx_offset += kNumAgents;
            for (int idx = 0; idx < kNumAgents; idx++)
            {
                int feature_idx = kEnvId * kNumAgents * kNumAgents +
                                  idx * kNumAgents + kAgentId;
                observations_arr[obs_arr_idx_offset + idx] =
                    promised_mitigation_rate[feature_idx] /
                    promised_mitigation_rate_norm;
            }
            obs_arr_idx_offset += kNumAgents;

            for (int idx = 0; idx < kNumAgents; idx++)
            {
                int feature_idx = kEnvId * kNumAgents * kNumAgents +
                                  kAgentId * kNumAgents + idx;
                observations_arr[obs_arr_idx_offset + idx] =
                    requested_mitigation_rate[feature_idx] /
                    requested_mitigation_rate_norm;
            }
            obs_arr_idx_offset += kNumAgents;
            for (int idx = 0; idx < kNumAgents; idx++)
            {
                int feature_idx = kEnvId * kNumAgents * kNumAgents +
                                  idx * kNumAgents + kAgentId;
                observations_arr[obs_arr_idx_offset + idx] =
                    requested_mitigation_rate[feature_idx] /
                    requested_mitigation_rate_norm;
            }
            obs_arr_idx_offset += kNumAgents;

            for (int idx = 0; idx < kNumAgents; idx++)
            {
                int feature_idx = kEnvId * kNumAgents * kNumAgents +
                                  kAgentId * kNumAgents + idx;
                observations_arr[obs_arr_idx_offset + idx] =
                    proposal_decisions[feature_idx] /
                    proposal_decisions_norm;
            }
            obs_arr_idx_offset += kNumAgents;
            for (int idx = 0; idx < kNumAgents; idx++)
            {
                int feature_idx = kEnvId * kNumAgents * kNumAgents +
                                  idx * kNumAgents + kAgentId;
                observations_arr[obs_arr_idx_offset + idx] =
                    proposal_decisions[feature_idx] /
                    proposal_decisions_norm;
            }
            obs_arr_idx_offset += kNumAgents;
        }

        CudaGenerateActionMasks(
            kEnvId,
            kAgentId,
            kNumAgents,
            kAgentArrayIdx,
            kActionLen,
            kNumSavingsActions,
            kNumDiscreteActionLevels,
            kNegotiationOn,
            action_masks_arr,
            minimum_mitigation_rate_all_regions);
    }

    __device__ void CudaProposalStep(
        float *abatement_cost_all_regions,
        int *activity_timestep,
        float *capital_all_regions,
        float *capital_depreciation_all_regions,
        float *consumption_all_regions,
        float *current_balance_all_regions,
        float *damages_all_regions,
        float *global_carbon_mass,
        float *global_exogenous_emissions,
        float *global_land_emissions,
        float *global_temperature,
        float *gross_output_all_regions,
        float *intensity_all_regions,
        float *investment_all_regions,
        float *labor_all_regions,
        float *max_export_limit_all_regions,
        float *minimum_mitigation_rate_all_regions,
        float *mitigation_cost_all_regions,
        float *mitigation_rate_all_regions,
        float *production_all_regions,
        float *production_factor_all_regions,
        float *promised_mitigation_rate,
        float *requested_mitigation_rate,
        float *proposal_decisions,
        float *reward_all_regions,
        float *savings_all_regions,
        float *social_welfare_all_regions,
        int *stage,
        int *timestep,
        float *tariffs,
        float *utility_all_regions,
        float abatement_cost_all_regions_norm,
        float activity_timestep_norm,
        float capital_all_regions_norm,
        float capital_depreciation_all_regions_norm,
        float consumption_all_regions_norm,
        float current_balance_all_regions_norm,
        float damages_all_regions_norm,
        float global_carbon_mass_norm,
        float global_exogenous_emissions_norm,
        float global_land_emissions_norm,
        float global_temperature_norm,
        float gross_output_all_regions_norm,
        float intensity_all_regions_norm,
        float investment_all_regions_norm,
        float labor_all_regions_norm,
        float max_export_limit_all_regions_norm,
        float minimum_mitigation_rate_all_regions_norm,
        float mitigation_cost_all_regions_norm,
        float mitigation_rate_all_regions_norm,
        float production_all_regions_norm,
        float production_factor_all_regions_norm,
        float promised_mitigation_rate_norm,
        float proposal_decisions_norm,
        float requested_mitigation_rate_norm,
        float reward_all_regions_norm,
        float savings_all_regions_norm,
        float social_welfare_all_regions_norm,
        float stage_norm,
        float tariffs_norm,
        float timestep_norm,
        float utility_all_regions_norm,
        float *observations_arr,
        int *action_masks_arr,
        int *actions_arr,
        float *rewards_arr,
        int *done_arr,
        int *env_timestep_arr,
        const int kNumDiscreteActionLevels,
        const bool kNegotiationOn,
        const int global_temperature_len,
        const int global_carbon_mass_len,
        const int global_exogenous_emissions_len,
        const int global_land_emissions_len,
        const int kNumMitigationRateActions,
        const int kNumSavingsActions,
        const int kNumExportActions,
        const int kNumImportActions,
        const int kNumTariffActions,
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents,
        const int kAgentArrayIdx,
        const int kNumFeatures,
        const int kActionLen,
        const int kNumActions,
        const int kEpisodeLength)
    {
        if (kAgentId < kNumAgents)
        {
            assert(kNegotiationOn);
            assert(stage[kEnvId] == 1);

            // Update proposal values corresponding to this Agents proposals
            const int kPropArrIdxOffset = kEnvId * kNumAgents * kNumAgents +
                                          kAgentId * kNumAgents;
            const int kActionIdxOffset = kNumSavingsActions +
                                         kNumMitigationRateActions +
                                         kNumExportActions +
                                         kNumImportActions +
                                         kNumTariffActions;
            const int kActionArrIdxOffset =
                kEnvId * kNumAgents * kNumActions +
                kAgentId * kNumActions + kActionIdxOffset;

            for (int idx = 0; idx < kNumAgents; idx++)
            {
                promised_mitigation_rate[kPropArrIdxOffset + idx] = static_cast<float>(
                                                                        actions_arr[kActionArrIdxOffset + 2 * idx + 0]) /
                                                                    kNumDiscreteActionLevels;
            }

            for (int idx = 0; idx < kNumAgents; idx++)
            {
                requested_mitigation_rate[kPropArrIdxOffset + idx] = static_cast<float>(
                                                                         actions_arr[kActionArrIdxOffset + 2 * idx + 1]) /
                                                                     kNumDiscreteActionLevels;
            }


            // Wait here for all agents to update promised_mitigation_rate
            // and requested_mitigation_rate.
            __syncthreads();

            // -------------------------------
            // Generate observation
            // -------------------------------
            CudaGenerateObservation(
                abatement_cost_all_regions,
                activity_timestep,
                capital_all_regions,
                capital_depreciation_all_regions,
                consumption_all_regions,
                current_balance_all_regions,
                damages_all_regions,
                global_carbon_mass,
                global_exogenous_emissions,
                global_land_emissions,
                global_temperature,
                gross_output_all_regions,
                intensity_all_regions,
                investment_all_regions,
                labor_all_regions,
                max_export_limit_all_regions,
                minimum_mitigation_rate_all_regions,
                mitigation_cost_all_regions,
                mitigation_rate_all_regions,
                production_all_regions,
                production_factor_all_regions,
                promised_mitigation_rate,
                requested_mitigation_rate,
                proposal_decisions,
                reward_all_regions,
                savings_all_regions,
                social_welfare_all_regions,
                stage,
                timestep,
                tariffs,
                utility_all_regions,
                abatement_cost_all_regions_norm,
                activity_timestep_norm,
                capital_all_regions_norm,
                capital_depreciation_all_regions_norm,
                consumption_all_regions_norm,
                current_balance_all_regions_norm,
                damages_all_regions_norm,
                global_carbon_mass_norm,
                global_exogenous_emissions_norm,
                global_land_emissions_norm,
                global_temperature_norm,
                gross_output_all_regions_norm,
                intensity_all_regions_norm,
                investment_all_regions_norm,
                labor_all_regions_norm,
                max_export_limit_all_regions_norm,
                minimum_mitigation_rate_all_regions_norm,
                mitigation_cost_all_regions_norm,
                mitigation_rate_all_regions_norm,
                production_all_regions_norm,
                production_factor_all_regions_norm,
                promised_mitigation_rate_norm,
                proposal_decisions_norm,
                requested_mitigation_rate_norm,
                reward_all_regions_norm,
                savings_all_regions_norm,
                social_welfare_all_regions_norm,
                stage_norm,
                tariffs_norm,
                timestep_norm,
                utility_all_regions_norm,
                kNumSavingsActions,
                kNumDiscreteActionLevels,
                kNegotiationOn,
                observations_arr,
                action_masks_arr,
                global_temperature_len,
                global_carbon_mass_len,
                global_exogenous_emissions_len,
                global_land_emissions_len,
                kEnvId,
                kAgentId,
                kNumAgents,
                kAgentArrayIdx,
                kNumFeatures,
                kActionLen);

            // Update rewards array
            rewards_arr[kAgentArrayIdx] = 0.0;

            // Wait here for all agents before checking for the done condition
            __syncthreads();
        }

        // Use only agent 0's thread to set done_arr
        if (kAgentId == 0)
        {
            done_arr[kEnvId] = 0;
        }
    }

    __device__ void CudaEvaluationStep(
        float *abatement_cost_all_regions,
        int *activity_timestep,
        float *capital_all_regions,
        float *capital_depreciation_all_regions,
        float *consumption_all_regions,
        float *current_balance_all_regions,
        float *damages_all_regions,
        float *global_carbon_mass,
        float *global_exogenous_emissions,
        float *global_land_emissions,
        float *global_temperature,
        float *gross_output_all_regions,
        float *intensity_all_regions,
        float *investment_all_regions,
        float *labor_all_regions,
        float *max_export_limit_all_regions,
        float *minimum_mitigation_rate_all_regions,
        float *mitigation_cost_all_regions,
        float *mitigation_rate_all_regions,
        float *production_all_regions,
        float *production_factor_all_regions,
        float *promised_mitigation_rate,
        float *requested_mitigation_rate,
        float *proposal_decisions,
        float *reward_all_regions,
        float *savings_all_regions,
        float *social_welfare_all_regions,
        int *stage,
        int *timestep,
        float *tariffs,
        float *utility_all_regions,
        float abatement_cost_all_regions_norm,
        float activity_timestep_norm,
        float capital_all_regions_norm,
        float capital_depreciation_all_regions_norm,
        float consumption_all_regions_norm,
        float current_balance_all_regions_norm,
        float damages_all_regions_norm,
        float global_carbon_mass_norm,
        float global_exogenous_emissions_norm,
        float global_land_emissions_norm,
        float global_temperature_norm,
        float gross_output_all_regions_norm,
        float intensity_all_regions_norm,
        float investment_all_regions_norm,
        float labor_all_regions_norm,
        float max_export_limit_all_regions_norm,
        float minimum_mitigation_rate_all_regions_norm,
        float mitigation_cost_all_regions_norm,
        float mitigation_rate_all_regions_norm,
        float production_all_regions_norm,
        float production_factor_all_regions_norm,
        float promised_mitigation_rate_norm,
        float proposal_decisions_norm,
        float requested_mitigation_rate_norm,
        float reward_all_regions_norm,
        float savings_all_regions_norm,
        float social_welfare_all_regions_norm,
        float stage_norm,
        float tariffs_norm,
        float timestep_norm,
        float utility_all_regions_norm,
        float *observations_arr,
        int *action_masks_arr,
        int *actions_arr,
        float *rewards_arr,
        int *done_arr,
        int *env_timestep_arr,
        const int kNumDiscreteActionLevels,
        const bool kNegotiationOn,
        const int global_temperature_len,
        const int global_carbon_mass_len,
        const int global_exogenous_emissions_len,
        const int global_land_emissions_len,
        const int kNumMitigationRateActions,
        const int kNumSavingsActions,
        const int kNumExportActions,
        const int kNumImportActions,
        const int kNumTariffActions,
        const int kNumProposalActions,
        const int kNumEvaluationActions,
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents,
        const int kAgentArrayIdx,
        const int kNumFeatures,
        const int kActionLen,
        const int kNumActions,
        const int kEpisodeLength)
    {
        if (kAgentId < kNumAgents)
        {
            assert(kNegotiationOn);
            assert(stage[kEnvId] == 2);

            // Update proposal_decisions data
            const int kDecisionArrIdxOffset = kEnvId * kNumAgents * kNumAgents;

            const int kActionIdxOffset = kNumSavingsActions +
                                         kNumMitigationRateActions +
                                         kNumExportActions +
                                         kNumImportActions +
                                         kNumTariffActions +
                                         kNumProposalActions;
            const int kActionArrIdxOffset =
                kEnvId * kNumAgents * kNumActions +
                kAgentId * kNumActions + kActionIdxOffset;

            for (int idx = 0; idx < kNumEvaluationActions; idx++)
            {
                proposal_decisions[kDecisionArrIdxOffset +
                                   kAgentId * kNumAgents + idx] =
                    static_cast<float>(
                        actions_arr[kActionArrIdxOffset + idx]);
            }

            // Force set the evaluation for own proposal to reject
            proposal_decisions[kDecisionArrIdxOffset +
                               kAgentId * kNumAgents + kAgentId] = 0;

            // Wait here for all agents to update proposal_decisions
            __syncthreads();

            // Use promised_mitigation_rate, requested_mitigation_rate,
            // and proposal_decisions to compute minimum_mitigation_rate

            float min_mutmp = 0;
            for (int idx = 0; idx < kNumAgents; idx++)
            {
                min_mutmp = fmaxf(
                    min_mutmp,
                    promised_mitigation_rate[kDecisionArrIdxOffset + kAgentId * kNumAgents + idx] * proposal_decisions[kDecisionArrIdxOffset + idx * kNumAgents + kAgentId]);
            }

            for (int idx = 0; idx < kNumAgents; idx++)
            {
                min_mutmp = fmaxf(
                    min_mutmp,
                    requested_mitigation_rate[kDecisionArrIdxOffset + idx * kNumAgents + kAgentId] * proposal_decisions[kDecisionArrIdxOffset + kAgentId * kNumAgents + idx]);
            }

            minimum_mitigation_rate_all_regions[kAgentArrayIdx] = min_mutmp;

            // Wait here for all agents to process
            __syncthreads();

            // -------------------------------
            // Generate observation
            // -------------------------------
            CudaGenerateObservation(
                abatement_cost_all_regions,
                activity_timestep,
                capital_all_regions,
                capital_depreciation_all_regions,
                consumption_all_regions,
                current_balance_all_regions,
                damages_all_regions,
                global_carbon_mass,
                global_exogenous_emissions,
                global_land_emissions,
                global_temperature,
                gross_output_all_regions,
                intensity_all_regions,
                investment_all_regions,
                labor_all_regions,
                max_export_limit_all_regions,
                minimum_mitigation_rate_all_regions,
                mitigation_cost_all_regions,
                mitigation_rate_all_regions,
                production_all_regions,
                production_factor_all_regions,
                promised_mitigation_rate,
                requested_mitigation_rate,
                proposal_decisions,
                reward_all_regions,
                savings_all_regions,
                social_welfare_all_regions,
                stage,
                timestep,
                tariffs,
                utility_all_regions,
                abatement_cost_all_regions_norm,
                activity_timestep_norm,
                capital_all_regions_norm,
                capital_depreciation_all_regions_norm,
                consumption_all_regions_norm,
                current_balance_all_regions_norm,
                damages_all_regions_norm,
                global_carbon_mass_norm,
                global_exogenous_emissions_norm,
                global_land_emissions_norm,
                global_temperature_norm,
                gross_output_all_regions_norm,
                intensity_all_regions_norm,
                investment_all_regions_norm,
                labor_all_regions_norm,
                max_export_limit_all_regions_norm,
                minimum_mitigation_rate_all_regions_norm,
                mitigation_cost_all_regions_norm,
                mitigation_rate_all_regions_norm,
                production_all_regions_norm,
                production_factor_all_regions_norm,
                promised_mitigation_rate_norm,
                proposal_decisions_norm,
                requested_mitigation_rate_norm,
                reward_all_regions_norm,
                savings_all_regions_norm,
                social_welfare_all_regions_norm,
                stage_norm,
                tariffs_norm,
                timestep_norm,
                utility_all_regions_norm,
                kNumSavingsActions,
                kNumDiscreteActionLevels,
                kNegotiationOn,
                observations_arr,
                action_masks_arr,
                global_temperature_len,
                global_carbon_mass_len,
                global_exogenous_emissions_len,
                global_land_emissions_len,
                kEnvId,
                kAgentId,
                kNumAgents,
                kAgentArrayIdx,
                kNumFeatures,
                kActionLen);

            // Update rewards array
            rewards_arr[kAgentArrayIdx] = 0.0;

            // Wait here for all agents before checking for the done condition
            __syncthreads();
        }

        // Use only agent 0's thread to set done_arr
        if (kAgentId == 0)
        {
            done_arr[kEnvId] = 0;
        }
    }

    __device__ void CudaActivityStep(
        const int xt_0,
        const int xdelta,
        const int xN,
        const float *xphi_t,
        const float *xb_t,
        const float *xphi_m,
        const float *xb_m,
        const float xeta,
        const int xm_at_1750,
        const float xf_0,
        const float xf_1,
        const int xt_f,
        const float xe_l0,
        const float xdelta_el,
        const int xM_AT_0,
        const int xM_UP_0,
        const int xM_LO_0,
        const float xe_0,
        const float xq_0,
        const float xmu_0,
        const float xf_2x,
        const float xT_2x,
        const float *xgamma,
        const float *xtheta_2,
        const float *xa_1,
        const float *xa_2,
        const int *xa_3,
        const float *xdelta_K,
        const float *xalpha,
        const float *xrho,
        const int *xL_0,
        const float *xL_a,
        const float *xl_g,
        const float *xA_0,
        const float *xg_a,
        const float *xdelta_a,
        const float *xsigma_0,
        const float *xg_sigma,
        const float *xdelta_sigma,
        const int *xp_b,
        const float *xdelta_pb,
        const float *xscale_1,
        const float *xscale_2,
        const float *xT_AT_0,
        const float *xT_LO_0,
        const float *xK_0,
        float *abatement_cost_all_regions,
        int *activity_timestep,
        float *capital_all_regions,
        float *capital_depreciation_all_regions,
        float *consumption_all_regions,
        float *current_balance_all_regions,
        float *damages_all_regions,
        float *global_carbon_mass,
        float *global_exogenous_emissions,
        float *global_land_emissions,
        float *global_temperature,
        float *gross_output_all_regions,
        float *intensity_all_regions,
        float *investment_all_regions,
        float *labor_all_regions,
        float *max_export_limit_all_regions,
        float *minimum_mitigation_rate_all_regions,
        float *mitigation_cost_all_regions,
        float *mitigation_rate_all_regions,
        float *production_all_regions,
        float *production_factor_all_regions,
        float *promised_mitigation_rate,
        float *requested_mitigation_rate,
        float *proposal_decisions,
        float *reward_all_regions,
        float *savings_all_regions,
        float *social_welfare_all_regions,
        int *stage,
        int *timestep,
        float *tariffs,
        float *desired_imports,
        float *scaled_imports,
        float *tariffed_imports,
        float *future_tariffs,
        float *utility_all_regions,
        float abatement_cost_all_regions_norm,
        float activity_timestep_norm,
        float capital_all_regions_norm,
        float capital_depreciation_all_regions_norm,
        float consumption_all_regions_norm,
        float current_balance_all_regions_norm,
        float damages_all_regions_norm,
        float global_carbon_mass_norm,
        float global_exogenous_emissions_norm,
        float global_land_emissions_norm,
        float global_temperature_norm,
        float gross_output_all_regions_norm,
        float intensity_all_regions_norm,
        float investment_all_regions_norm,
        float labor_all_regions_norm,
        float max_export_limit_all_regions_norm,
        float minimum_mitigation_rate_all_regions_norm,
        float mitigation_cost_all_regions_norm,
        float mitigation_rate_all_regions_norm,
        float production_all_regions_norm,
        float production_factor_all_regions_norm,
        float promised_mitigation_rate_norm,
        float proposal_decisions_norm,
        float requested_mitigation_rate_norm,
        float reward_all_regions_norm,
        float savings_all_regions_norm,
        float social_welfare_all_regions_norm,
        float stage_norm,
        float tariffs_norm,
        float timestep_norm,
        float utility_all_regions_norm,
        float *aux_ms,
        const float *sub_rate,
        const float *dom_pref,
        const float *for_pref,
        float *observations_arr,
        int *action_masks_arr,
        int *actions_arr,
        float *rewards_arr,
        int *done_arr,
        int *env_timestep_arr,
        int *current_year,
        const int kEndYear,
        const int kNumDiscreteActionLevels,
        const float kBalanceInterestRate,
        const bool kNegotiationOn,
        const int global_temperature_len,
        const int global_carbon_mass_len,
        const int global_exogenous_emissions_len,
        const int global_land_emissions_len,
        const int kNumSavingsActions,
        const int kNumMitigationRateActions,
        const int kNumExportActions,
        const int kNumImportActions,
        const int kNumTariffActions,
        const int kNumProposalActions,
        const int kNumEvaluationActions,
        const int kEnvId,
        const int kAgentId,
        const int kNumAgents,
        const int kAgentArrayIdx,
        const int kNumFeatures,
        const int kActionLen,
        const int kNumActions,
        const int kEpisodeLength)
    {
        // Perform this only once per env
        if (kAgentId == 0)
        {
            activity_timestep[kEnvId] += 1;

            // Compute global features
            global_exogenous_emissions[kEnvId] = get_exogenous_emissions(
                xf_0,
                xf_1,
                xt_f,
                activity_timestep[kEnvId]);

            global_land_emissions[kEnvId] = get_land_emissions(
                xe_l0,
                xdelta_el,
                activity_timestep[kEnvId],
                kNumAgents);
        }

        // Wait here until timestep has been updated
        __syncthreads();

        if (kNegotiationOn)
        {
            assert(stage[kEnvId] == 0);
        }
        else
        {
            assert(activity_timestep[kEnvId] == env_timestep_arr[kEnvId]);
        }

        if (kAgentId < kNumAgents)
        {
            const int kActionIdxOffset =
                kEnvId * kNumAgents * kNumActions +
                kAgentId * kNumActions;

            const int kSavingsActionIdx = 0;
            savings_all_regions[kAgentArrayIdx] =
                static_cast<float>(
                    actions_arr[kActionIdxOffset + kSavingsActionIdx]) /
                kNumDiscreteActionLevels;

            const int kMitigationRateActionidx = kSavingsActionIdx +
                                                 kNumSavingsActions;
            mitigation_rate_all_regions[kAgentArrayIdx] =
                static_cast<float>(
                    actions_arr[kActionIdxOffset + kMitigationRateActionidx]) /
                kNumDiscreteActionLevels;

            const int kExportActionidx = kMitigationRateActionidx +
                                         kNumMitigationRateActions;
            max_export_limit_all_regions[kAgentArrayIdx] =
                static_cast<float>(
                    actions_arr[kActionIdxOffset + kExportActionidx]) /
                kNumDiscreteActionLevels;

            const int kTariffActionIdx = kExportActionidx + kNumExportActions;
            for (
                int region_id = 0; region_id < kNumTariffActions; region_id++)
            {
                future_tariffs[kAgentArrayIdx * kNumAgents + region_id] =
                    static_cast<float>(
                        actions_arr[kActionIdxOffset +
                                    kTariffActionIdx +
                                    region_id]) /
                    (kNumDiscreteActionLevels);
            }

            const int kImportsActionIdx = kTariffActionIdx + kNumTariffActions;
            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                desired_imports[kAgentArrayIdx * kNumAgents + region_id] =
                    static_cast<float>(
                        actions_arr[kActionIdxOffset +
                                    kImportsActionIdx +
                                    region_id]) /
                    (kNumDiscreteActionLevels);
            }

            mitigation_cost_all_regions[kAgentArrayIdx] =
                get_mitigation_cost(
                    xp_b[kAgentId],
                    xtheta_2[kAgentId],
                    xdelta_pb[kAgentId],
                    activity_timestep[kEnvId],
                    intensity_all_regions[kAgentArrayIdx]);

            const int kTArrayIdxOffset = kEnvId * global_temperature_len;

            damages_all_regions[kAgentArrayIdx] = get_damages(
                global_temperature[kTArrayIdxOffset],
                xa_1[kAgentId],
                xa_2[kAgentId],
                xa_3[kAgentId]);

            abatement_cost_all_regions[kAgentArrayIdx] = get_abatement_cost(
                mitigation_rate_all_regions[kAgentArrayIdx],
                mitigation_cost_all_regions[kAgentArrayIdx],
                xtheta_2[kAgentId]);

            production_all_regions[kAgentArrayIdx] = get_production(
                production_factor_all_regions[kAgentArrayIdx],
                capital_all_regions[kAgentArrayIdx],
                labor_all_regions[kAgentArrayIdx],
                xgamma[kAgentId]);

            gross_output_all_regions[kAgentArrayIdx] = get_gross_output(
                damages_all_regions[kAgentArrayIdx],
                abatement_cost_all_regions[kAgentArrayIdx],
                production_all_regions[kAgentArrayIdx]);

            // float tmp = gross_output_all_regions[kAgentArrayIdx];
            // int tmp1 = int(tmp * 100 + 0.5);
            // tmp = float(tmp1) / 100;
            // gross_output_all_regions[kAgentArrayIdx] = tmp;

            float gov_balance_prev =
                current_balance_all_regions[kAgentArrayIdx] *
                (1 + kBalanceInterestRate);

            current_balance_all_regions[kAgentArrayIdx] = gov_balance_prev;

            investment_all_regions[kAgentArrayIdx] = get_investment(
                savings_all_regions[kAgentArrayIdx],
                gross_output_all_regions[kAgentArrayIdx]);

            float max_potential_exports = get_max_potential_exports(
                max_export_limit_all_regions[kAgentArrayIdx],
                gross_output_all_regions[kAgentArrayIdx],
                investment_all_regions[kAgentArrayIdx]);

            __syncthreads();

            int scaled_imports_arr_shape[] =
                {gridDim.x, kNumAgents, kNumAgents};

            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                int agent_arr_index_ij[] = {kEnvId, kAgentId, region_id};
                int dimension = 3;
                int ij_index = get_flattened_array_index(
                    agent_arr_index_ij,
                    scaled_imports_arr_shape,
                    dimension);
                scaled_imports[ij_index] = desired_imports[ij_index] *
                                           gross_output_all_regions[kAgentArrayIdx];
            }

            // Set import bid to self as zero
            int agent_arr_index_ij[] = {kEnvId, kAgentId, kAgentId};
            int dimension = 3;
            int ij_index = get_flattened_array_index(
                agent_arr_index_ij,
                scaled_imports_arr_shape,
                dimension);
            scaled_imports[ij_index] = 0.0;

            float total_scaled_imports = 0.0;
            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                int agent_arr_index_ij[] = {kEnvId, kAgentId, region_id};
                int dimension = 3;
                int ij_index = get_flattened_array_index(
                    agent_arr_index_ij,
                    scaled_imports_arr_shape,
                    dimension);
                total_scaled_imports += scaled_imports[ij_index];
            }

            if (total_scaled_imports >
                gross_output_all_regions[kAgentArrayIdx])
            {
                for (int region_id = 0; region_id < kNumAgents; region_id++)
                {
                    int agent_arr_index_ij[] = {kEnvId, kAgentId, region_id};
                    int dimension = 3;
                    int ij_index = get_flattened_array_index(
                        agent_arr_index_ij,
                        scaled_imports_arr_shape,
                        dimension);
                    scaled_imports[ij_index] = scaled_imports[ij_index] /
                                               total_scaled_imports *
                                               gross_output_all_regions[kAgentArrayIdx];
                }
            }

            int init_capital_multiplier = 10;
            float debt_ratio =
                gov_balance_prev / init_capital_multiplier * xK_0[kAgentId];
            debt_ratio = fminf(0, debt_ratio);
            debt_ratio = fmaxf(-1, debt_ratio);

            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                int agent_arr_index_ij[] = {kEnvId, kAgentId, region_id};
                int dimension = 3;
                int ij_index = get_flattened_array_index(
                    agent_arr_index_ij,
                    scaled_imports_arr_shape,
                    dimension);
                scaled_imports[ij_index] = scaled_imports[ij_index] *
                                           (1 + debt_ratio);
            }

            __syncthreads();

            float total_desired_exports = 0.0;
            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                int agent_arr_index_ji[] = {kEnvId, region_id, kAgentId};
                int dimension = 3;
                int ji_index = get_flattened_array_index(
                    agent_arr_index_ji,
                    scaled_imports_arr_shape,
                    dimension);
                total_desired_exports += scaled_imports[ji_index];
            }

            if (total_desired_exports > max_potential_exports)
            {
                for (int region_id = 0;
                     region_id < kNumAgents;
                     region_id++)
                {
                    int agent_arr_index_ji[] =
                        {kEnvId, region_id, kAgentId};
                    int dimension = 3;
                    int ji_index = get_flattened_array_index(
                        agent_arr_index_ji,
                        scaled_imports_arr_shape,
                        dimension);
                    scaled_imports[ji_index] =
                        static_cast<float>(scaled_imports[ji_index]) /
                        static_cast<float>(total_desired_exports) *
                        max_potential_exports;
                }
            }

            __syncthreads();

            float tariff_revenue = 0.0;

            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                int agent_arr_index_ij[] =
                    {kEnvId, kAgentId, region_id};
                int dimension = 3;
                int ij_index = get_flattened_array_index(
                    agent_arr_index_ij,
                    scaled_imports_arr_shape,
                    dimension);
                tariffed_imports[ij_index] =
                    scaled_imports[ij_index] * (1 - tariffs[ij_index]);
            }

            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                int agent_arr_index_ij[] =
                    {kEnvId, kAgentId, region_id};
                int dimension = 3;
                int ij_index = get_flattened_array_index(
                    agent_arr_index_ij,
                    scaled_imports_arr_shape,
                    dimension);
                tariff_revenue +=
                    scaled_imports[ij_index] * tariffs[ij_index];
            }

            __syncthreads();

            float domestic_consumption = get_consumption(
                savings_all_regions[kAgentArrayIdx],
                gross_output_all_regions[kAgentArrayIdx],
                scaled_imports,
                kAgentId,
                kNumAgents);

            consumption_all_regions[kAgentArrayIdx] = get_armington_agg(
                domestic_consumption,
                tariffed_imports,
                sub_rate,
                dom_pref,
                for_pref,
                kEnvId,
                kAgentId,
                kNumAgents);

            utility_all_regions[kAgentArrayIdx] = get_utility(
                labor_all_regions[kAgentArrayIdx],
                consumption_all_regions[kAgentArrayIdx],
                xalpha[kAgentId]);

            social_welfare_all_regions[kAgentArrayIdx] = get_social_welfare(
                utility_all_regions[kAgentArrayIdx],
                xrho[kAgentId],
                xdelta,
                activity_timestep[kEnvId]);

            reward_all_regions[kAgentArrayIdx] =
                utility_all_regions[kAgentArrayIdx];

            gov_balance_prev = current_balance_all_regions[kAgentArrayIdx];
            float gov_balance = gov_balance_prev;
            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                int agent_arr_index_ij[] = {kEnvId, kAgentId, region_id};
                int agent_arr_index_ji[] = {kEnvId, region_id, kAgentId};
                int dimension = 3;
                int ji_index = get_flattened_array_index(
                    agent_arr_index_ji,
                    scaled_imports_arr_shape,
                    dimension);
                int ij_index = get_flattened_array_index(
                    agent_arr_index_ij,
                    scaled_imports_arr_shape,
                    dimension);
                gov_balance += xdelta * scaled_imports[ji_index];
                gov_balance -= xdelta * scaled_imports[ij_index];
            }
            current_balance_all_regions[kAgentArrayIdx] = gov_balance;

            // Wait here till all the previous computations are finished
            // by all the agent threads.
            __syncthreads();

            const int kMArrayIdxOffset = kEnvId * global_carbon_mass_len;
            float m_at = global_carbon_mass[kMArrayIdxOffset];

            update_global_temperature(
                xphi_t,
                global_temperature,
                xb_t,
                xf_2x,
                m_at,
                xm_at_1750,
                global_exogenous_emissions[kEnvId],
                global_temperature_len,
                kEnvId,
                kAgentId,
                kNumAgents);

            aux_ms[kAgentArrayIdx] = get_aux_m(
                intensity_all_regions[kAgentArrayIdx],
                mitigation_rate_all_regions[kAgentArrayIdx],
                production_all_regions[kAgentArrayIdx],
                global_land_emissions[kEnvId]);

            // Wait here till all the previous computations are finished
            // by all the agent threads.
            __syncthreads();

            float sum_aux_ms = 0;
            for (int idx = 0; idx < kNumAgents; idx++)
            {
                sum_aux_ms += aux_ms[kEnvId * kNumAgents + idx];
            }

            update_global_carbon_mass(
                xphi_m,
                global_carbon_mass,
                xb_m,
                sum_aux_ms,
                global_carbon_mass_len,
                kEnvId,
                kAgentId,
                kNumAgents);

            capital_depreciation_all_regions[kAgentArrayIdx] = pow(
                1 - xdelta_K[kAgentId], xdelta);

            capital_all_regions[kAgentArrayIdx] = get_capital(
                capital_depreciation_all_regions[kAgentArrayIdx],
                capital_all_regions[kAgentArrayIdx],
                xdelta,
                investment_all_regions[kAgentArrayIdx]);

            labor_all_regions[kAgentArrayIdx] = get_labor(
                labor_all_regions[kAgentArrayIdx],
                xL_a[kAgentId],
                xl_g[kAgentId]);

            production_factor_all_regions[kAgentArrayIdx] =
                get_production_factor(
                    production_factor_all_regions[kAgentArrayIdx],
                    xg_a[kAgentId],
                    xdelta_a[kAgentId],
                    xdelta,
                    activity_timestep[kEnvId]);

            intensity_all_regions[kAgentArrayIdx] = get_carbon_intensity(
                intensity_all_regions[kAgentArrayIdx],
                xg_sigma[kAgentId],
                xdelta_sigma[kAgentId],
                xdelta,
                activity_timestep[kEnvId]);

            for (int region_id = 0; region_id < kNumAgents; region_id++)
            {
                tariffs[kAgentArrayIdx * kNumAgents + region_id] =
                    future_tariffs[kAgentArrayIdx * kNumAgents + region_id];
            }

            __syncthreads(); // Wait here for all agents to process

            // -------------------------------
            // Generate observation
            // -------------------------------
            CudaGenerateObservation(
                abatement_cost_all_regions,
                activity_timestep,
                capital_all_regions,
                capital_depreciation_all_regions,
                consumption_all_regions,
                current_balance_all_regions,
                damages_all_regions,
                global_carbon_mass,
                global_exogenous_emissions,
                global_land_emissions,
                global_temperature,
                gross_output_all_regions,
                intensity_all_regions,
                investment_all_regions,
                labor_all_regions,
                max_export_limit_all_regions,
                minimum_mitigation_rate_all_regions,
                mitigation_cost_all_regions,
                mitigation_rate_all_regions,
                production_all_regions,
                production_factor_all_regions,
                promised_mitigation_rate,
                requested_mitigation_rate,
                proposal_decisions,
                reward_all_regions,
                savings_all_regions,
                social_welfare_all_regions,
                stage,
                timestep,
                tariffs,
                utility_all_regions,
                abatement_cost_all_regions_norm,
                activity_timestep_norm,
                capital_all_regions_norm,
                capital_depreciation_all_regions_norm,
                consumption_all_regions_norm,
                current_balance_all_regions_norm,
                damages_all_regions_norm,
                global_carbon_mass_norm,
                global_exogenous_emissions_norm,
                global_land_emissions_norm,
                global_temperature_norm,
                gross_output_all_regions_norm,
                intensity_all_regions_norm,
                investment_all_regions_norm,
                labor_all_regions_norm,
                max_export_limit_all_regions_norm,
                minimum_mitigation_rate_all_regions_norm,
                mitigation_cost_all_regions_norm,
                mitigation_rate_all_regions_norm,
                production_all_regions_norm,
                production_factor_all_regions_norm,
                promised_mitigation_rate_norm,
                proposal_decisions_norm,
                requested_mitigation_rate_norm,
                reward_all_regions_norm,
                savings_all_regions_norm,
                social_welfare_all_regions_norm,
                stage_norm,
                tariffs_norm,
                timestep_norm,
                utility_all_regions_norm,
                kNumSavingsActions,
                kNumDiscreteActionLevels,
                kNegotiationOn,
                observations_arr,
                action_masks_arr,
                global_temperature_len,
                global_carbon_mass_len,
                global_exogenous_emissions_len,
                global_land_emissions_len,
                kEnvId,
                kAgentId,
                kNumAgents,
                kAgentArrayIdx,
                kNumFeatures,
                kActionLen);

            // Update rewards array
            rewards_arr[kAgentArrayIdx] = reward_all_regions[kAgentArrayIdx];

            // Wait here for all agents before checking for the done condition
            __syncthreads();
        }

        // Use only agent 0's thread to set done_arr
        if (kAgentId == 0)
        {
            current_year[kEnvId] += xdelta;
            if (current_year[kEnvId] == kEndYear)
            {
                done_arr[kEnvId] = 1;
            }
        }
    }

    // CUDA version of the step() function in rice.py
    __global__ void CudaRiceStep(
        const float *xb_m,
        const float *xb_t,
        const int xdelta,
        const float xe_l0,
        const float xf_2x,
        const int xM_AT_0,
        const int xm_at_1750,
        const int xM_LO_0,
        const int xM_UP_0,
        const int xN,
        const float *xphi_m,
        const float *xphi_t,
        const float xT_2x,
        const float xdelta_el,
        const float xe_0,
        const float xeta,
        const float xf_0,
        const float xf_1,
        const float xmu_0,
        const float xq_0,
        const int xt_0,
        const int xt_f,
        const float *xA_0,
        const float *xK_0,
        const int *xL_0,
        const float *xL_a,
        const float *xT_AT_0,
        const float *xT_LO_0,
        const float *xa_1,
        const float *xa_2,
        const int *xa_3,
        const float *xalpha,
        const float *xdelta_a,
        const float *xdelta_K,
        const float *xdelta_pb,
        const float *xdelta_sigma,
        const float *xg_a,
        const float *xg_sigma,
        const float *xgamma,
        const float *xl_g,
        const int *xp_b,
        const float *xrho,
        const float *xscale_1,
        const float *xscale_2,
        const float *xsigma_0,
        const float *xtheta_2,
        float *abatement_cost_all_regions,
        int *activity_timestep,
        float *capital_all_regions,
        float *capital_depreciation_all_regions,
        float *consumption_all_regions,
        float *current_balance_all_regions,
        float *damages_all_regions,
        float *desired_imports,
        float *future_tariffs,
        float *global_carbon_mass,
        float *global_exogenous_emissions,
        float *global_land_emissions,
        float *global_temperature,
        float *gross_output_all_regions,
        float *intensity_all_regions,
        float *investment_all_regions,
        float *labor_all_regions,
        float *max_export_limit_all_regions,
        float *minimum_mitigation_rate_all_regions,
        float *mitigation_cost_all_regions,
        float *mitigation_rate_all_regions,
        float *production_all_regions,
        float *production_factor_all_regions,
        float *promised_mitigation_rate,
        float *proposal_decisions,
        float *requested_mitigation_rate,
        float *reward_all_regions,
        float *savings_all_regions,
        float *scaled_imports,
        float *social_welfare_all_regions,
        int *stage,
        float *tariffed_imports,
        float *tariffs,
        int *timestep,
        float *utility_all_regions,
        float abatement_cost_all_regions_norm,
        float activity_timestep_norm,
        float capital_all_regions_norm,
        float capital_depreciation_all_regions_norm,
        float consumption_all_regions_norm,
        float current_balance_all_regions_norm,
        float damages_all_regions_norm,
        float desired_imports_norm,
        float future_tariffs_norm,
        float global_carbon_mass_norm,
        float global_exogenous_emissions_norm,
        float global_land_emissions_norm,
        float global_temperature_norm,
        float gross_output_all_regions_norm,
        float intensity_all_regions_norm,
        float investment_all_regions_norm,
        float labor_all_regions_norm,
        float max_export_limit_all_regions_norm,
        float minimum_mitigation_rate_all_regions_norm,
        float mitigation_cost_all_regions_norm,
        float mitigation_rate_all_regions_norm,
        float production_all_regions_norm,
        float production_factor_all_regions_norm,
        float promised_mitigation_rate_norm,
        float proposal_decisions_norm,
        float requested_mitigation_rate_norm,
        float reward_all_regions_norm,
        float savings_all_regions_norm,
        float scaled_imports_norm,
        float social_welfare_all_regions_norm,
        float stage_norm,
        float tariffed_imports_norm,
        float tariffs_norm,
        float timestep_norm,
        float utility_all_regions_norm,
        const int kNumDiscreteActionLevels,
        const float kBalanceInterestRate,
        const bool kNegotiationOn,
        float *aux_ms,
        const float *sub_rate,
        const float *dom_pref,
        const float *for_pref,
        int *current_year,
        int kEndYear,
        float *observations_arr,
        int *action_masks_arr,
        int *actions_arr,
        float *rewards_arr,
        int *done_arr,
        int *env_timestep_arr,
        const int kNumAgents,
        const int kEpisodeLength)
    {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;
        const int kAgentArrayIdx = kEnvId * kNumAgents + kAgentId;

        const int kNumNegotiationStages = 3;

        // Increment time ONCE -- only 1 thread can do this.
        if (kAgentId == 0)
        {
            env_timestep_arr[kEnvId] += 1;
            timestep[kEnvId] = env_timestep_arr[kEnvId];

            // Determine the stage
            if (kNegotiationOn)
            {
                stage[kEnvId] =
                    env_timestep_arr[kEnvId] % kNumNegotiationStages;
            }
        }

        // Wait here until timestep has been updated
        __syncthreads();

        assert(env_timestep_arr[kEnvId] > 0 &&
               env_timestep_arr[kEnvId] <= kEpisodeLength);

        // Defining some constants for the RICE simulation
        const int global_temperature_len = 2;
        const int global_carbon_mass_len = 3;
        const int global_exogenous_emissions_len = 1;
        const int global_land_emissions_len = 1;

        // obs features
        // ============

        // # Global features that are observable by all regions
        // global_features = [
        //     "global_temperature",
        //     "global_carbon_mass",
        //     "global_exogenous_emissions",
        //     "global_land_emissions",
        //     "timestep",
        // ]

        // # Public features that are observable by all regions
        // public_features = [
        //     "capital_all_regions",
        //     "capital_depreciation_all_regions",
        //     "labor_all_regions",
        //     "gross_output_all_regions",
        //     "investment_all_regions",
        //     "consumption_all_regions",
        //     "savings_all_regions",
        //     "mitigation_rate_all_regions",
        //     "tariffs",
        //     "max_export_limit_all_regions",
        //     "current_balance_all_regions",
        // ]

        // # Private features that are private to each region.
        // private_features = [
        //     "production_factor_all_regions",
        //     "intensity_all_regions",
        //     "mitigation_cost_all_regions",
        //     "damages_all_regions",
        //     "abatement_cost_all_regions",
        //     "production_all_regions",
        //     "utility_all_regions",
        //     "social_welfare_all_regions",
        //     "reward_all_regions",
        // ]
        int kNumFeatures = kNumAgents + // region_indicator
                           global_temperature_len +
                           global_carbon_mass_len +
                           global_exogenous_emissions_len +
                           global_land_emissions_len +
                           1 +                       // timestep
                           10 * kNumAgents +         // public features
                           kNumAgents * kNumAgents + // tariffs
                           9;                        // private features

        const int kNumSavingsActions = 1;
        const int kNumMitigationRateActions = 1;
        const int kNumExportActions = 1;
        const int kNumImportActions = kNumAgents;
        const int kNumTariffActions = kNumAgents;

        int kNumActions = kNumSavingsActions +
                          kNumMitigationRateActions +
                          kNumExportActions +
                          kNumImportActions +
                          kNumTariffActions;

        int kActionLen =
            kNumSavingsActions * kNumDiscreteActionLevels +
            kNumMitigationRateActions * kNumDiscreteActionLevels +
            kNumExportActions * kNumDiscreteActionLevels +
            kNumImportActions * kNumDiscreteActionLevels +
            kNumTariffActions * kNumDiscreteActionLevels;

        const int kNumProposalActions = 2 * kNumAgents;
        const int kNumEvaluationActions = kNumAgents;

        // Resetting agent rewards to 0
        reward_all_regions[kAgentArrayIdx] = 0;

        if (kNegotiationOn)
        {
            // global_features += ["stage"]

            // public_features += []

            // private_features += [
            //     "minimum_mitigation_rate",
            // ]

            // bilateral_features += [
            //     "promised_mitigation_rate",
            //     "requested_mitigation_rate",
            //     "proposal_decisions",
            // ]
            kNumFeatures += 1 +
                            3 * 2 * kNumAgents +
                            1;

            kNumActions += kNumProposalActions + kNumEvaluationActions;

            kActionLen += 2 * kNumAgents * kNumDiscreteActionLevels +
                          kNumAgents * 2;
        }

        if (kNegotiationOn)
        {
            if (stage[kEnvId] == 1)
            {
                CudaProposalStep(
                    abatement_cost_all_regions,
                    activity_timestep,
                    capital_all_regions,
                    capital_depreciation_all_regions,
                    consumption_all_regions,
                    current_balance_all_regions,
                    damages_all_regions,
                    global_carbon_mass,
                    global_exogenous_emissions,
                    global_land_emissions,
                    global_temperature,
                    gross_output_all_regions,
                    intensity_all_regions,
                    investment_all_regions,
                    labor_all_regions,
                    max_export_limit_all_regions,
                    minimum_mitigation_rate_all_regions,
                    mitigation_cost_all_regions,
                    mitigation_rate_all_regions,
                    production_all_regions,
                    production_factor_all_regions,
                    promised_mitigation_rate,
                    requested_mitigation_rate,
                    proposal_decisions,
                    reward_all_regions,
                    savings_all_regions,
                    social_welfare_all_regions,
                    stage,
                    timestep,
                    tariffs,
                    utility_all_regions,
                    abatement_cost_all_regions_norm,
                    activity_timestep_norm,
                    capital_all_regions_norm,
                    capital_depreciation_all_regions_norm,
                    consumption_all_regions_norm,
                    current_balance_all_regions_norm,
                    damages_all_regions_norm,
                    global_carbon_mass_norm,
                    global_exogenous_emissions_norm,
                    global_land_emissions_norm,
                    global_temperature_norm,
                    gross_output_all_regions_norm,
                    intensity_all_regions_norm,
                    investment_all_regions_norm,
                    labor_all_regions_norm,
                    max_export_limit_all_regions_norm,
                    minimum_mitigation_rate_all_regions_norm,
                    mitigation_cost_all_regions_norm,
                    mitigation_rate_all_regions_norm,
                    production_all_regions_norm,
                    production_factor_all_regions_norm,
                    promised_mitigation_rate_norm,
                    proposal_decisions_norm,
                    requested_mitigation_rate_norm,
                    reward_all_regions_norm,
                    savings_all_regions_norm,
                    social_welfare_all_regions_norm,
                    stage_norm,
                    tariffs_norm,
                    timestep_norm,
                    utility_all_regions_norm,
                    observations_arr,
                    action_masks_arr,
                    actions_arr,
                    rewards_arr,
                    done_arr,
                    env_timestep_arr,
                    kNumDiscreteActionLevels,
                    kNegotiationOn,
                    global_temperature_len,
                    global_carbon_mass_len,
                    global_exogenous_emissions_len,
                    global_land_emissions_len,
                    kNumMitigationRateActions,
                    kNumSavingsActions,
                    kNumExportActions,
                    kNumImportActions,
                    kNumTariffActions,
                    kEnvId,
                    kAgentId,
                    kNumAgents,
                    kAgentArrayIdx,
                    kNumFeatures,
                    kActionLen,
                    kNumActions,
                    kEpisodeLength);
            }
            else if (stage[kEnvId] == 2)
            {
                CudaEvaluationStep(
                    abatement_cost_all_regions,
                    activity_timestep,
                    capital_all_regions,
                    capital_depreciation_all_regions,
                    consumption_all_regions,
                    current_balance_all_regions,
                    damages_all_regions,
                    global_carbon_mass,
                    global_exogenous_emissions,
                    global_land_emissions,
                    global_temperature,
                    gross_output_all_regions,
                    intensity_all_regions,
                    investment_all_regions,
                    labor_all_regions,
                    max_export_limit_all_regions,
                    minimum_mitigation_rate_all_regions,
                    mitigation_cost_all_regions,
                    mitigation_rate_all_regions,
                    production_all_regions,
                    production_factor_all_regions,
                    promised_mitigation_rate,
                    requested_mitigation_rate,
                    proposal_decisions,
                    reward_all_regions,
                    savings_all_regions,
                    social_welfare_all_regions,
                    stage,
                    timestep,
                    tariffs,
                    utility_all_regions,
                    abatement_cost_all_regions_norm,
                    activity_timestep_norm,
                    capital_all_regions_norm,
                    capital_depreciation_all_regions_norm,
                    consumption_all_regions_norm,
                    current_balance_all_regions_norm,
                    damages_all_regions_norm,
                    global_carbon_mass_norm,
                    global_exogenous_emissions_norm,
                    global_land_emissions_norm,
                    global_temperature_norm,
                    gross_output_all_regions_norm,
                    intensity_all_regions_norm,
                    investment_all_regions_norm,
                    labor_all_regions_norm,
                    max_export_limit_all_regions_norm,
                    minimum_mitigation_rate_all_regions_norm,
                    mitigation_cost_all_regions_norm,
                    mitigation_rate_all_regions_norm,
                    production_all_regions_norm,
                    production_factor_all_regions_norm,
                    promised_mitigation_rate_norm,
                    proposal_decisions_norm,
                    requested_mitigation_rate_norm,
                    reward_all_regions_norm,
                    savings_all_regions_norm,
                    social_welfare_all_regions_norm,
                    stage_norm,
                    tariffs_norm,
                    timestep_norm,
                    utility_all_regions_norm,
                    observations_arr,
                    action_masks_arr,
                    actions_arr,
                    rewards_arr,
                    done_arr,
                    env_timestep_arr,
                    kNumDiscreteActionLevels,
                    kNegotiationOn,
                    global_temperature_len,
                    global_carbon_mass_len,
                    global_exogenous_emissions_len,
                    global_land_emissions_len,
                    kNumMitigationRateActions,
                    kNumSavingsActions,
                    kNumExportActions,
                    kNumImportActions,
                    kNumTariffActions,
                    kNumProposalActions,
                    kNumEvaluationActions,
                    kEnvId,
                    kAgentId,
                    kNumAgents,
                    kAgentArrayIdx,
                    kNumFeatures,
                    kActionLen,
                    kNumActions,
                    kEpisodeLength);
            }
        }
        if (stage[kEnvId] == 0)
        {
            CudaActivityStep(
                xt_0,
                xdelta,
                xN,
                xphi_t,
                xb_t,
                xphi_m,
                xb_m,
                xeta,
                xm_at_1750,
                xf_0,
                xf_1,
                xt_f,
                xe_l0,
                xdelta_el,
                xM_AT_0,
                xM_UP_0,
                xM_LO_0,
                xe_0,
                xq_0,
                xmu_0,
                xf_2x,
                xT_2x,
                xgamma,
                xtheta_2,
                xa_1,
                xa_2,
                xa_3,
                xdelta_K,
                xalpha,
                xrho,
                xL_0,
                xL_a,
                xl_g,
                xA_0,
                xg_a,
                xdelta_a,
                xsigma_0,
                xg_sigma,
                xdelta_sigma,
                xp_b,
                xdelta_pb,
                xscale_1,
                xscale_2,
                xT_AT_0,
                xT_LO_0,
                xK_0,
                abatement_cost_all_regions,
                activity_timestep,
                capital_all_regions,
                capital_depreciation_all_regions,
                consumption_all_regions,
                current_balance_all_regions,
                damages_all_regions,
                global_carbon_mass,
                global_exogenous_emissions,
                global_land_emissions,
                global_temperature,
                gross_output_all_regions,
                intensity_all_regions,
                investment_all_regions,
                labor_all_regions,
                max_export_limit_all_regions,
                minimum_mitigation_rate_all_regions,
                mitigation_cost_all_regions,
                mitigation_rate_all_regions,
                production_all_regions,
                production_factor_all_regions,
                promised_mitigation_rate,
                requested_mitigation_rate,
                proposal_decisions,
                reward_all_regions,
                savings_all_regions,
                social_welfare_all_regions,
                stage,
                timestep,
                tariffs,
                desired_imports,
                scaled_imports,
                tariffed_imports,
                future_tariffs,
                utility_all_regions,
                abatement_cost_all_regions_norm,
                activity_timestep_norm,
                capital_all_regions_norm,
                capital_depreciation_all_regions_norm,
                consumption_all_regions_norm,
                current_balance_all_regions_norm,
                damages_all_regions_norm,
                global_carbon_mass_norm,
                global_exogenous_emissions_norm,
                global_land_emissions_norm,
                global_temperature_norm,
                gross_output_all_regions_norm,
                intensity_all_regions_norm,
                investment_all_regions_norm,
                labor_all_regions_norm,
                max_export_limit_all_regions_norm,
                minimum_mitigation_rate_all_regions_norm,
                mitigation_cost_all_regions_norm,
                mitigation_rate_all_regions_norm,
                production_all_regions_norm,
                production_factor_all_regions_norm,
                promised_mitigation_rate_norm,
                proposal_decisions_norm,
                requested_mitigation_rate_norm,
                reward_all_regions_norm,
                savings_all_regions_norm,
                social_welfare_all_regions_norm,
                stage_norm,
                tariffs_norm,
                timestep_norm,
                utility_all_regions_norm,
                aux_ms,
                sub_rate,
                dom_pref,
                for_pref,
                observations_arr,
                action_masks_arr,
                actions_arr,
                rewards_arr,
                done_arr,
                env_timestep_arr,
                current_year,
                kEndYear,
                kNumDiscreteActionLevels,
                kBalanceInterestRate,
                kNegotiationOn,
                global_temperature_len,
                global_carbon_mass_len,
                global_exogenous_emissions_len,
                global_land_emissions_len,
                kNumSavingsActions,
                kNumMitigationRateActions,
                kNumExportActions,
                kNumImportActions,
                kNumTariffActions,
                kNumProposalActions,
                kNumEvaluationActions,
                kEnvId,
                kAgentId,
                kNumAgents,
                kAgentArrayIdx,
                kNumFeatures,
                kActionLen,
                kNumActions,
                kEpisodeLength);
        }
    }
}
