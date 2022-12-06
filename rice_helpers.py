# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Helper functions for the rice simulation
"""
import os

import numpy as np
import yaml

_SMALL_NUM = 1e-0  # small number added to handle consumption blow-up


# Load calibration data from yaml files
def read_yaml_data(yaml_file):
    """Helper function to read yaml configuration data."""
    with open(yaml_file, "r", encoding="utf-8") as file_ptr:
        file_data = file_ptr.read()
    file_ptr.close()
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data


def set_rice_params(yamls_folder=None):
    """Helper function to read yaml data and set environment configs."""
    assert yamls_folder is not None
    dice_params = read_yaml_data(os.path.join(yamls_folder, "default.yml"))
    file_list = sorted(os.listdir(yamls_folder))  #
    yaml_files = []
    for file in file_list:
        if file[-4:] == ".yml" and file != "default.yml":
            yaml_files.append(file)

    rice_params = []
    for file in yaml_files:
        rice_params.append(read_yaml_data(os.path.join(yamls_folder, file)))

    # Overwrite rice params
    num_regions = len(rice_params)
    for k in dice_params["_RICE_CONSTANT"].keys():
        dice_params["_RICE_CONSTANT"][k] = [
            dice_params["_RICE_CONSTANT"][k]
        ] * num_regions
    for idx, param in enumerate(rice_params):
        for k in param["_RICE_CONSTANT"].keys():
            dice_params["_RICE_CONSTANT"][k][idx] = param["_RICE_CONSTANT"][k]

    return dice_params, num_regions


# RICE dynamics
def get_mitigation_cost(p_b, theta_2, delta_pb, timestep, intensity):
    """Obtain the cost for mitigation."""
    return p_b / (1000 * theta_2) * pow(1 - delta_pb, timestep - 1) * intensity


def get_exogenous_emissions(f_0, f_1, t_f, timestep):
    """Obtain the amount of exogeneous emissions."""
    return f_0 + min(f_1 - f_0, (f_1 - f_0) / t_f * (timestep - 1))


def get_land_emissions(e_l0, delta_el, timestep, num_regions):
    """Obtain the amount of land emissions."""
    return e_l0 * pow(1 - delta_el, timestep - 1)/num_regions


def get_production(production_factor, capital, labor, gamma):
    """Obtain the amount of goods produced."""
    return production_factor * pow(capital, gamma) * pow(labor / 1000, 1 - gamma)


def get_damages(t_at, a_1, a_2, a_3):
    """Obtain damages."""
    return 1 / (1 + a_1 * t_at + a_2 * pow(t_at, a_3))


def get_abatement_cost(mitigation_rate, mitigation_cost, theta_2):
    """Compute the abatement cost."""
    return mitigation_cost * pow(mitigation_rate, theta_2)


def get_gross_output(damages, abatement_cost, production):
    """Compute the gross production output, taking into account
    damages and abatement cost."""
    return damages * (1 - abatement_cost) * production


def get_investment(savings, gross_output):
    """Obtain the investment cost."""
    return savings * gross_output


def get_consumption(gross_output, investment, exports):
    """Obtain the consumption cost."""
    total_exports = np.sum(exports)
    assert gross_output - investment - total_exports > -1e-5, "consumption cannot be negative!"
    return max(0.0, gross_output - investment - total_exports)


def get_max_potential_exports(x_max, gross_output, investment):
    """Determine the maximum potential exports."""
    if x_max * gross_output <= gross_output - investment:
        return x_max * gross_output
    return gross_output - investment


def get_capital_depreciation(x_delta_k, x_delta):
    """Compute the global capital depreciation."""
    return pow(1 - x_delta_k, x_delta)


def get_global_temperature(
    phi_t, temperature, b_t, f_2x, m_at, m_at_1750, exogenous_emissions
):
    """Get the temperature levels."""
    return np.dot(phi_t, temperature) + np.dot(
        b_t, f_2x * np.log(m_at / m_at_1750) / np.log(2) + exogenous_emissions
    )


def get_aux_m(intensity, mitigation_rate, production, land_emissions):
    """Auxiliary variable to denote carbon mass levels."""
    return intensity * (1 - mitigation_rate) * production + land_emissions


def get_global_carbon_mass(phi_m, carbon_mass, b_m, aux_m):
    """Get the carbon mass level."""
    return np.dot(phi_m, carbon_mass) + np.dot(b_m, aux_m)


def get_capital(capital_depreciation, capital, delta, investment):
    """Evaluate capital."""
    return capital_depreciation * capital + delta * investment


def get_labor(labor, l_a, l_g):
    """Compute total labor."""
    return labor * pow((1 + l_a) / (1 + labor), l_g)


def get_production_factor(production_factor, g_a, delta_a, delta, timestep):
    """Compute the production factor."""
    return production_factor * (
        np.exp(0.0033) + g_a * np.exp(-delta_a * delta * (timestep - 1))
    )


def get_carbon_intensity(intensity, g_sigma, delta_sigma, delta, timestep):
    """Determine the carbon emission intensity."""
    return intensity * np.exp(
        -g_sigma * pow(1 - delta_sigma, delta * (timestep - 1)) * delta
    )


def get_utility(labor, consumption, alpha):
    """Obtain the utility."""
    return (
        (labor / 1000.0)
        * (pow(consumption / (labor / 1000.0) + _SMALL_NUM, 1 - alpha) - 1)
        / (1 - alpha)
    )


def get_social_welfare(utility, rho, delta, timestep):
    """Compute social welfare"""
    return utility / pow(1 + rho, delta * timestep)


def get_armington_agg(
    c_dom,
    c_for,  # np.array
    sub_rate=0.5,  # in (0,1)
    dom_pref=0.5,  # in [0,1]
    for_pref=None,  # np.array
):
    """
    Armington aggregate from Lessmann,2009.
    Consumption goods from different regions act as imperfect substitutes.
    As such, consumption of domestic and foreign goods are scaled according to
    relative preferences, as well as a substitution rate, which are modeled
    by a CES functional form.
    Inputs :
        `C_dom`     : A scalar representing domestic consumption. The value of
                    C_dom is what is left over from initial production after
                    investment and exports are deducted.
        `C_for`     : An array reprensenting foreign consumption. Each element
                    is the consumption imported from a given country.
        `sub_rate`  : A substitution parameter in (0,1). The elasticity of
                    substitution is 1 / (1 - sub_rate).
        `dom_pref`  : A scalar in [0,1] representing the relative preference for
                    domestic consumption over foreign consumption.
        `for_pref`  : An array of the same size as `C_for`. Each element is the
                    relative preference for foreign goods from that country.
    """

    c_dom_pref = dom_pref * (c_dom ** sub_rate)
    c_for_pref = np.sum(for_pref * pow(c_for, sub_rate))

    c_agg = (c_dom_pref + c_for_pref) ** (1 / sub_rate)  # CES function
    return c_agg
