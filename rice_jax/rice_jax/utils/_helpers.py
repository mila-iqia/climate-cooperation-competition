import importlib
import os
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import optimistix as optx
import yaml


def i_to_agent_str(i, prefix="region-", suffix="") -> str:
    """Converts an index to a string name for each region.

    @usage
    >>> i_to_agent_str(0)
    'region-00'
    >>> i_to_agent_str(10)
    'region-10'
    """
    return f"{prefix}{i:02d}{suffix}"


def oneoveralpha_objective_function(
    oneoveralpha, a, tau, irf0, irC, irT, pert_carb_stock, temperature
):
    """Objective function for finding the right alpha value."""
    b = a * tau * (1 - jnp.exp(-100 * oneoveralpha / tau))
    return jnp.sum(b) - oneoveralpha * (
        irf0 + irC * pert_carb_stock + irT * temperature
    )


def solve_for_alpha(prev_alpha, a, tau, irf0, irC, irT, pert_carb_stock, temperature):
    """Use optimistix to find alpha value."""
    initial_guess = 1.0 / prev_alpha

    # Define problem for optimistix
    def fn(x, args):
        return oneoveralpha_objective_function(
            x, a, tau, irf0, irC, irT, pert_carb_stock, temperature
        )

    # Use Newton's method
    # solver = optx.Newton(rtol=1e-5, atol=1e-5)
    solver = optx.Bisection(rtol=1e-4, atol=1e-4)
    result = optx.root_find(
        fn, solver, initial_guess, options=dict(lower=0.01, upper=100)
    )

    # Extract and clip alpha to valid range
    alpha = 1.0 / result.value
    return alpha


def load_region_yamls(num_regions: int):
    assert num_regions in [3, 7, 20], "Supported number of regions are 3, 7, 20"
    yaml_file_directory = importlib.resources.files("rice_jax").joinpath(
        "./region_yamls/"
    )
    region_yamls = []
    for file in sorted(os.listdir(f"{yaml_file_directory}/{num_regions}_regions")):
        if file.endswith(".yml"):
            with open(f"{yaml_file_directory}/{num_regions}_regions/{file}", "r") as f:
                region = yaml.safe_load(f)
                region = region["_RICE_CONSTANT"]  # remove redundant key
                region_yamls.append(region)

    # ximport_ is an exception, sice it is an array for each region
    ximport_ = [region["ximport"] for region in region_yamls]
    ximport_ = [dict(sorted(x.items())) for x in ximport_]  # sort by region id
    ximport_ = [list(x.values()) for x in ximport_]

    region_params = {
        k: np.array([region[k] for region in region_yamls])
        for k in region_yamls[0].keys()
    }
    region_params["ximport"] = np.array(ximport_)

    # default params, that apply generally:
    with open(f"{yaml_file_directory}/default.yml", "r") as f:
        default_params = yaml.safe_load(f)
        dice_params = default_params["_DICE_CONSTANT"]
        rice_params = default_params["_RICE_CONSTANT"]

        # in dice_params, convert all lists (or lists of lists) to tuples
        def list_to_tuples(value):
            if isinstance(value, list):
                return tuple(list_to_tuples(v) for v in value)
            return value

        dice_params = {k: list_to_tuples(v) for k, v in dice_params.items()}

        default_params = {**dice_params, **rice_params}

    # merge the region_params, overwriting the key,values that are already present
    params = {**default_params, **region_params}

    # allow for dot notation
    params = SimpleNamespace(**params)

    return params
