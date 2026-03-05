import os
import sys

# Ensure the inner `rice_jax` package directory is on sys.path when running
# this script directly (repo layout: repo/rice_jax/rice_jax/...).
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inner_pkg_path = os.path.join(repo_root, "rice_jax")
if inner_pkg_path not in sys.path:
    sys.path.insert(0, inner_pkg_path)

from rice_jax._rice import Rice
from rice_jax.utils._helpers import load_region_yamls
import jax


def run_probe(steps=6, region_idx=18):
    params = load_region_yamls(20)
    env = Rice(region_params=params, num_regions=20)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    print("Initial production_factor_all_regions[{}]:".format(region_idx), state["production_factor_all_regions"][region_idx])
    print("Initial capital_all_regions[{}]:".format(region_idx), state["capital_all_regions"][region_idx])

    for t in range(steps):
        # sample deterministic-zero actions (use sample_action then zero)
        akey = jax.random.PRNGKey(t + 1)
        actions = env.sample_action(akey)
        # zero actions (use zeros_like to be safe)
        actions = jax.tree_util.tree_map(lambda x: x * 0.0, actions)

        (_, _, _, _, _), state = env.step_env(akey, state, actions)

        print(f"Step {t+1} production_factor_all_regions[{region_idx}]:", state["production_factor_all_regions"][region_idx])
        print(f"Step {t+1} capital_all_regions[{region_idx}]:", state["capital_all_regions"][region_idx])


if __name__ == '__main__':
    run_probe()
