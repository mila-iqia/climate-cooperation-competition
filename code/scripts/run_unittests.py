# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Unit tests for the rice simulation
"""
import importlib.util as iu
import logging
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np
from evaluate_submission import get_results_dir

_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(_ROOT_DIR)
_REGION_YAMLS = "region_yamls"

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_BASE_FILE_PATH = "<ADD_PATH_TO_THE_OPEN_SOURCE_CODE_HERE>"


def import_class_from_path(class_name=None, path=None):
    """
    Helper function to import class from a path.
    """
    assert class_name is not None
    assert path is not None
    spec = iu.spec_from_file_location(class_name, path)
    module_from_spec = iu.module_from_spec(spec)
    spec.loader.exec_module(module_from_spec)
    return getattr(module_from_spec, class_name)


def fetch_base_env(base_folder="/tmp/_base"):
    """
    Download the base version of the code from GitHub.
    """
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    os.makedirs(base_folder, exist_ok=False)

    print(
        "\nDownloading a base version of the code from GitHub"
        " to run consistency checks..."
    )
    prev_dir = os.getcwd()
    os.chdir(base_folder)
    subprocess.call(["wget", "-O", "rice.py", _BASE_FILE_PATH])
    shutil.copytree(
        os.path.join(_ROOT_DIR, "region_yamls"),
        os.path.join(base_folder, "region_yamls"),
    )

    base_rice = import_class_from_path("Rice", os.path.join(base_folder, "rice.py"))()

    # Clean up base code
    os.chdir(prev_dir)
    shutil.rmtree(base_folder)
    return base_rice


class TestEnv(unittest.TestCase):
    """
    The env testing class.
    """

    @classmethod
    def setUpClass(cls):
        """Set-up"""

        cls.framework = "rllib"

        cls.base_env = fetch_base_env()  # Fetch the base version from GitHub
        try:
            cls.env = import_class_from_path(
                "Rice", os.path.join(cls.results_dir, "rice.py")
            )()
        except Exception as err:
            raise ValueError(
                "The Rice environment could not be instantiated !"
            ) from err

        cls.results_dir = None

        base_env_action_nvec = np.array(cls.base_env.action_space[0].nvec)
        cls.base_env_random_actions = {
            agent_id: np.random.randint(
                low=0 * base_env_action_nvec, high=base_env_action_nvec - 1
            )
            for agent_id in range(cls.base_env.num_agents)
        }
        sample_agent_id = 0
        env_action_nvec = np.array(cls.env.action_space[sample_agent_id].nvec)
        len_negotiation_actions = len(env_action_nvec) - len(base_env_action_nvec)
        cls.env_random_actions = {
            agent_id: np.append(
                cls.base_env_random_actions[agent_id],
                np.zeros(len_negotiation_actions, dtype=np.int32),
            )
            for agent_id in range(cls.env.num_agents)
        }

    def test_env_attributes(self):
        """
        Test the env attributes are consistent with the base version.
        """
        for attribute in [
            "all_constants",
            "num_regions",
            "num_agents",
            "start_year",
            "end_year",
            "num_discrete_action_levels",
        ]:
            np.testing.assert_array_equal(
                getattr(self.base_env, attribute), getattr(self.env, attribute)
            )

        features = [
            "activity_timestep",
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "consumption_all_regions",
            "savings_all_regions",
            "mitigation_rate_all_regions",
            "tariffs",
            "max_export_limit_all_regions",
            "current_balance_all_regions",
            "production_factor_all_regions",
            "intensity_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
            "scaled_imports",
        ]

        # Test equivalence after reset
        self.base_env.reset()
        self.env.reset()

        for feature in features:
            np.testing.assert_array_equal(
                getattr(self.base_env, "global_state")[feature]["value"][0],
                getattr(self.env, "global_state")[feature]["value"][0],
            )

        # Test equivalence after stepping through the env
        for timestep in range(self.base_env.episode_length):
            self.base_env.timestep += 1
            self.base_env.climate_and_economy_simulation_step(
                self.base_env_random_actions
            )

            self.env.timestep += 1
            self.env.climate_and_economy_simulation_step(self.env_random_actions)

            for feature in features:
                np.testing.assert_array_equal(
                    getattr(self.base_env, "global_state")[feature]["value"][timestep],
                    getattr(self.env, "global_state")[feature]["value"][timestep],
                )

    def test_env_reset(self):
        """
        Test the env reset output
        """
        obs_at_reset = self.env.reset()
        self.assertEqual(len(obs_at_reset), self.env.num_agents)

    def test_env_step(self):
        """
        Test the env step output
        """
        assert isinstance(
            self.env.action_space, dict
        ), "Action space must be a dictionary keyed by agent ids."
        assert sorted(list(self.env.action_space.keys())) == list(
            range(self.env.num_agents)
        )

        # Test with all random actions
        obs, rew, done, _ = self.env.step(self.env_random_actions)
        self.assertEqual(list(obs.keys()), list(rew.keys()))
        assert list(done.keys()) == ["__all__"]

    def test_cpu_gpu_consistency_checks(self):
        """
        Run the CPU/GPU environment consistency checks
        (only if using the CUDA version of the env)
        """
        if self.framework == "warpdrive":
            # Execute the CPU-GPU consistency checks
            subprocess.call(["python", "run_cpu_gpu_env_consistency_checks.py"])


if __name__ == "__main__":
    logging.info("Running env unit tests...")

    results_dir, parser = get_results_dir()
    parser.add_argument("unittest_args", nargs="*")
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args

    TestEnv.results_dir = results_dir
    if _REGION_YAMLS not in os.listdir(results_dir):
        shutil.copytree(
            os.path.join(_ROOT_DIR, "region_yamls"),
            os.path.join(results_dir, "region_yamls"),
        )

    if ".warpdrive" in os.listdir(results_dir):
        TestEnv.framework = "warpdrive"
    else:
        assert ".rllib" in os.listdir(results_dir), (
            f"Missing identifier file! "
            f"Either the .rllib or the .warpdrive "
            f"file must be present in the results directory: {results_dir}"
        )

    unittest.main()
