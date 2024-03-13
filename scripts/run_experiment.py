import os
from fixed_paths import PUBLIC_REPO_DIR
import subprocess
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--exp",
    "-e",
    type=str,
    default="none",
    help="experiment_type",
)

parser.add_argument(
    "--iterations",
    "-i",
    type=int,
    default=1,
    help="experiment_type",
)

args = parser.parse_args()
exp = args.exp
iterations = args.iterations

EXP = { "tariff":"1709895189_default.zip", #NoNegotiator Tariff
        "none":"1709895189_default.zip"} #Default for testing, basicclubdescretedefect

if __name__ == "__main__":

    for i in tqdm(range(iterations)):
        subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "conduct_experiment.py"),
            "-r",
            f"Submissions/{EXP[exp]}",
            "-e",
            exp
        ]
        )