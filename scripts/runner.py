import subprocess
import os
from fixed_paths import PUBLIC_REPO_DIR
for i in range(1):
    subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "train_with_rllib.py"),
            "--yaml",
            "rice_rllib_discrete.yaml",
        ]
    )