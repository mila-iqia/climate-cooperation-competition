import os
import re
import subprocess
from pathlib import Path

_path = Path(os.path.abspath(__file__))
PUBLIC_REPO_DIR = str(_path.parent.parent.absolute())
os.chdir(PUBLIC_REPO_DIR)


def test_training_run():
    env = os.environ.copy()
    env["CONFIG_FILE"] = "./tests/rice_rllib_test.yaml"
    command = subprocess.run(["python", "./scripts/train_with_rllib.py"],
                             env=env,
                             capture_output=True)
    assert command.returncode == 0
    output = command.stdout.decode("utf-8")

    file_reg = re.compile(r'is created at: (.*$)', flags=re.M)
    print(output)
    match = file_reg.search(output).group(1)
    assert os.path.exists(match)
