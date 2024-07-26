import os 
import subprocess
from tqdm import tqdm
from fixed_paths import PUBLIC_REPO_DIR

#NOTE: conduct_experiment.py slows after approx 25 executions
# repeatedly calling the script from here speeds up rollout gathering
for i in tqdm(range(100)):
    subprocess.call(
       [ "python",
        os.path.join(
                PUBLIC_REPO_DIR, "scripts", "conduct_experiment.py"
            )]
    )