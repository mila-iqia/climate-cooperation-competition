# Competition Submission FAQ

## Useful Links:

[Competition registration form](https://docs.google.com/forms/d/e/1FAIpQLSe2SWnhJaRpjcCa3idq7zIFubRoH0pATLOP7c1Y0kMXOV6U4w/viewform)

[Submission form](https://docs.google.com/forms/d/e/1FAIpQLSdATpPMnhjXNFAnGNRU2kuufwD5HFilGxgIXFK9QKsqrDbkog/viewform)

[Leaderboard](http://34.111.184.174/)

[Competition website](https://mila-iqia.github.io/climate-cooperation-competition/)

[Contact email](TODO: ADD EMAIL)

## How can I register for the competition?
Please fill out [this](https://docs.google.com/forms/d/e/1FAIpQLSe2SWnhJaRpjcCa3idq7zIFubRoH0pATLOP7c1Y0kMXOV6U4w/viewform) Google form in order to register for the competition. You will only need to provide an email address and a team name. You will also need to be willing to open-source your code after the competition.

After you submit your registration form, we will register it internally. You will need your team name in order to make submissions towards the competition.

## Where can I submit my solution?
NOTE: Please register for the competition (see the steps above), if you have not done so. Your team must be registered before you can submit your solutions.

In order to submit your solution, you need to fill out the Google form [here](https://docs.google.com/forms/d/e/1FAIpQLSdATpPMnhjXNFAnGNRU2kuufwD5HFilGxgIXFK9QKsqrDbkog/viewform).

Please select your registered team name from the drop-down menu, and upload a zip file containing the submission files - we will be providing scripts to help you create the zip file. You will also need to be willing to make your submission open-source (after the competition).

If you do not see your team name in the drop-down menu, please contact us (TODO: add contact email), and we will resolve that for you.

## What are the files provided?
We provide the base version of the RICE (regional integrated climate environment) simulation environment written in Python (`rice.py`).

The RICE simulation may be trained only using one of these two frameworks:
1. [RLlib](https://docs.ray.io/en/latest/rllib/index.html#:~:text=RLlib%20is%20an%20open%2Dsource,large%20variety%20of%20industry%20applications): The pythonic environment can be trained on your local CPU machine using open-source RL framework, RLlib.
2. [WarpDrive](https://github.com/salesforce/warp-drive): WarpDrive is a GPU-based framework that allows for [over 10x faster training](https://arxiv.org/pdf/2108.13976.pdf) compared to CPU-based training. It requires the simulation to be written out in CUDA C, and we also provide a starter version of the simulation environment written in CUDA C (`rice_step.cu`)

We also provide starter scripts to train the simulation you build with either of the above frameworks.
Note that we only allow these two options, since our backend submission evaluation process only supports these at the moment.

### File Structure
Below is the detailed file tree, and file descriptions.
```commandline
ROOT_DIR
├── rice.py
├── rice_helpers.py
├── region_yamls

├── rice_step.cu
├── rice_cuda.py
├── rice_build.cu

└── scripts
    ├── train_with_rllib.py
    ├── rice_rllib.yaml
    ├── torch_models.py
    
    ├── train_with_warp_drive.py
    ├── rice_warpdrive.yaml
    ├── run_cpu_gpu_env_consistency_checks.py
    
    ├── run_unittests.py    
    ├── create_submission_zip.py
    └── evaluate_submission.py   
```

### Environment files
- `rice.py`: This python script contains the base Rice class. This is written in [OpenAI Gym](https://gym.openai.com/) style with the `reset()` and `step()` functionalities. The step() function comprises an implementation of the `climate_and_economy_simulation_step` which dictate the dynamics of the climate and economy simulation, and should not be altered by the user. We have also provided a simple implementation of bilateral negotiation between regions via the `proposal_step()` and `evaluation_step()` methods. Users can extend the simulation by adding additional proposal strategies, for example, and incorporating them in the `step()` function. However, please do not modify any of the equations dictating the environment dynamics in the `climate_and_economy_simulation_step()`. All the helper functions related to modeling the climate and economic simulation are located in `rice_helpers.py`. Region-specific environment parameters are provided in the `region_yamls` directory.


- `rice_step.cu`
This is the CUDA C version of the step() function that is required for use with WarpDrive. To get started with WarpDrive, we recommend following these [tutorials](https://github.com/salesforce/warp-drive/tree/master/tutorials). While WarpDrive requires writing the simulation in CUDA C, it also offers orders-of-magnitude speedups for end-to-end training, since it performs rollouts and training all on the GPU. `rice_cuda.py` nd `rice_build.cu` are necessary files for copying simulation data to the GPU and compiling the CUDA code.  


### Scripts for performing training
Once you build your simulation, you can use either of the following scripts to perform training.
- `train_with_rllib.py`: this script performs end-to-end training with RLlib. The experiment run configuration will be read in from `rice_rllib.yaml`, which contains the environment configuration, logging and saving settings and the trainer and policy network parameters. The duration of training can be set via the `num_episodes` parameter. We have also provided an initial implementation of a linear PyTorch policy model in `torch_models.py`. You can [add other policy models](https://docs.ray.io/en/latest/rllib/rllib-concepts.html) you wish to use into that file.

USAGE: The training script (with RLlib) is invoked using (from the root directory)
```commandline
    python scripts/train_with_rllib.py
```

- `train_with_warp_drive.py`: this script performs end-to-end training with WarpDrive. The experiment run configuration will be read in from `rice_warpdrive.yaml`. Currently, WarpDrive just supports the Advantage Actor-Critic (A2C) and the Proximal Policy Optimization (PPO) algorithms, and the fully-connected policy model.

USAGE: The training script (with WarpDrive) is invoked using
```commandline
    python scripts/train_with_warpdrive.py
```

As training progresses, some key metrics (such as the mean episode reward) are printed on screen for your reference. At the end of training, a zipped submission file is automatically created and saved for your reference. The zipped file essentially comprises the following

- An identifier file (`.rllib` or `.warpdrive`) indicating which framework was used towards training.
- The environment files - `rice.py` and `rice_helpers.py`.
- A copy of the yaml configuration file (`rice_rllib.yaml` or `rice_warpdrive.yaml`) used for training.
- PyTorch policy model(s) (of type ".state_dict") containing the trained weights for the policy network(s). Only the trained policy model for the final timestep will be copied over into the submission zip. If you would like to instead submit the trained policy model at a different timestep, please see the section below on creating your submission file.
- For submissions using WarpDrive, the submission will also contain CUDA-specific files `rice_step.cu` and `rice_cuda.py` that were used for training.

### Scripts for creating the zipped submission file
As mentioned above, the zipped file required for submission is automatically created post-training. However, for any reason (for example, for providing a trained policy model at a different timestep), you can create the zipped submission yourself using the `create_submizzion_zip.py` script. Accordingly, create a new directory (say `submission_dir`) with all the relevant files (see the section above), and you can then simply invoke
```commandline
python scripts/create_submission_zip.py -r <PATH-TO-SUBMISSION-DIR>
```

That will first validate that the submission directory contains all the required files, and then provide you a zipped file that can you use towards your submission.

### Scripts for unit testing
In order to make sure that all the submissions are consistent in that they comply within the rules of the competition, we have also added unit tests. These are automatically run also when the evaluation is performed. The script currently performs the following tests

- Test that the environment attributes (such as the RICE and DICE constants, the simulation period and the number of regions) are consistent with the base environment class that we also provide.
- Test that the `climate_and_economy_simulation_step()` is consistent with the base class. As aforementioned, users are free to add different negotiation strategies such as multi-lateral negotiations or climate clubs, but should not modify the equations underlying the climate and economic dynamics in the world.
- Test that the environment resetting and stepping yield outputs in the desired format (for instance, observations are a dictionary keyed by region id, and so are rewards.)
- If the user used WarpDrive, we also perform consistency checks to verify that the CUDA implementation of the rice environment is consistent with the pythonic version.

USAGE: You may invoke the unit tests on a submission file via
```commandline
python scripts/run_unittests.py -r <PATH-TO-ZIP-FILE>
```

### Scripts for performance evaluation
Before you actually upload your submission files, you can also evaluate and score your submission on your end using this script. The evaluation script essentially validates the submission files, performs unit testing and computes the metrics for evaluation. To compute the metrics, we first instantiate a trainer, load the policy model with the saved parameters, and then generate several episode rollouts to measure the impact of the policy on the environment.

USAGE: You may evaluate the submission file using
```commandline
python scripts/evaluate_submission.py -r <PATH-TO-ZIP-FILE>
```
Please verify that you can indeed evaluate your submission, before actually uploading it.

## What is the evaluation process?
After you submit your solution, we will be using the same evaluation script that is provided to you, to score your submissions, but using several rollout episodes to average the metrics such as the average rewards, the global temperature rise, capital, production, and many more. We will then rank the submissions based on the various metrics.The score computed by the evaluation process should be similar to the score computed on your end, since they use the same scripts.

## Where can I see the leaderboard?
The competition leaderboard is displayed [here](http://34.111.184.174/). After you submit your valid submission, please give it a few minutes to perform an evaluation of your submission and refresh the leaderboard.

## What happens when I make an invalid submission?
An "invalid submission" may refer to a submission wherein some or all of the submission files are missing, or the submission files are inconsistent with the base version, basically anything that fails in the evaluation process. Any invalid solution cannot be evaluated, and hence will not feature in the leaderboard. While we can let you know if your submission is invalid, the process is not automated, so we may not be able to do it promptly. To avoid any issues, please use the `create_submission_zip` script to create your zipped submission file.


## How many submissions are allowed per team?
There is no limit on the number of submissions per team. Feel free to submit as many solutions as you would like. We will only be using your submission with the highest evaluation score towards the leaderboard.

## Requirements


## Docker image
