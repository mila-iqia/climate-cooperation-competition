# Code overview

**For the mathematical background and scientific references, please see [the white paper](https://github.com/mila-iqia/climate-cooperation-competition/blob/website/website/src/pdf/ai-for-global-climate-cooperation-competition_white-paper.pdf).**


We provide the base version of the RICE-N (regional integrated climate environment) simulation environment written in Python (`rice.py`).


## File Structure

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


## Environment files

- `rice.py`: This python script contains the base Rice class. This is written in [OpenAI Gym](https://gym.openai.com/) style with the `reset()` and `step()` functionalities. The step() function comprises an implementation of the `climate_and_economy_simulation_step` which dictate the dynamics of the climate and economy simulation, and should not be altered by the user. We have also provided a simple implementation of bilateral negotiation between regions via the `proposal_step()` and `evaluation_step()` methods. Users can extend the simulation by adding additional proposal strategies, for example, and incorporating them in the `step()` function. However, please do not modify any of the equations dictating the environment dynamics in the `climate_and_economy_simulation_step()`. All the helper functions related to modeling the climate and economic simulation are located in `rice_helpers.py`. Region-specific environment parameters are provided in the `region_yamls` directory.


- `rice_step.cu`
This is the CUDA C version of the step() function that is required for use with WarpDrive. To get started with WarpDrive, we recommend following these [tutorials](https://github.com/salesforce/warp-drive/tree/master/tutorials). While WarpDrive requires writing the simulation in CUDA C, it also offers orders-of-magnitude speedups for end-to-end training, since it performs rollouts and training all on the GPU. `rice_cuda.py` nd `rice_build.cu` are necessary files for copying simulation data to the GPU and compiling the CUDA code.

While implementing the simulation in CUDA C on the GPU offers significantly faster simulations, it requires careful memory management. To make sure that everything works properly, one approach is to first implement your simulation logic in Python. You can then implement the same logic in CUDA C and check the simulation behaviors are the same. To help with this process, we provide an environment consistency checker method to do consistency tests between Python and CUDA C simulations. Before training your CUDA C code, please run the consistency checker to ensure the Python and CUDA C implementations are consistent.
```commandline
python scripts/run_env_cpu_gpu_consistency_checks.py
```