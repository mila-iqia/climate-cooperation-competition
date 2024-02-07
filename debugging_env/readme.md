# What is this folder about?  
This folder contains the debugging environment for the project. Allowing the user to debug the project in a wandb controlled environment with static (and predetermined) actions.  

# Login to wandb  
It is recommended that you login to wandb before running the environment.  
```bash
wandb login
```

# How to use this env?  
If you have a specific parameter set would like to check, you can use the `trajectory_mitigation_saving_simulations.py` file to run the environment with the specific parameters.  

You should modify the line in the main function:  
for example:  
```python
run_single_experiment(is_wandb=True, mitigation_rate = [0,5,0,0,5,0,0,0,0,0,0,0,0,0,0,9,0,8,0], savings_rate=2.5, pliability=0.9, damage_type="updated", abatement_cost_type="path_dependent", debugging_folder=None)
```
which means the exp is logged with `wandb`, the `mitigation_rate` is set to `[0,5,0,0,5,0,0,0,0,0,0,0,0,0,0,9,0,8,0]` and the `savings_rate` is set to `2.5`. Both of them can be set to a constant or a time series of actions.

the `pliability` is set to `0.9`, the `damage_type` is set to `updated`, the `abatement_cost_type` is set to `path_dependent` and the `debugging_folder` is set to `None`.  

`debugging_folder` refers to which folder contains the yaml file for the environment. If you want to use the default environment (27 regions), you can set it to `None` or `region_yamls`. There are a 2 region version in the `2_region`. To use it, just set the `debugging_folder` to `2_region`.

`abatement_cost_type`, `pliability` are responsible for the calculation of abatement.

To run a batch of parameters, you can use the `run.py` file to sweep through the parameters space.  
```python
python run.py
```