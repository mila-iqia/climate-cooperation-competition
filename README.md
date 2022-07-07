# Competition: Fostering Global Cooperation to Mitigate Climate Change

This is the code respository for the competition on modeling global cooperation in the RICE-N Integrated Assessment Model. This competition is co-organized by MILA and Salesforce Research.

The RICE-N IAM is an agent-based model that incorporates DICE climate-economic dynamics and multi-lateral negotiation protocols between several fictitious nations.

In this competition, you will design negotiation protocols and contracts between nations. You will use the simulation and agents to evaluate their impact on the climate and the economy. 


## Resources

- For all information and the leaderboard, see [https://mila-iqia.github.io/climate-cooperation-competition](https://mila-iqia.github.io/climate-cooperation-competition).
- [Getting Started Jupyter Notebook](getting_started.ipynb)
- [Colab Tutorial Notebook](https://colab.research.google.com/github/mila-iqia/climate-cooperation-competition/blob/main/Colab_Tutorial.ipynb)
- For the mathematical background and scientific references, please see [the white paper](https://github.com/mila-iqia/climate-cooperation-competition/blob/website/website/src/pdf/ai-for-global-climate-cooperation-competition_white-paper.pdf).



## Installation

You can get a copy of the code by cloning the repo using Git: 

```
git clone https://github.com/mila-iqia/climate-cooperation-competition
cd climate-cooperation-competition
```

We recommend using a virtual environment (such as provided by ```virtualenv``` or Anaconda).

You can install the dependencies using pip:

```
pip install -r requirements.txt
```

Then run the getting started Jupyter notebook, by starting Jupyter:

```
jupyter notebook
``` 

and then navigating to ```getting_started.ipynb```.

It provides a quick walkthrough for registering for the competition and creating a valid submission.


## Training with reinforcement learning

RL agents can be trained using the RICE-N simulation using one of these two frameworks:

1. [RLlib](https://docs.ray.io/en/latest/rllib/index.html#:~:text=RLlib%20is%20an%20open%2Dsource,large%20variety%20of%20industry%20applications): The pythonic environment can be trained on your local CPU machine using open-source RL framework, RLlib.
2. [WarpDrive](https://github.com/salesforce/warp-drive): WarpDrive is a GPU-based framework that allows for [over 10x faster training](https://arxiv.org/pdf/2108.13976.pdf) compared to CPU-based training. It requires the simulation to be written out in CUDA C, and we also provide a starter version of the simulation environment written in CUDA C (`rice_step.cu`)

We also provide starter scripts to train the simulation you build with either of the above frameworks.

Note that we only allow these two options, since our backend submission evaluation process only supports these at the moment.


For training with RLlib, `rllib (1.0.0)`, `torch (1.10)` and `gym (0.21)` packages are required.

For training with WarpDrive, the `rl-warp-drive (>=1.6.5)` package is needed.

Note that these requirements are automatically installed (or updated) when you run the corresponding training scripts.


## Docker image (for GPU training)

We have also provided a sample dockerfile for your reference. It mainly uses a Nvidia PyTorch base image, and installs the `pycuda` package as well. Note: `pycuda` is only required if you would like to train using WarpDrive.


# Customizing and running the simulation

See the ```Tutorial.ipynb```. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mila-iqia/climate-cooperation-competition/blob/main/Colab_Tutorial.ipynb) for details. 

It provides examples on modifying the code to implement different negotiation protocols. It describes ways of changing the agent observations and action spaces corresponding to the proposed negotiation protocols and implementing the negotiation logic in the provided code. 

The notebook has a walkthrough of how to train RL agents with the simulation code and how to visualize results from the simulation after running it with a set of agents.


# Training RL agents in your simulation

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


# Contributing

We are always looking for contributors from various domains to help us make this simulation more realistic. 

If there are bugs or corner cases, please open a PR detailing the issue and consider submitting to Track 3!


# Citation

To cite this code, please use the information in ```CITATION.cff``` and the following bibtex entry:

```
@software{Zhang_RICE-N_2022,
author = {Zhang, Tianyu and Srinivasa, Sunil and Wiliams, Andrew and Phade, Soham and Zhang, Yang and Gupta, Prateek and Bengio, Yoshua and Zheng, Stephan},
month = {7},
title = {{RICE-N}},
url = {https://github.com/mila-iqia/climate-cooperation-competition},
version = {1.0.0},
year = {2022}
}
```


# License

For license information, see ```LICENSE.txt```.
