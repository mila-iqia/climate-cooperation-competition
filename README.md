# Competition: Fostering Global Cooperation to Mitigate Climate Change

[![PyTorch 1.9.0](https://img.shields.io/badge/PyTorch-1.9.0-ee4c2c?logo=pytorch&logoColor=white%22)](https://pytorch.org/docs/1.12/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-3713/)
[![Warp drive 1.7.0](https://img.shields.io/badge/warp_drive-1.7.0-blue.svg)](https://github.com/salesforce/warp-drive/)
[![Ray 1.0.0](https://img.shields.io/badge/ray[rllib]-1.0.0-blue.svg)](https://docs.ray.io/en/latest/index.html)
[![Paper](http://img.shields.io/badge/paper-arxiv.2208.07004-B31B1B.svg)](https://arxiv.org/abs/2208.07004)
[![Code Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mila-iqia/climate-cooperation-competition/blob/main/Colab_Tutorial.ipynb)

<a href="https://www.kaggle.com/kernels/fork-version/105300459"><img src="https://www.kaggle.com/static/images/site-logo.svg" align="left" height="40" width="40" ></a>(Code Tutorial Notebook on Kaggle with free GPU available)


This is the code respository for the competition on modeling global cooperation in the RICE-N Integrated Assessment Model. This competition is co-organized by MILA and Salesforce Research.

The RICE-N IAM is an agent-based model that incorporates DICE climate-economic dynamics and multi-lateral negotiation protocols between several fictitious nations.

In this competition, you will design negotiation protocols and contracts between nations. You will use the simulation and agents to evaluate their impact on the climate and the economy. 

We recommend that GPU users use ``warp_drive`` and CPU users use ``rllib``.

## Tutorial
- For all information and the leaderboard, see [our official website](https://www.ai4climatecoop.org).
- [How to Get Started](getting_started.ipynb)
- [Code Tutorial Notebook with **free GPU**](https://colab.research.google.com/github/mila-iqia/climate-cooperation-competition/blob/main/Colab_Tutorial.ipynb)
- [Code Kaggle Tutorial Notebook with **free GPU**](https://www.kaggle.com/kernels/fork-version/105300459)

## Resources
- For the mathematical background and scientific references, please see [the white paper](https://deliverypdf.ssrn.com/delivery.php?ID=579098091025080122123095015088114126057046084059055038121023114094110112070107123088059057002107022006023123122016086089001013042072002040020075022078097115093071118048047053064022064117095120085074022123010099031092026025094015125094099080071097079070&EXT=pdf&INDEX=TRUE).
- Other free GPU resources: [Baidu Paddle](https://aistudio.baidu.com/), [MegStudio](https://studio.brainpp.com/)


## Installation

Notice: we recommend using `Linux` or `MacOS`. For Windows users, we recommend to use virtual machine running `Ubuntu 20.04` or [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install).

You can get a copy of the code by cloning the repo using Git: 

```
git clone https://github.com/mila-iqia/climate-cooperation-competition
cd climate-cooperation-competition
```

As an alternative, one can also use:

```
git clone https://e.coding.net/ai4climatecoop/ai4climatecoop/climate-cooperation-competition.git
cd climate-cooperation-competition
```

We recommend using a virtual environment (such as provided by ```virtualenv``` or Anaconda).

You can install the dependencies using pip:

```
pip install -r requirements.txt
```

## Get Started
Then run the getting started Jupyter notebook, by starting Jupyter:

```
jupyter notebook
``` 

and then navigating to [getting_started.ipynb](getting_started.ipynb).

It provides a quick walkthrough for registering for the competition and creating a valid submission.


## Training with reinforcement learning

RL agents can be trained using the RICE-N simulation using one of these two frameworks:

1. [RLlib](https://docs.ray.io/en/latest/rllib/index.html#:~:text=RLlib%20is%20an%20open%2Dsource,large%20variety%20of%20industry%20applications): The pythonic environment can be trained on your local CPU machine using open-source RL framework, RLlib.
2. [WarpDrive](https://github.com/salesforce/warp-drive): WarpDrive is a GPU-based framework that allows for [over 10x faster training](https://arxiv.org/pdf/2108.13976.pdf) compared to CPU-based training. It requires the simulation to be written out in CUDA C, and we also provide a starter version of the simulation environment written in CUDA C ([rice_step.cu](rice_step.cu))

We also provide starter scripts to train the simulation you build with either of the above frameworks.

Note that we only allow these two options, since our backend submission evaluation process only supports these at the moment.


For training with RLlib, `rllib (1.0.0)`, `torch (1.9.0)` and `gym (0.21)` packages are required.



For training with WarpDrive, the `rl-warp-drive (>=1.6.5)` package is needed.

Note that these requirements are automatically installed (or updated) when you run the corresponding training scripts.


## Docker image (for GPU users)

We have also provided a sample dockerfile for your reference. It mainly uses a Nvidia PyTorch base image, and installs the `pycuda` package as well. Note: `pycuda` is only required if you would like to train using WarpDrive.


## Docker image (for CPU users)

Thanks for the contribution from @muxspace. We also have an end-to-end docker environment ready for CPU users. Please refer to [README_CPU.md](README_CPU.md) for more details.

# Customizing and running the simulation

See the [Colab_Tutorial.ipynb](Tutorial.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mila-iqia/climate-cooperation-competition/blob/main/Colab_Tutorial.ipynb) for details. 

It provides examples on modifying the code to implement different negotiation protocols. It describes ways of changing the agent observations and action spaces corresponding to the proposed negotiation protocols and implementing the negotiation logic in the provided code. 

The notebook has a walkthrough of how to train RL agents with the simulation code and how to visualize results from the simulation after running it with a set of agents.

For those who have **limited access to Colab**, please try to use [**free GPUs on Kaggle**](https://www.kaggle.com/kernels/fork-version/105300459). Please notice that Kaggle platform requires mobile phone verification to be able to access the GPUs. One may find the **settings** to get GPUs and internet connect on the right hand side after clicking on the link above and login.

# Training RL agents in your simulation

Once you build your simulation, you can use either of the following scripts to perform training.

- [train_with_rllib.py](/scripts/train_with_rllib.py): this script performs end-to-end training with RLlib. The experiment run configuration will be read in from [rice_rllib.yaml](/scripts/rice_rllib.yaml), which contains the environment configuration, logging and saving settings and the trainer and policy network parameters. The duration of training can be set via the `num_episodes` parameter. We have also provided an initial implementation of a linear PyTorch policy model in [torch_models.py](/scripts/torch_models.py). You can [add other policy models](https://docs.ray.io/en/latest/rllib/rllib-concepts.html) you wish to use into that file.

USAGE: The training script (with RLlib) is invoked using (from the root directory)
```commandline
    python scripts/train_with_rllib.py
```

- [train_with_warp_drive.py](/scripts/train_with_warp_drive.py): this script performs end-to-end training with WarpDrive. The experiment run configuration will be read in from [rice_warpdrive.yaml](/scripts/rice_warpdrive.yaml). Currently, WarpDrive just supports the Advantage Actor-Critic (A2C) and the Proximal Policy Optimization (PPO) algorithms, and the fully-connected policy model.

USAGE: The training script (with WarpDrive) is invoked using
```commandline
    python scripts/train_with_warpdrive.py
```

As training progresses, some key metrics (such as the mean episode reward) are printed on screen for your reference. At the end of training, a zipped submission file is automatically created and saved for your reference. The zipped file essentially comprises the following

- An identifier file (`.rllib` or `.warpdrive`) indicating which framework was used towards training.
- The environment files - [rice.py](rice.py) and [rice_helpers.py](rice_helpers.py).
- A copy of the yaml configuration file ([rice_rllib.yaml](/scripts/rice_rllib.yaml) or [rice_warpdrive.yaml](/scripts/rice_warpdrive.yaml)) used for training.
- PyTorch policy model(s) (of type ".state_dict") containing the trained weights for the policy network(s). Only the trained policy model for the final timestep will be copied over into the submission zip. If you would like to instead submit the trained policy model at a different timestep, please see the section below on creating your submission file.
- For submissions using WarpDrive, the submission will also contain CUDA-specific files [rice_step.cu](rice_step.cu) and [rice_cuda](rice_cuda.py) that were used for training.


# Contributing

We are always looking for contributors from various domains to help us make this simulation more realistic. 

If there are bugs or corner cases, please open a PR detailing the issue and consider submitting to Track 3!


# Citation

To cite this code, please use the information in [CITATION.cff](CITATION.cff) and the following bibtex entry:

```
@software{Zhang_RICE-N_2022,
author = {Zhang, Tianyu and Srinivasa, Sunil and Williams, Andrew and Phade, Soham and Zhang, Yang and Gupta, Prateek and Bengio, Yoshua and Zheng, Stephan},
month = {7},
title = {{RICE-N}},
url = {https://github.com/mila-iqia/climate-cooperation-competition},
version = {1.0.0},
year = {2022}
}

@misc{https://doi.org/10.48550/arxiv.2208.07004,
  doi = {10.48550/ARXIV.2208.07004},
  url = {https://arxiv.org/abs/2208.07004},
  author = {Zhang, Tianyu and Williams, Andrew and Phade, Soham and Srinivasa, Sunil and Zhang, Yang and Gupta, Prateek and Bengio, Yoshua and Zheng, Stephan},
  title = {AI for Global Climate Cooperation: Modeling Global Climate Negotiations, Agreements, and Long-Term Cooperation in RICE-N},  
  publisher = {arXiv},
  year = {2022}
}

```

# License

For license information, see ```LICENSE.txt```.
