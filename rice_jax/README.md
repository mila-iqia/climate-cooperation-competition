# rice_jax: JAX-Based RICE-N Multi-Agent Environment

[![Python 3.10+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

This subfolder contains a high-performance implementation of the Rice-N climate-economy model based on Jax and JymKit.

---

## 📦 Installation

For those using [uv](https://docs.astral.sh/uv/getting-started/installation/), the implementation can directly be run via `uv run main.py`.

```bash
cd rice_jax
uv run main.py <options>
```

Alternatively, install the project as an editable package in your favourite virtual environment software. E.g. using conda:

```bash
cd rice_jax
conda create -n rice-jax python=3.11
conda activate rice-jax
pip install -e .

python main.py
```

for CUDA support, additionally run `pip install jax[cuda]`.

---

## ⚙️ Configuration

All environment and agent settings live in the [`config_yamls`](rice_jax/rice_jax/config_yamls) folder. By default you can start from:

[rice_jax/rice_jax/config_yamls/default.yml](rice_jax/rice_jax/config_yamls/default.yml)

```yaml
seed: 0
num_simultaneous_training_runs: 2

trainer_settings:
  ...

env_settings:
  num_regions: 3             # 3, 7, or 20
  scenario: "default"   
  ...
```

---

## 🚀 Usage

The main entrypoint is [`rice_jax/main.py`](rice_jax/main.py).

```bash
cd rice_jax
python main.py \
  -t 1e6 \
  --yaml default \
  [--load_model path/to/checkpoint.eqx] \
  [--debug] \
  [--wandb]
```

- `-t`  
  convenience parameter to override `trainer_settings.total_timesteps` (defaults to 1e6)

- `--yaml, -y`  
  selects a config in [`config_yamls/{yaml}.yml`](rice_jax/rice_jax/config_yamls) (defaults to "default")

- `--load_model, -l`  
  path to a saved Equinox model (`.eqx`) or evaluate

- `--debug, -d`  
  run in debug mode. Uses the agent specified in main as "DebugAgent". This agent uses predefined actions at every step.

- `--wandb, -w`  
  enable Weights & Biases logging

### 1. Training a new agent

When the `debug` and `load_model` flag are *not* set. A new PPO agent will be trained from scratch based on the settings of the provided YAML file.

The training loop is defined in [`rice_jax/main.py`](rice_jax/main.py). Models are saved to `saved_models/{scenario}_{num_regions}_{timestamp}.eqx`. Training should, when CUDA is enabled, take roughly 1-3 minutes. Note that when starting training, the training function is first compiled. It may then appear as if nothing is happening for the first 30-60 minutes. Additionally, when wandb is enabled, no feedback in the terminal is given until training finishes.

if `num_simultaneous_training_runs` > 1, multiple agents will be trained simultaneously on different seeds. Wandb is disabled during training in this mode.

After training, all will play a few additional episodes as evaluation. See the next section.

### 2. Evaluation

After training, when loading a pre-trained model, or when using the debug agent, the agent will perform a few evaluation runs. During these runs, the environment state at every step will be logged and returned. If the `wandb` flag is set, these state logs will be logged to wandb where each environment run can be inspected.
