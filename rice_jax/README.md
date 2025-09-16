# rice_jax: JAX-Based RICE-N Multi-Agent Environment

[![Python 3.11+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

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

## 🚀 Usage

The main entrypoint for an example run + logging is in [`rice_jax/main.py`](rice_jax/main.py).

### Basic Training
```bash
cd rice_jax
python main.py -t 1000000  # Train for 1M timesteps
```


## ⚙️ Configuration

Configuration is handled through command-line arguments using `tyro`. Key settings include:

- **Environment Settings**: Number of regions (3, 7, or 20), reward modes, action discretization
- **Trainer Settings**: PPO hyperparameters, learning rates, training steps
- **Scenarios**: `default`, `optimal_mitigation`, or `basic_club`
- **Agent Types**: `ppo` (default) or `fixed_action` for debugging

### Key Arguments
- `-t, --total_timesteps` - Number of training timesteps (default: 1M)
- `--load_model` - Path to saved model for evaluation
- `--agent` - Agent type: `ppo` (default) or `fixed_action` 
- `--scenario` - Environment scenario: `default`, `optimal_mitigation`, or `basic_club`
- `--seed` - Random seed (default: 0)

### Environment Settings
- `--env_settings.num_regions` - Number of regions: 3, 7, or 20
- `--env_settings.diff_reward_mode` - Use differential rewards
- `--env_settings.negotiation_on` - Enable negotiation between regions

### Training Settings  
- `--trainer_settings.learning_rate` - Learning rate (default: 2.5e-4)
- `--trainer_settings.num_envs` - Number of parallel environments (default: 4)
- `--trainer_settings.num_steps` - Steps per update (default: 100)

All settings can be overridden via command-line arguments. See usage the main file for exact parameters or run `python main.py --help` for all options.

## 📊 Logging & Visualization

The included main file automatically runs a few episodes after training. The episode logs of these episodes
are saved and visualizations can be created (see plotting_example.ipynb) for an example to create plots from the episode logs.

## 🌍 Scenarios

Three different environment scenarios are available:

- **`default`** - Standard RICE-N climate-economy model
- **`optimal_mitigation`** - Environment with optimal mitigation strategies
- **`basic_club`** - Basic climate club formation scenario

Select scenario with `--scenario` argument.

## Wandb

No wandb config is included for now.