# RICE-JAX workshop: getting started

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
cd workshop
uv sync          # installs rice_jax (editable, from ../rice_jax) and its dependencies
uv run python main.py
```

This trains PPO on the example scenario and writes learning curves, an episode log
(JSON) and plots to `outputs/`. The first minute is JAX compiling the training loop;
it's cached in `.jit-cache/`, so rerunning with unchanged code and settings starts in ~10 s.

## What's here

- `main.py`: sets up PPO, trains on a scenario and plots the results.
- `workshop/util.py`: `make_env`, `log_episode` and the default PPO hyperparameters (`DEFAULT_PPO_PARAMS`).
- `workshop/example_extension/my_scenario.py`: an example scenario, a free-trade bloc.

To build your own, copy `example_extension/` to a new folder next to it, edit
`my_scenario.py`, and point the scenario import in `main.py` to your folder.

## Writing a scenario

Subclass `Rice` and override the hooks you need (call `super()` first):

| Hook | Controls |
|------|----------|
| `generate_action_masks(state)` | which action levels each region may choose |
| `generate_observation(state)` | what each region observes |
| `generate_rewards(new_state, old_state)` | each region's reward |
| `generate_info(state, actions, rewards)` | extra values to log each step |
| `calc_damages`, `calc_abatement_costs`, `calc_utilities`, ... | the model's equations |

See `rice_jax/rice_jax/core/env.py` for all of them, and
`rice_jax/rice_jax/core/scenarios.py` for more examples. The keys of `state` are
the fields in `episode_data` of any episode log.

## Logging your own values

Whatever `generate_info` adds to the info dict is available in three places:

- **During training:** PPO calls `log_function(info, iteration)` every
  `log_interval`, with each value shaped `(num_steps, num_envs)`. See
  `log_function` in `main.py`.
- **In a rollout:** `log_episode` returns the infos stacked per step.
- **In the episode JSON:** they're saved next to the full state.

The `metrics` that `ppo.train` returns only hold the mean episode return per region.
