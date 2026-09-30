"""Train PPO on the example scenario, then plot the learning curves and one episode.

    uv run python main.py

Outputs go to outputs/logs (JSON) and outputs/plots (PNG).
"""

import jax
import jaxnasium as jym
from jaxnasium.algorithms import PPO
from rice_jax.utils import save_training_metrics
from workshop.example_extension.my_scenario import FreeTradeBloc

from workshop import DEFAULT_PPO_PARAMS, log_episode, make_env

SCENARIO = FreeTradeBloc  # The custom senario
NUM_REGIONS = 3  # 3, 7 or 20 are the included values of rice
TOTAL_TIMESTEPS = 500_000  # ~1 min compile + ~30 s training on a laptop GPU

# more keys can be logged by updating "generate_info" in a scenario
LOG_KEYS = ["bloc_mitigation", "outsider_mitigation"]

# enables jit cache to improve compilation times.
# this creates a hidden .jit-cache folder in the current dir, update if
# you'd like it somewhere else
jym.enable_compilation_cache(".jit-cache")


def log_function(info, iteration):
    """Custom log function. Called by PPO every `log_interval` with the `info` dict of that training
    iteration. Every value has shape (num_steps, num_envs)."""
    values = ", ".join(f"{k}={info[k].mean():.2f}" for k in LOG_KEYS if k in info)
    print(f"iteration {iteration}: {values}")


env = make_env(SCENARIO, num_regions=NUM_REGIONS)
train_key, log_key = jax.random.split(jax.random.PRNGKey(0))

ppo = PPO(
    total_timesteps=TOTAL_TIMESTEPS,
    log_function=log_function,  # or "tqdm" for a progress bar
    log_interval=0.1,  # every 10% of the iterations
    num_envs=16,  # lower if memory does not allow
    **DEFAULT_PPO_PARAMS,
)

# Bit awkward, but initializing agents on the cpu is faster; so lets do that upfront
# (in our tests the training compile dropped from ~70 s to ~30 s with 16 envs)
with jax.default_device(jax.devices("cpu")[0]):
    agent = ppo.init_agent(train_key, env)
agent = jax.tree.map(
    lambda x: jax.device_put(x, jax.devices()[0]) if isinstance(x, jax.Array) else x,
    agent,
)

# Not required, but does not hurt: compile up front and report how long it took
train_fn = jym.precompile(ppo.train, train_key, env, agent)
# metrics: mean training episode return per region, per iteration
agent, metrics = train_fn()
save_training_metrics(metrics, ppo.batch_size, "outputs/logs", "outputs/plots")

returns = agent.evaluate(log_key, env, num_eval_episodes=1)
print("Return per region (greedy policy):", {r: v.item() for r, v in returns.items()})

# The same info keys are available per step of a rollout
path, infos = log_episode(env, agent, log_key)
print("Episode log:", path)
for k in LOG_KEYS:
    if k in infos:
        print(f"{k} per step:", [round(x, 2) for x in infos[k].tolist()])
