"""
Script to train the damage parameter (xa_updated) to match ground truth gross outputs.

This demonstrates:
1. Generating ground truth runs with a target damage parameter
2. Initializing the parameter at 0
3. Using gradient descent to recover the parameter
4. Computing uncertainty estimates using the Hessian (2nd-order derivatives)
5. Generating ensemble predictions with uncertainty bounds
6. Visualizing the results with 3 plots:
   - Gross outputs over time
   - Parameter distribution (Gaussian approximation)
   - Histogram of total outputs (sum over entire rollout) with ground truth
"""

import jax
import jax.numpy as jnp
from rice_jax import Rice
from rice_jax.utils import load_region_yamls, i_to_agent_str
import optax
import matplotlib.pyplot as plt
import numpy as np

# Configuration
NUM_REGIONS = 3
TARGET_DMG_PARAMETER = 0.7438  # The "true" damage parameter we want to recover
LEARNING_RATE = 0.005
NUM_ITERATIONS = 400
NUM_ENSEMBLE_SAMPLES = 100


def setup_environment(dmg_parameter):
    """Create environment with specified damage parameter"""
    region_params = load_region_yamls(NUM_REGIONS)
    region_params.xa_updated = dmg_parameter
    env = Rice(
        num_regions=NUM_REGIONS,
        region_params=region_params,
        dmg_function="updated"  # Use the 'updated' damage function
    )
    return env


def get_fixed_actions(env: Rice):
    """Return fixed actions for all agents"""
    actions = optax.tree.zeros_like(env.sample_action(jax.random.PRNGKey(1)))
    for a in range(env.num_regions):
        actions[i_to_agent_str(a)]["mitigation_rate"] = jnp.array(0.0)
        actions[i_to_agent_str(a)]["savings_rate"] = jnp.array(2.5)
    return actions


def step_env(env: Rice, seed: int, state: dict, actions: dict):
    """Single environment step"""
    key = jax.random.PRNGKey(seed)
    timestep, new_state = env.step(key, state, actions)
    return new_state, new_state


def do_rollout(dmg_parameter, *, return_all_states: bool = False):
    """
    Run a full rollout with the given damage parameter.
    Returns sum of gross outputs over all timesteps (and optionally all states).
    """
    env = setup_environment(dmg_parameter)
    actions = get_fixed_actions(env)
    _, state = env.reset(jax.random.PRNGKey(0))
    
    final_state, stack_of_states = jax.lax.scan(
        lambda s, i: step_env(env, i, s, actions),
        state,
        jnp.arange(env.episode_length - 1)
    )
    
    sum_gross_outputs = jnp.sum(stack_of_states["gross_output_all_regions"])
    
    if return_all_states:
        return sum_gross_outputs, stack_of_states
    return sum_gross_outputs


def compute_hessian_and_uncertainty(dmg_parameter, ground_truth_sum_outputs, sigma2_obs=None):
    """
    Compute the Hessian (second-order derivative) at the optimized parameter
    to estimate uncertainty using the curvature of the loss function.
    
    Returns:
        hessian_L: Second derivative of loss w.r.t. parameter
        std_theta: Standard deviation estimate
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
    """
    f = do_rollout(dmg_parameter)
    f_prime = jax.grad(do_rollout)(dmg_parameter)
    f_second = jax.grad(lambda t: jax.grad(do_rollout)(t))(dmg_parameter)
    residual = f - ground_truth_sum_outputs

    # exact second derivative of your MSE loss L = (f-y)^2
    hessian_L = 2.0 * (f_prime ** 2) + 2.0 * residual * f_second

    if sigma2_obs is None:
        if jnp.abs(residual) < 1e-12:
            raise RuntimeError("Residual ≈ 0 and sigma2_obs not provided. Hessian-based variance is not identifiable.")
        # crude estimate of sigma2 from residual (only valid if you have multiple datapoints)
        sigma2_obs = float((residual ** 2))

    var_theta = (2.0 * sigma2_obs) / jnp.maximum(hessian_L, 1e-12)
    std_theta = jnp.sqrt(var_theta)

    ci_lower = dmg_parameter - 1.96 * std_theta
    ci_upper = dmg_parameter + 1.96 * std_theta
    
    return hessian_L, std_theta, ci_lower, ci_upper


def generate_ensemble_predictions(dmg_parameter, std_dev, num_samples=100):
    """
    Generate an ensemble of predictions by sampling from the Gaussian
    distribution around the optimized parameter.
    
    Returns:
        sampled_params: Array of sampled parameters
        ensemble_outputs: Array of gross outputs over time for each sample
        ensemble_temps: Array of temperatures over time for each sample
    """
    key = jax.random.PRNGKey(42)
    sampled_params = dmg_parameter + std_dev * jax.random.normal(key, (num_samples,))
    
    ensemble_outputs = []
    ensemble_temps = []
    
    for i, param in enumerate(sampled_params):
        _, states = do_rollout(float(param), return_all_states=True)
        outputs_over_time = jnp.sum(states["gross_output_all_regions"], axis=1)
        temps_over_time = states["global_temperature"][:, 0]
        ensemble_outputs.append(outputs_over_time)
        ensemble_temps.append(temps_over_time)
    
    ensemble_outputs = jnp.array(ensemble_outputs)
    ensemble_temps = jnp.array(ensemble_temps)
    
    return sampled_params, ensemble_outputs, ensemble_temps


def plot_gross_outputs_over_time(ground_truth, initial, trained, 
                                  target_param, trained_param, output_path):
    """Plot for gross outputs over time."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.plot(ground_truth, label=fr'Ground Truth ($a={target_param:.4f}$)', 
            linewidth=3, color='green', alpha=0.8)
    ax.plot(initial, label=fr'Before Training ($a=0.0$)', 
            linewidth=2.5, color='red', linestyle='--', alpha=0.7)
    ax.plot(trained, label=fr'After Training ($a={trained_param:.4f}$)', 
            linewidth=2.5, color='blue', linestyle=':', alpha=0.8)
    
    ax.set_xlabel('Timestep', fontsize=13)
    ax.set_ylabel('Sum of Gross Outputs (All Regions)', fontsize=13)
    ax.set_title('Gross Outputs Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_parameter_distribution(trained_param, std_dev, target_param, 
                                ci_lower, ci_upper, output_path):
    """Plot for parameter distribution with Gaussian fit."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x_range = np.linspace(
        float(trained_param - 4 * std_dev), 
        float(trained_param + 4 * std_dev), 
        200
    )
    
    gaussian = (1 / (float(std_dev) * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x_range - float(trained_param)) / float(std_dev)) ** 2)
    
    ax.plot(x_range, gaussian, 'b-', linewidth=3, label='Gaussian Fit', alpha=0.8)
    ax.fill_between(x_range, 0, gaussian, alpha=0.2, color='blue')
    
    ci_mask = (x_range >= float(ci_lower)) & (x_range <= float(ci_upper))
    ax.fill_between(x_range[ci_mask], 0, gaussian[ci_mask], 
                    alpha=0.3, color='blue', label=f'95% CI [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    ax.axvline(x=target_param, color='green', linestyle='--', linewidth=2.5, 
              label=f'Target Parameter ({target_param:.4f})', alpha=0.8)
    ax.axvline(x=float(trained_param), color='darkblue', linestyle='-', linewidth=2.5,
              label=f'Trained Parameter ({float(trained_param):.4f})', alpha=0.8)
    
    ax.set_xlim(0.6, 0.8)
    ax.set_xlabel(r'Damage Parameter $a$', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title('Parameter Distribution (Gaussian Approximation)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_total_output_histogram(ensemble_outputs, ground_truth_total, 
                                 num_samples, output_path):
    """Plot for histogram of total outputs over entire rollout."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    total_outputs = jnp.sum(ensemble_outputs, axis=1)
    
    mean_total = float(jnp.mean(total_outputs))
    std_total = float(jnp.std(total_outputs))
    percentile_2_5 = float(jnp.percentile(total_outputs, 2.5))
    percentile_97_5 = float(jnp.percentile(total_outputs, 97.5))
    
    n, bins, patches = ax.hist(np.array(total_outputs), bins=30, density=True, 
                               alpha=0.7, color='skyblue', edgecolor='black', 
                               linewidth=1.2, label='Ensemble Distribution')
    
    x_range = np.linspace(float(jnp.min(total_outputs)), float(jnp.max(total_outputs)), 200)
    gaussian = (1 / (std_total * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x_range - mean_total) / std_total) ** 2)
    ax.plot(x_range, gaussian, 'b-', linewidth=2.5, alpha=0.8, label='Gaussian Fit')
    
    ax.axvline(x=ground_truth_total, color='green', linestyle='--', linewidth=3, 
              label=f'Ground Truth ({ground_truth_total:.1f})', alpha=0.9, zorder=10)
    ax.axvline(x=mean_total, color='blue', linestyle='-', linewidth=2.5,
              label=f'Ensemble Mean ({mean_total:.1f})', alpha=0.8)
    
    ax.axvspan(percentile_2_5, percentile_97_5, alpha=0.2, color='blue',
              label=f'95% CI [{percentile_2_5:.1f}, {percentile_97_5:.1f}]')
    
    ax.set_xlabel('Total Sum of Gross Outputs (Entire Rollout)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title(f'Distribution of Total Outputs Over Full Episode ({num_samples} ensemble members)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("================")    
    print("Training Damage Parameter to Match Ground Truth Gross Outputs")
    print("================")
    
    print(f"Step 1:Generating ground truth data with damage parameter = {TARGET_DMG_PARAMETER}")
    ground_truth_sum_outputs, ground_truth_states = do_rollout(
        TARGET_DMG_PARAMETER, return_all_states=True
    )
    print(f"Ground truth sum of gross outputs: {ground_truth_sum_outputs:.2f}")
    
    ground_truth_outputs_over_time = jnp.sum(
        ground_truth_states["gross_output_all_regions"], axis=1
    )
    ground_truth_temps_over_time = ground_truth_states["global_temperature"][:, 0]
    
    print(f"Step 2: Initializing trainable damage parameter at 0.0")
    trainable_dmg_parameter = jnp.array(0.0)
    
    initial_sum_outputs, initial_states = do_rollout(
        trainable_dmg_parameter, return_all_states=True
    )
    initial_outputs_over_time = jnp.sum(
        initial_states["gross_output_all_regions"], axis=1
    )
    initial_temps_over_time = initial_states["global_temperature"][:, 0]
    print(f"Initial sum of gross outputs: {initial_sum_outputs:.2f}")
    print(f"Initial loss: {(initial_sum_outputs - ground_truth_sum_outputs)**2:.2f}")
    
    def loss_fn(dmg_parameter):
        """Loss function: MSE between predicted and ground truth sum of outputs"""
        predicted_sum_outputs = do_rollout(dmg_parameter)
        loss = (predicted_sum_outputs - ground_truth_sum_outputs) ** 2
        return loss
    
    print(f"Step 3: Training with learning rate = {LEARNING_RATE}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print("================")
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(trainable_dmg_parameter)
    
    loss_history = []
    param_history = []
    
    for iteration in range(NUM_ITERATIONS):
        loss, grads = jax.value_and_grad(loss_fn)(trainable_dmg_parameter)
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable_dmg_parameter = optax.apply_updates(trainable_dmg_parameter, updates)
        
        loss_history.append(float(loss))
        param_history.append(float(trainable_dmg_parameter))
        
        if iteration % 20 == 0 or iteration == NUM_ITERATIONS - 1:
            print(f"Iteration {iteration:3d}: Loss = {loss:12.4f}, "
                  f"Parameter = {trainable_dmg_parameter:.6f}")
    
    print("================")
    print(f"Training complete!")
    print(f"Final trained parameter: {trainable_dmg_parameter:.6f}")
    print(f"Target parameter: {TARGET_DMG_PARAMETER:.6f}")
    print(f"Difference: {abs(trainable_dmg_parameter - TARGET_DMG_PARAMETER):.6f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    
    trained_sum_outputs, trained_states = do_rollout(
        trainable_dmg_parameter, return_all_states=True
    )
    trained_outputs_over_time = jnp.sum(
        trained_states["gross_output_all_regions"], axis=1
    )
    trained_temps_over_time = trained_states["global_temperature"][:, 0]
    
    print(f"Step 4: Computing uncertainty estimates using Hessian (2nd derivative)")
    hessian, std_dev, ci_lower, ci_upper = compute_hessian_and_uncertainty(
        trainable_dmg_parameter, ground_truth_sum_outputs
    )
    print(f"Hessian (2nd derivative): {hessian:.4f}")
    print(f"Standard deviation: {std_dev:.6f}")
    print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"Target within CI: {ci_lower <= TARGET_DMG_PARAMETER <= ci_upper}")
    
    print(f"Step 5: Generating {NUM_ENSEMBLE_SAMPLES} ensemble predictions from Gaussian")
    sampled_params, ensemble_outputs, ensemble_temps = generate_ensemble_predictions(
        trainable_dmg_parameter, std_dev, num_samples=NUM_ENSEMBLE_SAMPLES
    )
    print(f"Sampled parameters range: [{jnp.min(sampled_params):.6f}, {jnp.max(sampled_params):.6f}]")
    
    ensemble_mean = jnp.mean(ensemble_outputs, axis=0)
    ensemble_std = jnp.std(ensemble_outputs, axis=0)
    ensemble_percentile_2_5 = jnp.percentile(ensemble_outputs, 2.5, axis=0)
    ensemble_percentile_97_5 = jnp.percentile(ensemble_outputs, 97.5, axis=0)
    

    
    print(f"Step 6: Generating 3 plots...")
    
    plot_gross_outputs_over_time(
        ground_truth_outputs_over_time,
        initial_outputs_over_time,
        trained_outputs_over_time,
        TARGET_DMG_PARAMETER,
        float(trainable_dmg_parameter),
        'rice_jax/plot_gross_outputs_over_time.png'
    )
    
    plot_parameter_distribution(
        trainable_dmg_parameter,
        std_dev,
        TARGET_DMG_PARAMETER,
        ci_lower,
        ci_upper,
        'rice_jax/plot_parameter_distribution.png'
    )

    ground_truth_total_output = float(jnp.sum(ground_truth_outputs_over_time))
    plot_total_output_histogram(
        ensemble_outputs,
        ground_truth_total_output,
        NUM_ENSEMBLE_SAMPLES,
        'rice_jax/plot_total_output_histogram.png'
    )
    
    print("================")
    print("Done!")
    print("================")


if __name__ == "__main__":
    main()

