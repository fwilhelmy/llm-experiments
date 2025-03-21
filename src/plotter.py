import os
import numpy as np
import matplotlib.pyplot as plt

def convert_value(val):
    """Convert a tensor to a float if necessary."""
    return val.item() if hasattr(val, "item") else val

def compute_mean_std(y_values_runs):
    """
    Compute mean and standard deviation across runs at each x-axis point.
    
    Parameters:
      y_values_runs: list of lists
         Each inner list contains metric values for a run.
         
    Returns:
      means: list of mean values per x-axis index.
      stds: list of standard deviation values per x-axis index.
    """
    n_points = len(y_values_runs[0])
    means, stds = [], []
    for i in range(n_points):
        values = [convert_value(run[i]) for run in y_values_runs]
        means.append(np.mean(values))
        stds.append(np.std(values))
    return means, stds

def get_metric_data(y_values_runs, steps_runs, mode, seed_index=0):
    """
    Extract the x-axis and y-axis data based on the chosen mode.
    
    Parameters:
      y_values_runs: list of lists
         Metric values for each run.
      steps_runs: list of lists
         X-axis values (steps) for each run.
      mode: str
         One of "average", "specific", or "std".
      seed_index: int
         Index of the run to use in "specific" mode.
    
    Returns:
      steps: list of x-axis values.
      curve: list of computed y-axis values (mean, specific, or mean for std mode).
      stds: list of standard deviations (only for "std" mode; otherwise None).
    """
    if mode in ["average", "std"]:
        if not steps_runs:
            return None, None, None
        # Assumes all runs share the same x-axis values.
        steps = steps_runs[0]
        means, stds = compute_mean_std(y_values_runs)
        return steps, means, stds
    elif mode == "specific":
        if seed_index < len(y_values_runs):
            steps = steps_runs[seed_index]
            specific_values = [convert_value(val) for val in y_values_runs[seed_index]]
            return steps, specific_values, None
        else:
            return None, None, None
    else:
        raise ValueError("Mode must be 'average', 'specific', or 'std'.")

def plot_all(data, save_path, mode="average", seed_index=0):
    """
    Plot 4 metrics individually for multiple configurations.
    The 4 metrics are:
      - Train Loss
      - Train Accuracy
      - Eval (Test) Loss
      - Eval (Test) Accuracy
    
    Each figure compares the corresponding metric for all configurations.
    
    Parameters:
      data: dict
         New data structure where each key is a configuration name and the value is a dict containing:
            - 'train': {'loss': [...], 'accuracy': [...]}
            - 'test': {'loss': [...], 'accuracy': [...]}
            - 'all_steps': [...]
      save_path: str
         Directory to save the plots.
      mode: str, optional
         "average" (default) to average across runs,
         "specific" to use a specific run (see seed_index),
         or "std" to show the mean with an error band (mean ± std).
      seed_index: int, optional
         When mode is "specific", the run index to display.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Define the 4 metric plots: train loss, train accuracy, eval loss, eval accuracy.
    for data_type, metric in [('train', 'loss'), ('train', 'accuracy'),
                              ('test', 'loss'), ('test', 'accuracy')]:
        plt.figure()
        for config_name, metrics in data.items():
            steps_runs = metrics['all_steps']
            y_values_runs = metrics[data_type][metric]
            steps, curve, stds = get_metric_data(y_values_runs, steps_runs, mode, seed_index)
            if steps is None or curve is None:
                print(f"Warning: Data not available for configuration '{config_name}' for {data_type} {metric} in mode '{mode}'.")
                continue
            plt.plot(steps, curve, label=config_name)
            if mode == "std" and stds is not None:
                plt.fill_between(steps,
                                 np.array(curve) - np.array(stds),
                                 np.array(curve) + np.array(stds),
                                 alpha=0.2)
        plt.xlabel("Steps")
        plt.title(f"{data_type.capitalize()} {metric.capitalize()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(save_path, f"{data_type}_{metric}.png")
        plt.savefig(fig_path)
        plt.close()

def plot_configuration(metrics, save_path, mode="average", seed_index=0):
    """
    For a single configuration, plot the combined metrics.
    
    This function produces two plots:
      - One for loss (with the train line in blue and the eval line in red)
      - One for accuracy (with the train line in blue and the eval line in red)
    
    Parameters:
      metrics: dict
         The metrics for the configuration
      config_name: str
         The configuration to plot.
      save_path: str
         Directory to save the plots.
      mode: str, optional
         "average" (default) to average across runs,
         "specific" to use a specific run (see seed_index),
         or "std" to show the mean with an error band.
      seed_index: int, optional
         When mode is "specific", the run index to display.
    """
    
    steps_runs = metrics['all_steps']
    
    for metric in ['loss', 'accuracy']:
        plt.figure()
        # Retrieve train data.
        steps_train, curve_train, stds_train = get_metric_data(metrics['train'][metric], steps_runs, mode, seed_index)
        # Retrieve eval data (using 'test').
        steps_eval, curve_eval, stds_eval = get_metric_data(metrics['test'][metric], steps_runs, mode, seed_index)
        
        if (steps_train is None or curve_train is None or
            steps_eval is None or curve_eval is None):
            print(f"Warning: Data not available for metric '{metric}' in mode '{mode}'.")
            continue
        
        # Plot train (blue) and eval (red).
        plt.plot(steps_train, curve_train, color='blue', label="Train")
        plt.plot(steps_eval, curve_eval, color='red', label="Eval")
        if mode == "std":
            plt.fill_between(steps_train,
                             np.array(curve_train) - np.array(stds_train),
                             np.array(curve_train) + np.array(stds_train),
                             color='blue', alpha=0.2)
            plt.fill_between(steps_eval,
                             np.array(curve_eval) - np.array(stds_eval),
                             np.array(curve_eval) + np.array(stds_eval),
                             color='red', alpha=0.2)
        plt.xlabel("Steps")
        plt.title(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(save_path, f"{metric}.png")
        plt.savefig(fig_path)
        plt.close()

# Example usage:
# To compare multiple configurations by plotting 4 individual figures:
# plot_all(new_data, "./plots", mode="average")
#
# To plot combined (loss and accuracy) metrics for a specific configuration:
# plot_configuration(new_data, "lstm", "./plots", mode="std")
