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
        steps = steps_runs[0]  # Assumes all runs share the same x-axis values.
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

def generate_plots(new_data, save_path, mode="average", seed_index=0):
    """
    Generate plots from the new data structure with three modes:
      - "average": plots the average across runs.
      - "specific": plots a specific run (based on seed_index).
      - "std": plots the mean curve and fills an error band (mean ± std).
    
    Parameters:
      new_data: dict
         New data structure containing experiments. Each key is an experiment name (e.g., 'lstm', 'gpt'),
         with a tuple value consisting of:
           1. A list of lists of file paths (unused for plotting).
           2. A dictionary with metrics, which must include:
              - 'train': a dict with keys 'loss' and 'accuracy', each a list (one per run)
              - 'test': a dict with keys 'loss' and 'accuracy', each a list (one per run)
              - 'all_steps': a list of lists of x-axis values (one per run)
           3. A list of log directories (one per run) used for labeling.
      
      save_path: str
         Directory where the generated plot images will be saved.
      
      mode: str, optional
         "average" (default) to average across runs,
         "specific" to show a specific run (see seed_index),
         or "std" to plot mean and standard deviation (error band) across runs.
      
      seed_index: int, optional
         When mode is "specific", the run index to display (default is 0).
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Define the metric pairs to plot: (data_type, metric)
    metric_pairs = [('train', 'loss'), ('train', 'accuracy'),
                    ('test', 'loss'), ('test', 'accuracy')]
    
    for data_type, metric in metric_pairs:
        plt.figure()
        for exp_name, exp_val in new_data.items():
            metrics = exp_val
            steps_runs = metrics['all_steps']
            y_values_runs = metrics[data_type][metric]
            
            steps, curve, stds = get_metric_data(y_values_runs, steps_runs, mode, seed_index)
            if steps is None or curve is None:
                print(f"Warning: Data not available for experiment '{exp_name}' in mode '{mode}'.")
                continue
                
            plt.plot(steps, curve, label=exp_name)
            if mode == "std" and stds is not None:
                plt.fill_between(steps,
                                 np.array(curve) - np.array(stds),
                                 np.array(curve) + np.array(stds),
                                 alpha=0.2)
        
        plt.xlabel("Steps")
        plt.ylabel(metric.capitalize())
        plt.title(f"{data_type.capitalize()} {metric.capitalize()}")
        plt.legend()
        plt.grid(True)  # Add grid lines to the plot.
        plt.tight_layout()
        fig_path = os.path.join(save_path, f"{data_type}_{metric}.png")
        plt.savefig(fig_path)
        plt.close()

# Example usage:
# generate_plots(new_data, "./plots", mode="average")
# generate_plots(new_data, "./plots", mode="specific", seed_index=0)
# generate_plots(new_data, "./plots", mode="std")
