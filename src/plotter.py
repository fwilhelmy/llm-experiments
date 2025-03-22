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
         One of "mean", "specific", or "std".
      seed_index: int
         Index of the run to use in "specific" mode.
    
    Returns:
      steps: list of x-axis values.
      curve: list of computed y-axis values (mean, specific, or mean for std mode).
      stds: list of standard deviations (only for "std" mode; otherwise None).
    """
    if mode in ["mean", "std"]:
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
        raise ValueError("Mode must be 'mean', 'specific', or 'std'.")

def plot_all(data, save_path, mode="mean", seed_index=0):
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
         "mean" (default) to mean across runs,
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

def plot_configuration(metrics, save_path, mode="mean", seed_index=0):
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
         "mean" (default) to mean across runs,
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

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(metrics, axis_labels, key_metric, mode="mean", seed_index=0,
                 file_name="metrics", save_path=None, figsize=(8, 6), fontsize=12, show=True):
    """
    Plot a given metric performance across different configurations.
    
    Parameters:
      metrics: dict
          Dictionary with the structure:
              metrics[configuration_name][x_axis][run_id][metric_key]
          Example:
              {
                'lstm': {
                    0.1: [
                          {'total_elapsed': 2.18, 'step_time_avg': 0.0436},
                          {'total_elapsed': 2.10, 'step_time_avg': 0.0420}
                         ],
                    0.2: [ ... ]
                }
              }
      axis_labels: tuple
          A tuple (x_label, y_label) for the plot.
      key_metric: str
          The metric key to extract (e.g., "total_elapsed").
      mode: str, optional
          One of "mean", "std", or "specific". 
            - "mean": mean the metric over runs.
            - "std": mean with error band (mean ± std).
            - "specific": use the run at index seed_index.
      seed_index: int, optional
          When mode is "specific", the run index to select.
      file_name: str
          The base name of the file to save the plot (e.g., "metrics" will save "metrics.png").
      save_path: str, optional
          Directory in which to save the plot.
      figsize: tuple, optional
          Figure size.
      fontsize: int, optional
          Font size for labels and title.
      show: bool, optional
          If True, the plot is displayed; otherwise, it is closed.
    """
    plt.figure(figsize=figsize)
    
    # Iterate over each configuration.
    for config, config_data in metrics.items():
        x_values = []
        y_values = []
        std_values = []  # Only used if mode=="std"
        
        # Iterate over each x-axis value in this configuration.
        for x_val, runs in config_data.items():
            if not runs:
                continue
            
            # Extract values for key_metric from each run (if available).
            run_values = [run[key_metric] for run in runs if key_metric in run]
            if not run_values:
                continue
            
            if mode in ["mean", "std"]:
                avg_val = np.mean(run_values)
                x_values.append(x_val)
                y_values.append(avg_val)
                if mode == "std":
                    std_values.append(np.std(run_values))
            elif mode == "specific":
                if seed_index < len(run_values):
                    specific_val = run_values[seed_index]
                    x_values.append(x_val)
                    y_values.append(specific_val)
                else:
                    # Skip this x_val if the specific seed does not exist.
                    continue
            else:
                raise ValueError("Mode must be one of 'mean', 'std', or 'specific'.")
        
        # Sort data points by x-axis value.
        if x_values:
            sorted_indices = np.argsort(x_values)
            x_sorted = np.array(x_values)[sorted_indices]
            y_sorted = np.array(y_values)[sorted_indices]
            if mode == "std":
                std_sorted = np.array(std_values)[sorted_indices]
            
            plt.plot(x_sorted, y_sorted, marker='o', label=config)
            if mode == "std":
                plt.fill_between(x_sorted,
                                 y_sorted - std_sorted,
                                 y_sorted + std_sorted,
                                 alpha=0.2)
        else:
            print(f"Warning: No data available for configuration '{config}' for metric '{key_metric}'.")
    
    if axis_labels[0]:
        plt.xlabel(axis_labels[0], fontsize=fontsize)
    if axis_labels[1]:
        plt.ylabel(axis_labels[1], fontsize=fontsize)
    
    plt.title(file_name.capitalize(), fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(file_path)
    
    if show:
        plt.show()
    else:
        plt.close()

