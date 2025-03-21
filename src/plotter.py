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

def plot_individual_metric(new_data, save_path, data_type, metric, mode, seed_index):
    """
    Generate an individual plot for a given data type and metric.
    
    Parameters:
      new_data: dict containing experiments.
      save_path: directory to save the plot.
      data_type: e.g. 'train' or 'test'
      metric: e.g. 'loss' or 'accuracy'
      mode: plot mode ("average", "specific", or "std")
      seed_index: run index when mode is "specific"
    """
    plt.figure()
    for exp_name, metrics in new_data.items():
        steps_runs = metrics['all_steps']
        y_values_runs = metrics[data_type][metric]
        
        steps, curve, stds = get_metric_data(y_values_runs, steps_runs, mode, seed_index)
        if steps is None or curve is None:
            print(f"Warning: Data not available for experiment '{exp_name}' for {data_type} {metric} in mode '{mode}'.")
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
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(save_path, f"{data_type}_{metric}.png")
    plt.savefig(fig_path)
    plt.close()

def plot_combined_data_type(new_data, save_path, data_type, mode, seed_index):
    """
    Generate a combined plot for a given data type (training or validation) that shows both loss and accuracy curves.
    
    Parameters:
      new_data: dict containing experiments.
      save_path: directory to save the plot.
      data_type: 'train' or 'test' (where 'test' represents validation)
      mode: plot mode ("average", "specific", or "std")
      seed_index: run index when mode is "specific"
    """
    plt.figure()
    for exp_name, metrics in new_data.items():
        steps_runs = metrics['all_steps']
        # Get loss data.
        steps_loss, curve_loss, stds_loss = get_metric_data(metrics[data_type]['loss'],
                                                            steps_runs, mode, seed_index)
        # Get accuracy data.
        steps_acc, curve_acc, stds_acc = get_metric_data(metrics[data_type]['accuracy'],
                                                         steps_runs, mode, seed_index)
        if (steps_loss is None or curve_loss is None or
            steps_acc is None or curve_acc is None):
            print(f"Warning: Data not available for experiment '{exp_name}' for {data_type} metrics in mode '{mode}'.")
            continue
        
        # Assumes steps_loss and steps_acc are identical.
        plt.plot(steps_loss, curve_loss, label=f"{exp_name} loss")
        plt.plot(steps_acc, curve_acc, label=f"{exp_name} accuracy")
        
        if mode == "std":
            plt.fill_between(steps_loss,
                             np.array(curve_loss) - np.array(stds_loss),
                             np.array(curve_loss) + np.array(stds_loss),
                             alpha=0.2)
            plt.fill_between(steps_acc,
                             np.array(curve_acc) - np.array(stds_acc),
                             np.array(curve_acc) + np.array(stds_acc),
                             alpha=0.2)
    
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title(f"Combined {data_type.capitalize()} Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(save_path, f"combined_{data_type}.png")
    plt.savefig(fig_path)
    plt.close()

def generate_plots(data, save_path, mode="average", seed_index=0, combined_plots=False):
    """
    Generate plots from the new data structure.
    
    Modes:
      - "average": plots the average across runs.
      - "specific": plots a specific run (based on seed_index).
      - "std": plots the mean curve and fills an error band (mean ± std).
    
    If combined_plots is True, generates 2 plots:
      one for the training set and one for the validation set,
      each showing both loss and accuracy curves.
    Otherwise, generates 4 individual plots:
      train loss, train accuracy, validation loss, and validation accuracy.
    
    Parameters:
      data: dict containing experiments.
      save_path: directory where the generated plots will be saved.
      mode: plot mode ("average", "specific", or "std")
      seed_index: run index when mode is "specific"
      combined_plots: bool indicating whether to combine loss and accuracy for each data type.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if combined_plots:
        # Generate one plot for training and one for validation.
        for data_type in ['train', 'test']:
            plot_combined_data_type(data, save_path, data_type, mode, seed_index)
    else:
        # Generate individual plots.
        for data_type, metric in [('train', 'loss'), ('train', 'accuracy'),
                                  ('test', 'loss'), ('test', 'accuracy')]:
            plot_individual_metric(data, save_path, data_type, metric, mode, seed_index)

# Example usage:
# For 4 separate plots:
# generate_plots(new_data, "./plots", mode="average", combined_plots=False)
# For 2 combined plots (one for training and one for validation):
# generate_plots(new_data, "./plots", mode="std", combined_plots=True)
