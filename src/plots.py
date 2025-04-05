import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from utils import slice_by_steps
import torch

# -----------------------------
# Helper Data Processing Functions
# -----------------------------
def compute_plot_data(metrics, merge_ops=True):
    """
    Compute the aggregated curve and standard deviation for a metric.
    
    Parameters:
      metrics : NumPy array of shape (n_runs, n_op_orders, n_evals)
      merge_ops : bool, if True average over both runs and op orders;
                  if False, average over runs only (keeping op order curves separate).
                  
    Returns:
      curve : If merge_ops True, a 1D array of length n_evals.
              If merge_ops False, a 2D array (n_op_orders, n_evals).
      stds : Same shape as curve.
    """
    if merge_ops:
        curve = np.mean(metrics, axis=(0, 1))
        stds = np.std(metrics, axis=(0, 1))
    else:
        curve = np.mean(metrics, axis=0)  # Average over runs only.
        stds = np.std(metrics, axis=0)
    return curve, stds

def find_best_metric(metrics, optimum="min"):
    """
    Find the best (optimal) metric value.
    
    Parameters:
      metrics : NumPy array of shape (n_runs, n_op_orders, n_evals)
      optimum : str, "min" for loss or "max" for accuracy.
      
    Returns:
        best_idx: index along the evaluation axis.
        best_value: the optimal value.
        best_std: the std at that index.
    """
    curve, stds = compute_plot_data(metrics, merge_ops=True)
    optimum_fn = np.argmin if optimum == "min" else np.argmax
    best_idx = optimum_fn(curve)
    return best_idx, curve[best_idx], stds[best_idx]

def extract_config_best_metrics(all_metrics, metrics_to_extract=['loss', 'accuracy'], compute_std=True):
    """
    Extract best metrics (and the step at which they occur) for each configuration.
    
    Parameters:
      all_metrics : dict with configuration names as keys. Each value should contain:
                    - "train": {"loss": array, "accuracy": array}
                    - "test": {"loss": array, "accuracy": array}
                    - "steps": 1D array of evaluation steps.
      metrics_to_extract : list of metric names to extract (default: ['loss', 'accuracy']).
      compute_std : bool, if True, also include the standard deviation with the best value and the step.
      
    Returns:
      best_metrics : dict structured as:
         {
           "train": {"loss": {"value": [...], "step": [...], "std": [...] (if stds True)},
                     "accuracy": {"value": [...], "step": [...], "std": [...] (if stds True)},
           "test":  { ... }
         }
    """
    best_metrics = {}
    for mode in ['train', 'test']:
        best_metrics[mode] = {}
        for metric in metrics_to_extract:
            best_metrics[mode][metric] = {'value': [], 'step': []}
            if compute_std:
                best_metrics[mode][metric]['std'] = []
    
    for config in all_metrics.values():
        for mode in ['train', 'test']:
            for metric in metrics_to_extract:
                optimum = 'min' if metric == 'loss' else 'max'
                best_idx, best_val, best_std = find_best_metric(config[mode][metric], optimum=optimum)
                best_metrics[mode][metric]['value'].append(best_val)
                step = config['steps'][best_idx]
                best_metrics[mode][metric]['step'].append(step)
                if compute_std:
                    best_metrics[mode][metric]['std'].append(best_std)
    return best_metrics

# -----------------------------
# Standardized Plotting Functions
# -----------------------------

def plot_curve(ax, y_data, x_axis, label=None, color=None, merge_ops=True, **kwargs):
    """
    Compute aggregated plot data and then plot the curve and fill the std band.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        y_data (np.ndarray): Array with shape (n_runs, n_ops, n_evals).
        x_axis (np.ndarray): 1D array of x-axis values.
        label (str, optional): Label for the curve.
        color (str or tuple, optional): Color for the line and fill.
        merge_ops (bool, optional): Whether to average over both runs and op orders.
        **kwargs: Additional keyword arguments for ax.plot.
        
    Returns:
        matplotlib.lines.Line2D: The line object.
    """
    curve, std = compute_plot_data(y_data, merge_ops=merge_ops)
    line = ax.plot(x_axis, curve, label=label, color=color, **kwargs)[0]
    ax.fill_between(x_axis, curve - std, curve + std, color=color, alpha=0.2)
    return line
    """
    Plot an aggregated curve on a provided axis.
    
    This function computes the aggregated curve and standard deviation
    using compute_plot_data, then plots the curve and fills the area between
    (mean - std) and (mean + std).
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis on which to plot the curve.
        y_data (np.ndarray): Array of shape (n_runs, n_ops, n_evals) with y-axis values.
        x_axis (np.ndarray): 1D array of x-axis values.
        label (str, optional): Label for the curve.
        color (str or tuple, optional): Color for the line and fill.
        **kwargs: Additional keyword arguments for ax.plot.
    
    Returns:
        matplotlib.lines.Line2D: The line object created by the plot.
    """
    # Compute aggregated curve and standard deviation.
    curve, std = compute_plot_data(y_data, merge_ops=True)
    
    # Plot the mean curve.
    line = ax.plot(x_axis, curve, label=label, color=color, **kwargs)[0]
    
    # Fill the area between (curve - std) and (curve + std).
    ax.fill_between(x_axis, curve - std, curve + std, color=color, alpha=0.2)
    
    return line

import os
import matplotlib.pyplot as plt

def plot_config_loss_accs(data, save_directory, file_name=None,
                          show_std=True, x_ticks=None, loss_y_scale=False):
    """
    Plot metrics for a single configuration, stacking loss above accuracy in one figure.
    """
    os.makedirs(save_directory, exist_ok=True)
    steps = data['steps']
    
    # Create a figure with two vertically-stacked subplots sharing the x-axis.
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 10))
    
    # Define the order in which to plot: loss on top, accuracy on bottom.
    metrics = ['loss', 'accuracy']
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        # Compute the train and test curves along with standard deviations.
        curve_train, stds_train = compute_plot_data(data['train'][metric])
        curve_test, stds_test = compute_plot_data(data['test'][metric])
        
        # Plot the training and test curves.
        ax.plot(steps, curve_train, label="Train", color='blue')
        ax.plot(steps, curve_test, label="Test", color='red')
        
        # Plot the standard deviation as a shaded area if available.
        if show_std and stds_train is not None and stds_test is not None:
            ax.fill_between(steps, curve_train - stds_train, curve_train + stds_train,
                            color='blue', alpha=0.2)
            ax.fill_between(steps, curve_test - stds_test, curve_test + stds_test,
                            color='red', alpha=0.2)
        
        # Set the y-axis label.
        ax.set_ylabel(metric.capitalize())
        
        # Optionally set a logarithmic scale for loss.
        if loss_y_scale and metric == "loss":
            ax.set_yscale("log")
        
        ax.legend()
        ax.grid(True)
    
    # Set the shared x-axis label and adjust x-ticks if provided.
    axs[-1].set_xlabel("Steps")
    if x_ticks is not None:
        axs[-1].set_xticks(x_ticks)
    
    plt.tight_layout()
    
    out_file = f"_{file_name}" if file_name else ""
    plt.savefig(os.path.join(save_directory, f"loss_acc{out_file}.png"))
    plt.close()

def plot_config_ops(data, save_directory, file_name="operation_orders",
                    show_std=True, train_x_scale=False, x_ticks=None,
                    loss_y_scale=False):
    """
    Plot metrics for a single configuration, showing each operation order individually.
    """
    os.makedirs(save_directory, exist_ok=True)
    steps = data['steps']
    op_labels = data['operation_orders']
    
    # Compute curves per metric (each as 2D array: n_op_orders x n_evals)
    train_loss, std_train_loss = compute_plot_data(data['train']['loss'], merge_ops=False)
    test_loss, std_test_loss = compute_plot_data(data['test']['loss'], merge_ops=False)
    train_acc, std_train_acc = compute_plot_data(data['train']['accuracy'], merge_ops=False)
    test_acc, std_test_acc = compute_plot_data(data['test']['accuracy'], merge_ops=False)
    
    # Create a 2x2 grid with shared x-axis (only bottom row gets xlabel) and shared y-axis per row.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')
    
    # --- Top Left: Train Loss ---
    for i, op in enumerate(op_labels):
        axs[0, 0].plot(steps, train_loss[i], label=op)
        if show_std and std_train_loss is not None:
            axs[0, 0].fill_between(steps, train_loss[i] - std_train_loss[i],
                                   train_loss[i] + std_train_loss[i], alpha=0.2)
    if loss_y_scale:
        axs[0, 0].set_yscale("log")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_ylabel("Loss")
    
    # --- Top Right: Test Loss ---
    for i, op in enumerate(op_labels):
        axs[0, 1].plot(steps, test_loss[i], label=op)
        if show_std and std_test_loss is not None:
            axs[0, 1].fill_between(steps, test_loss[i] - std_test_loss[i],
                                   test_loss[i] + std_test_loss[i], alpha=0.2)
    if loss_y_scale:
        axs[0, 1].set_yscale("log")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # --- Bottom Left: Train Accuracy ---
    for i, op in enumerate(op_labels):
        axs[1, 0].plot(steps, train_acc[i], label=op)
        if show_std and std_train_acc is not None:
            axs[1, 0].fill_between(steps, train_acc[i] - std_train_acc[i],
                                   train_acc[i] + std_train_acc[i], alpha=0.2)
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].set_xlabel("Steps")
    
    # --- Bottom Right: Test Accuracy ---
    for i, op in enumerate(op_labels):
        axs[1, 1].plot(steps, test_acc[i], label=op)
        if show_std and std_test_acc is not None:
            axs[1, 1].fill_between(steps, test_acc[i] - std_test_acc[i],
                                   test_acc[i] + std_test_acc[i], alpha=0.2)
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel("Steps")

    axs[0, 0].set_title("Train", fontweight='bold')
    axs[0, 1].set_title("Test", fontweight='bold')
    
    if train_x_scale:
        for ax in [axs[0, 0], axs[1, 0]]:
            ax.set_xscale('log')
    
    if x_ticks is not None:
        for ax in axs.flat:
            ax.set_xticks(x_ticks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"{file_name}.png"))
    plt.close()

def plot_config_ops2(data, save_directory, file_name="ops2", x_ticks=None, loss_y_scale=False):
    """
    Plot per-operation-order metrics in a compact figure.
    
    Instead of the original approach, this function creates a figure with 2 rows and one
    column per operation order. For each op, it plots the train and test curves for loss (top)
    and accuracy (bottom) using the same plotting logic as in plot_config_loss_accs.
    
    Parameters:
        data (dict): A dictionary containing:
                     - 'steps': 1D array of evaluation steps.
                     - 'train': dict with keys 'loss' and 'accuracy', each with an array 
                                of shape (n_runs, n_ops, n_evals).
                     - 'test':  dict with keys 'loss' and 'accuracy', each with an array 
                                of shape (n_runs, n_ops, n_evals).
                     - 'operation_orders': list of labels for each op order.
        save_directory (str): Directory to save the resulting plot.
        file_name (str, optional): Base filename for saving the figure.
        x_ticks (list, optional): Custom x-axis tick values.
        loss_y_scale (bool, optional): If True, set the loss subplots to a logarithmic y-scale.
    """
    os.makedirs(save_directory, exist_ok=True)
    steps = data['steps']
    op_labels = ["Binary", "Ternary"]
    n_ops = len(op_labels)
    
    # Create a figure with 2 rows (loss and accuracy) and n_ops columns (one per operation order)
    fig, axs = plt.subplots(2, n_ops, figsize=(12, 10), sharex='col', sharey='row')
    
    # Ensure axs is 2D (if n_ops == 1, axs might be 1D)
    if n_ops == 1:
        axs = axs.reshape(2, 1)
    
    # Define colors for the two modes.
    colors = {'train': 'blue', 'test': 'red'}
    
    # Loop over each operation order
    for i in range(n_ops):
        ax_loss = axs[0, i]
        ax_acc  = axs[1, i]
        
        # Plot loss and accuracy curves for each mode.
        for mode in ['train', 'test']:
            y_loss = data[mode]['loss'][:, i:i+1, :]
            plot_curve(ax_loss, y_loss, steps, label=mode.capitalize(), color=colors[mode])
            y_acc = data[mode]['accuracy'][:, i:i+1, :]
            plot_curve(ax_acc, y_acc, steps, label=mode.capitalize(), color=colors[mode])
        
        # For the top (loss) subplot: add title only.
        ax_loss.set_title(f"{op_labels[i]} Operation", fontweight='bold')
        if loss_y_scale:
            ax_loss.set_yscale("log")
        if x_ticks is not None:
            ax_loss.set_xticks(x_ticks)
        ax_loss.legend()
        ax_loss.grid(True)
        
        # For the bottom (accuracy) subplot: add x-label.
        if x_ticks is not None:
            ax_acc.set_xticks(x_ticks)
        ax_acc.set_xlabel("Steps")
        ax_acc.legend()
        ax_acc.grid(True)
        
        # Set y-label only on left subplots.
        if i == 0:
            ax_loss.set_ylabel("Loss")
            ax_acc.set_ylabel("Accuracy")
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_directory, f"{file_name}.png"))
    plt.close(fig)

def plot_all_configs_metrics2(data, save_directory, file_name="all_configs",
                             show_std=True, merge_ops=True,
                             figures=[[('train','loss'), ('test','loss')],
                                      [('train','accuracy'), ('test','accuracy')]],
                             axis_labels=[["Train", "Test"], ["Loss", "Accuracy"]],
                             train_x_scale=False, loss_y_scale=False, x_ticks=None):
    """
    Plot grouped metrics for all configurations according to a custom layout.
    
    Parameters:
      data : dict with configuration names as keys. Each configuration should include:
             - "steps": 1D array of evaluation steps.
             - "train": dict with keys (e.g. "loss", "accuracy") whose values are arrays of shape (n_runs, n_op_orders, n_evals).
             - "test": dict with similar structure.
             - "label": optional label for the configuration.
      save_directory : str, directory to save the plot.
      file_name : str, output file name.
      show_std : bool, if True, show standard deviation as an error band.
      merge_ops : bool, if True, average over both runs and op orders; if False, keep op order curves separate.
      figures : nested list of tuples (mode, metric) specifying which subplot to compute.
                For example, [[('train','loss'), ('test','loss')],
                              [('train','accuracy'), ('test','accuracy')]].
      axis_labels : nested list (n_rows x n_cols) with labels:
                    - The first inner list gives the column headers (displayed as subplot titles on the top row).
                    - The second inner list gives the row labels (displayed on the y-axis of the leftmost subplots).
                    Default: [["Train", "Test"], ["Loss", "Accuracy"]].
      train_x_scale : bool, if True, set x-axis to log scale for subplots where mode is 'train'.
      loss_y_scale : bool, if True, set y-axis to log scale for subplots where metric is 'loss'.
      x_ticks : list or None, optional custom x-tick values.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Determine grid dimensions from the 'figures' parameter.
    n_rows = len(figures)
    n_cols = len(figures[0]) if n_rows > 0 else 0
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex='col', sharey='row')
    # Ensure axs is 2D.
    if n_rows == 1:
        axs = np.array([axs])
    if n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Loop over each configuration and plot its curves on each subplot.
    for config_name, config_metrics in data.items():
        steps = config_metrics['steps']
        label = config_metrics.get('label', config_name)
        for i in range(n_rows):
            for j in range(n_cols):
                mode, metric = figures[i][j]
                curve, stds = compute_plot_data(config_metrics[mode][metric], merge_ops=merge_ops)
                axs[i, j].plot(steps, curve, label=label)
                if show_std and stds is not None:
                    axs[i, j].fill_between(steps, curve - stds, curve + stds, alpha=0.2)
    
    # Format each subplot.
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            mode, metric = figures[i][j]
            if mode == 'train' and train_x_scale:
                ax.set_xscale("log")
            if metric == 'loss' and loss_y_scale:
                ax.set_yscale("log")
            # Set column header from axis_labels (top row).
            if i == 0:
                ax.set_title(axis_labels[0][j], fontweight='bold')
            # Set row label from axis_labels (left column).
            if j == 0:
                ax.set_ylabel(axis_labels[1][i])
            # Set x-label on bottom row.
            if i == n_rows - 1:
                ax.set_xlabel("Steps")
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            ax.legend()
            ax.grid(True)

    plt.get_current_fig_manager().window.state('zoomed')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"{file_name}.png"))
    plt.close()

def plot_all_configs_metrics_colobar(data, save_directory, file_name="all_configs",
                             show_std=True, colorbar=False, colorbar_scale=False, colorbar_title="Configuration",
                             figures=[[('train','loss'), ('test','loss')],
                                      [('train','accuracy'), ('test','accuracy')]],
                             axis_labels=[["Train", "Test"], ["Loss", "Accuracy"]],
                             train_x_scale=False, loss_y_scale=False, x_ticks=None):
    """
    Plot grouped metrics for all configurations according to a custom layout.
    The colormap and colorbar use either a log2 mapping or a discrete (raw) mapping
    based on the flag `colorbar_scale`.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    os.makedirs(save_directory, exist_ok=True)
    
    # Determine grid dimensions from the 'figures' parameter.
    n_rows = len(figures)
    n_cols = len(figures[0]) if n_rows > 0 else 0
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex='col', sharey='row')
    if n_rows == 1:
        axs = np.array([axs])
    if n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Get sorted configuration keys.
    configs_sorted = list(data.keys())
    
    # Set up normalization mapping based on the desired colorbar_scale mode.
    if colorbar_scale:
        # Use log2 values for mapping.
        config_numeric_values = [data[config]['label_value'] for config in configs_sorted]
        log_config_values = [np.log2(val) for val in config_numeric_values]
        min_log = min(log_config_values)
        max_log = max(log_config_values)
        config2norm = {
            config: (np.log2(data[config]['label_value']) - min_log) / (max_log - min_log) if max_log != min_log else 0.5
            for config in configs_sorted
        }
    else:
        # Use raw configuration values for mapping.
        config_numeric_values = [data[config]['label_value'] for config in configs_sorted]
        min_raw = min(config_numeric_values)
        max_raw = max(config_numeric_values)
        config2norm = {
            config: (data[config]['label_value'] - min_raw) / (max_raw - min_raw) if max_raw != min_raw else 0.5
            for config in configs_sorted
        }
    
    # Select the colormap.
    cmap = plt.cm.viridis
    
    # Loop over each configuration and plot its curves on each subplot.
    for config_name in configs_sorted:
        config_metrics = data[config_name]
        # Determine color based on the chosen normalization.
        color_val = config2norm[config_name]
        color = cmap(color_val)
        
        steps = config_metrics['steps']
        label = config_metrics.get('label', config_name)
        
        for i in range(n_rows):
            for j in range(n_cols):
                mode, metric = figures[i][j]
                # Assuming compute_plot_data is defined elsewhere.
                curve, stds = compute_plot_data(config_metrics[mode][metric], merge_ops=True)
                axs[i, j].plot(steps, curve, label=label, color=color)
                if show_std and stds is not None:
                    axs[i, j].fill_between(steps, curve - stds, curve + stds, color=color, alpha=0.2)
    
    # Format subplots.
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            mode, metric = figures[i][j]
            if mode == 'train' and train_x_scale:
                ax.set_xscale("log")
            if metric == 'accuracy' and mode == 'train':
                ax.legend()
            if metric == 'loss' and loss_y_scale:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(axis_labels[0][j].capitalize(), fontweight='bold')
            if j == 0:
                ax.set_ylabel(axis_labels[1][i].replace("_", " ").capitalize())
            if i == n_rows - 1:
                ax.set_xlabel("Steps")
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            ax.grid(True)
    
    plt.tight_layout()
    
    # Add a colorbar if requested.
    if colorbar:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.93, 0.07, 0.02, 0.85])
        
        if colorbar_scale:
            # Colorbar on a log2 scale.
            colorbar_norm = Normalize(vmin=min_log, vmax=max_log)
            sm = ScalarMappable(cmap=cmap, norm=colorbar_norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, label=colorbar_title)
        else:
            # Colorbar with discrete (raw) values.
            sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
            sm.set_array([])
            ticks = [config2norm[config] for config in configs_sorted]
            tick_labels = [str(data[config]['label_value']) for config in configs_sorted]
            cbar = fig.colorbar(sm, cax=cbar_ax, label=colorbar_title, ticks=ticks)
            cbar.ax.set_yticklabels(tick_labels)
    
    plt.savefig(os.path.join(save_directory, f"{file_name}.png"))
    plt.close()


def plot_all_configs_metrics_colobar_scale(data, save_directory, file_name="all_configs",
                             show_std=True, colorbar=False, colorbar_scale=False, colorbar_title="Configuration",
                             figures=[[('train','loss'), ('test','loss')],
                                      [('train','accuracy'), ('test','accuracy')]],
                             axis_labels=[["Train", "Test"], ["Loss", "Accuracy"]],
                             train_x_scale=False, loss_y_scale=False, x_ticks=None, figsize=(12, 10)):
    """
    Plot grouped metrics for all configurations according to a custom layout.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    os.makedirs(save_directory, exist_ok=True)
    
    # Determine grid dimensions from the 'figures' parameter.
    n_rows = len(figures)
    n_cols = len(figures[0]) if n_rows > 0 else 0
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex='col', sharey='row')
    if n_rows == 1:
        axs = np.array([axs])
    if n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Get sorted configuration keys.
    configs_sorted = list(data.keys())
    n_configs = len(configs_sorted)
    
    # --- New normalization mapping based on the actual config numeric value ---
    # Extract numeric values from config labels
    config_numeric_values = []
    for config in configs_sorted:
        value = data[config]['label_value']
        config_numeric_values.append(value)
    
    # Compute log2 values for each configuration
    log_config_values = [np.log2(val) for val in config_numeric_values]
    min_log = min(log_config_values)
    max_log = max(log_config_values)
    
    # Create normalized mapping (between 0 and 1) using log2 values
    config2norm = {
        config: (np.log2(data[config]['label_value']) - min_log) / (max_log - min_log) if max_log != min_log else 0.5
        for config in configs_sorted
    }
    
    # Select colormap and prepare a ScalarMappable for plotting curves.
    cmap = plt.cm.viridis
    
    # Loop over each configuration and plot its curves on each subplot.
    for config_name in configs_sorted:
        config_metrics = data[config_name]
        # Determine color based on the normalized mapping.
        color_val = config2norm[config_name]
        color = cmap(color_val)
        
        steps = config_metrics['steps']
        label = config_metrics.get('label', config_name)
        
        for i in range(n_rows):
            for j in range(n_cols):
                mode, metric = figures[i][j]
                # Assuming compute_plot_data is defined elsewhere.
                curve, stds = compute_plot_data(config_metrics[mode][metric], merge_ops=True)
                axs[i, j].plot(steps, curve, label=label, color=color)
                if show_std and stds is not None:
                    axs[i, j].fill_between(steps, curve - stds, curve + stds, color=color, alpha=0.2)
    
    # Format subplots.
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            mode, metric = figures[i][j]
            if mode == 'train' and train_x_scale:
                ax.set_xscale("log")
            if metric == 'accuracy' and mode == 'train':
                ax.legend()
            if metric == 'loss' and loss_y_scale:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(axis_labels[0][j].capitalize(), fontweight='bold')
            if j == 0:
                ax.set_ylabel(axis_labels[1][i].replace("_", " ").capitalize())
            if i == n_rows - 1:
                ax.set_xlabel("Steps")
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            ax.grid(True)
    
    plt.tight_layout()
    
    # Add a colorbar if requested.
    if colorbar:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.93, 0.07, 0.02, 0.85])
        
        if colorbar_scale:
            # Log2 scale: colorbar shows log2 values.
            colorbar_norm = Normalize(vmin=min_log, vmax=max_log)
            sm = ScalarMappable(cmap=cmap, norm=colorbar_norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, label=colorbar_title)
        else:
            # Discrete scale: use normalized mapping from 0 to 1.
            sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
            sm.set_array([])
            # Use the normalized config values as tick positions.
            ticks = [config2norm[config] for config in configs_sorted]
            # Use the original configuration values as tick labels.
            tick_labels = [str(data[config]['label_value']) for config in configs_sorted]
            cbar = fig.colorbar(sm, cax=cbar_ax, label=colorbar_title, ticks=ticks)
            cbar.ax.set_yticklabels(tick_labels)
    
    plt.savefig(os.path.join(save_directory, f"{file_name}.png"))
    plt.close()

def plot_all_configs_best_metrics(data, save_directory, file_name="comparative_best",
                                  show_std=True, x_label_title="Configuration",
                                  x_ticks=None, loss_y_scale=False):
    """
    Plot comparative best metrics (Loss and Accuracy) across configurations.
    """
    os.makedirs(save_directory, exist_ok=True)
    x_labels = [data[config].get('label', config) for config in data.keys()]
    best_metrics = extract_config_best_metrics(data, compute_std=show_std)
    
    # ---- Loss Figure ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(x_labels, best_metrics['train']['loss']['value'], marker='o', label='Best Train Loss')
    ax1.plot(x_labels, best_metrics['test']['loss']['value'], marker='o', label='Best Test Loss')
    if show_std:
        ax1.fill_between(x_labels,
                         np.array(best_metrics['train']['loss']['value']) - np.array(best_metrics['train']['loss']['std']),
                         np.array(best_metrics['train']['loss']['value']) + np.array(best_metrics['train']['loss']['std']),
                         alpha=0.2)
        ax1.fill_between(x_labels,
                         np.array(best_metrics['test']['loss']['value']) - np.array(best_metrics['test']['loss']['std']),
                         np.array(best_metrics['test']['loss']['value']) + np.array(best_metrics['test']['loss']['std']),
                         alpha=0.2)
    if loss_y_scale:
        ax1.set_yscale('log')
    ax1.set_ylabel("Loss (log scale)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(x_labels, best_metrics['train']['loss']['step'], marker='o', label='Best Train Loss Step')
    ax2.plot(x_labels, best_metrics['test']['loss']['step'], marker='o', label='Best Test Loss Step')
    if show_std:
        ax2.fill_between(x_labels,
                         np.array(best_metrics['train']['loss']['step']) - np.array(best_metrics['train']['loss']['std']),
                         np.array(best_metrics['train']['loss']['step']) + np.array(best_metrics['train']['loss']['std']),
                         alpha=0.2)
        ax2.fill_between(x_labels,
                         np.array(best_metrics['test']['loss']['step']) - np.array(best_metrics['test']['loss']['std']),
                         np.array(best_metrics['test']['loss']['step']) + np.array(best_metrics['test']['loss']['std']),
                         alpha=0.2)
    ax2.set_ylabel("Step")
    ax2.set_xlabel(x_label_title)
    if x_ticks is not None:
        ax2.set_xticks(x_ticks)
    ax2.legend()
    ax2.grid(True)
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"{file_name}_loss.png"))
    plt.close()
    
    # ---- Accuracy Figure ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(x_labels, best_metrics['train']['accuracy']['value'], marker='o', label='Best Train Accuracy')
    ax1.plot(x_labels, best_metrics['test']['accuracy']['value'], marker='o', label='Best Test Accuracy')
    if show_std:
        ax1.fill_between(x_labels,
                         np.array(best_metrics['train']['accuracy']['value']) - np.array(best_metrics['train']['accuracy']['std']),
                         np.array(best_metrics['train']['accuracy']['value']) + np.array(best_metrics['train']['accuracy']['std']),
                         alpha=0.2)
        ax1.fill_between(x_labels,
                         np.array(best_metrics['test']['accuracy']['value']) - np.array(best_metrics['test']['accuracy']['std']),
                         np.array(best_metrics['test']['accuracy']['value']) + np.array(best_metrics['test']['accuracy']['std']),
                         alpha=0.2)
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(x_labels, best_metrics['train']['accuracy']['step'], marker='o', label='Best Train Accuracy Step')
    ax2.plot(x_labels, best_metrics['test']['accuracy']['step'], marker='o', label='Best Test Accuracy Step')
    if show_std:
        ax2.fill_between(x_labels,
                         np.array(best_metrics['train']['accuracy']['step']) - np.array(best_metrics['train']['accuracy']['std']),
                         np.array(best_metrics['train']['accuracy']['step']) + np.array(best_metrics['train']['accuracy']['std']),
                         alpha=0.2)
        ax2.fill_between(x_labels,
                         np.array(best_metrics['test']['accuracy']['step']) - np.array(best_metrics['test']['accuracy']['std']),
                         np.array(best_metrics['test']['accuracy']['step']) + np.array(best_metrics['test']['accuracy']['std']),
                         alpha=0.2)
    ax2.set_ylabel("Step")
    ax2.set_xlabel(x_label_title)
    if x_ticks is not None:
        ax2.set_xticks(x_ticks)
    ax2.legend()
    ax2.grid(True)
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"{file_name}_accuracy.png"))
    plt.close()

def plot_all_configs_best_metrics_histogram(data, save_directory, file_name="comparative_best_histogram",
                                             show_std=True, x_label_title="Configuration",
                                             x_ticks=None, loss_y_scale=False):
    """
    Plot comparative best metrics (Loss and Accuracy) across configurations as grouped bar charts.
    
    For each metric (loss and accuracy), two vertically stacked subplots are created:
      - The top subplot shows the best metric values.
      - The bottom subplot shows the corresponding best evaluation steps.
    
    Error bars represent the standard deviation (if show_std is True).
    
    Parameters:
      data : dict with configuration names as keys. Each configuration should include:
             - "steps": 1D array of evaluation steps.
             - "train": dict with keys (e.g., "loss", "accuracy") whose values are arrays of shape (n_runs, n_ops, n_evals).
             - "test":  dict with similar structure.
      save_directory : str, directory where the plots will be saved.
      file_name : str, base name for the output files.
      show_std : bool, if True, display error bars.
      x_label_title : str, label for the x-axis.
      x_ticks : list or None, custom x-axis tick values.
      loss_y_scale : bool, if True, use a logarithmic scale for the loss value plot.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    best_metrics = extract_config_best_metrics(data, compute_std=show_std)
    x_labels = [data[config].get('label', config) for config in data.keys()]
    indices = np.arange(len(x_labels))
    bar_width = 0.35  # Width for grouped bars.
    
    # --- Loss Figure ---
    fig, (ax_val, ax_step) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Retrieve best loss values and steps for train and test.
    train_loss = best_metrics['train']['loss']['value']
    test_loss  = best_metrics['test']['loss']['value']
    train_loss_std = best_metrics['train']['loss']['std'] if show_std else None
    test_loss_std  = best_metrics['test']['loss']['std'] if show_std else None
    
    train_loss_step = best_metrics['train']['loss']['step']
    test_loss_step  = best_metrics['test']['loss']['step']
    # We use the same std for steps for simplicity.
    train_loss_step_std = best_metrics['train']['loss']['std'] if show_std else None
    test_loss_step_std  = best_metrics['test']['loss']['std'] if show_std else None
    
    # Top subplot: Best Loss Values.
    ax_val.bar(indices - bar_width/2, train_loss, bar_width, yerr=train_loss_std,
               label="Train", color="blue")
    ax_val.bar(indices + bar_width/2, test_loss, bar_width, yerr=test_loss_std,
               label="Test", color="red")
    ax_val.set_ylabel("Best Loss")
    if loss_y_scale:
        ax_val.set_yscale("log")
    ax_val.legend()
    ax_val.grid(True, axis="y")
    
    # Bottom subplot: Best Steps for Loss.
    ax_step.bar(indices - bar_width/2, train_loss_step, bar_width, yerr=train_loss_step_std,
                label="Train", color="blue")
    ax_step.bar(indices + bar_width/2, test_loss_step, bar_width, yerr=test_loss_step_std,
                label="Test", color="red")
    ax_step.set_ylabel("Step of Best Loss")
    ax_step.set_xlabel(x_label_title)
    ax_step.legend()
    ax_step.grid(True, axis="y")
    
    ax_step.set_xticks(indices)
    ax_step.set_xticklabels(x_labels, rotation=45, ha="right")
    
    fig.tight_layout()
    loss_file = os.path.join(save_directory, f"{file_name}_loss.png")
    fig.savefig(loss_file)
    plt.close(fig)
    
    # --- Accuracy Figure ---
    fig, (ax_val, ax_step) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    train_acc = best_metrics['train']['accuracy']['value']
    test_acc  = best_metrics['test']['accuracy']['value']
    train_acc_std = best_metrics['train']['accuracy']['std'] if show_std else None
    test_acc_std  = best_metrics['test']['accuracy']['std'] if show_std else None
    
    train_acc_step = best_metrics['train']['accuracy']['step']
    test_acc_step  = best_metrics['test']['accuracy']['step']
    train_acc_step_std = best_metrics['train']['accuracy']['std'] if show_std else None
    test_acc_step_std  = best_metrics['test']['accuracy']['std'] if show_std else None
    
    # Top subplot: Best Accuracy Values.
    ax_val.bar(indices - bar_width/2, train_acc, bar_width, yerr=train_acc_std,
               label="Train", color="blue")
    ax_val.bar(indices + bar_width/2, test_acc, bar_width, yerr=test_acc_std,
               label="Test", color="red")
    ax_val.set_ylabel("Best Accuracy")
    ax_val.legend()
    ax_val.grid(True, axis="y")
    
    # Bottom subplot: Best Steps for Accuracy.
    ax_step.bar(indices - bar_width/2, train_acc_step, bar_width, yerr=train_acc_step_std,
                label="Train", color="blue")
    ax_step.bar(indices + bar_width/2, test_acc_step, bar_width, yerr=test_acc_step_std,
                label="Test", color="red")
    ax_step.set_ylabel("Step of Best Accuracy")
    ax_step.set_xlabel(x_label_title)
    ax_step.legend()
    ax_step.grid(True, axis="y")
    
    ax_step.set_xticks(indices)
    ax_step.set_xticklabels(x_labels, rotation=45, ha="right")
    
    fig.tight_layout()
    acc_file = os.path.join(save_directory, f"{file_name}_accuracy.png")
    fig.savefig(acc_file)
    plt.close(fig)
from matplotlib.colors import Normalize
import numpy as np

class LogNorm2(Normalize):
    """
    Normalize a given value to the 0-1 range on a log base 2 scale.
    """
    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Auto-scale if needed
        if self.vmin is None or self.vmax is None:
            self.autoscale(value)
        # Apply log base 2 transformation
        result = np.log2(value) - np.log2(self.vmin)
        result /= (np.log2(self.vmax) - np.log2(self.vmin))
        return np.ma.masked_array(result)

    def inverse(self, value):
        # Inverse of the log base 2 transformation
        return 2 ** (value * (np.log2(self.vmax) - np.log2(self.vmin)) + np.log2(self.vmin))

def plot_all_metric_subplots(data, metric, save_path, x_label="L", show_std=True):
    """
    Create a 2x2 figure with subplots arranged as:
         Train           Test
      ----------------------------
      Best Value    |  (row 0)  Best Loss / Best Accuracy
      Best Step     |  (row 1)  Step of Best {metric}
    
    Each subplot is a grouped bar chart where:
      - The x-axis shows L values.
      - Each L group contains one bar per d value.
      - The bar height is either the best metric value (if "best value") 
        or the evaluation step at which the optimum is reached (if "best step").
      - Error bars (if show_std is True) indicate the standard deviation.
      - Bar colors are determined by a discrete colormap for the d options.
    
    Parameters:
      data (dict): Two-level dictionary as described.
      metric (str): Metric name (e.g. "accuracy" or "loss").
      save_path (str): Path to save the combined figure.
      x_label (str): Label for the x-axis.
      show_std (bool): Whether to display error bars.
    """
    # Determine common x-axis values (L values) and d keys.
    L_values = sorted(data.keys(), key=lambda x: float(x))
    first_L = L_values[0]
    # Extract d values from the keys; assumes keys follow a pattern with the d value at position 4.
    d_keys = sorted([int(key.split("_")[4]) for key in data[first_L].keys()])
    # Create a discrete colormap for the d values.
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(d_keys) - 1)) if len(d_keys) > 1 else cmap(0.5) for i in range(len(d_keys))]
    cmap_discrete = ListedColormap(colors)
    bounds = np.arange(len(d_keys) + 1) - 0.5
    norm_discrete = BoundaryNorm(bounds, cmap_discrete.N)

    n_L = len(L_values)
    n_d = len(d_keys)
    indices = np.arange(n_L)
    bar_width = 0.8 / n_d

    # Create a 2x2 figure for the four subplots.
    # Rows: best value (row 0), best step (row 1); Columns: train (col 0), test (col 1).
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex="col", sharey="row")
    measures = ["value", "step"]
    modes = ["train", "test"]

    # Determine the optimum direction based on the metric.
    optimum = "min" if metric == "loss" else "max"

    for i, measure in enumerate(measures):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            # Initialize arrays to hold best measures and errors.
            best_measure = np.zeros((n_L, n_d))
            best_err = np.zeros((n_L, n_d))
            
            # Loop over L values.
            for idx_L, L in enumerate(L_values):
                # Map d value to the configuration key.
                mapping = {}
                for key in data[L].keys():
                    d_val = int(key.split("_")[4])
                    mapping[d_val] = key
                # Loop over sorted d values.
                for idx_d, d_val in enumerate(d_keys):
                    key = mapping[d_val]
                    config = data[L][key]
                    steps = config['steps']
                    m_array = config[mode][metric]
                    best_idx, best_val, std_val = find_best_metric(m_array, optimum=optimum)
                    best_step = steps[best_idx]
                    
                    if measure == "value":
                        best_measure[idx_L, idx_d] = best_val
                        best_err[idx_L, idx_d] = std_val
                    else:  # measure == "step"
                        best_measure[idx_L, idx_d] = best_step
                        best_err[idx_L, idx_d] = std_val  # adjust if needed
            
            # Plot the grouped bars for this subplot.
            for k in range(n_d):
                x_offset = indices + (k - (n_d - 1) / 2) * bar_width
                ax.bar(x_offset, best_measure[:, k],
                       width=bar_width,
                       yerr=best_err[:, k] if show_std else None,
                       capsize=3,
                       color=colors[k],
                       label=f"d = {d_keys[k]}")
            
            # Set x-axis label only on bottom subplots.
            if i == len(measures) - 1:
                ax.set_xlabel(f"{x_label} value")
            else:
                ax.set_xlabel("")
            ax.set_xticks(indices)
            ax.set_xticklabels(L_values, rotation=45, ha="right")
            
            # Set y-axis label only on left subplots.
            if j == 0:
                if measure == "value":
                    ylabel = "Best Loss" if metric == "loss" else "Best Accuracy"
                else:
                    ylabel = "Step of Best " + metric.capitalize()
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")

            if metric == "loss" and measure == "value":
                ax.set_yscale("log")
            
            # Set title only on the top row. Left subplot: Train; Right subplot: Test.
            if i == 0:
                if mode == "train":
                    ax.set_title("Train", fontweight='bold')
                elif mode == "test":
                    ax.set_title("Test", fontweight='bold')
    
    # Create a discrete colorbar showing one color per d value.
    # Old discrete colorbar version:
    # sm = ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
    # sm.set_array([])
    # plt.tight_layout()
    # fig.subplots_adjust(right=0.90)
    # cbar_ax = fig.add_axes([0.93, 0.07, 0.02, 0.85])
    # cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(len(d_keys)))
    # cbar.ax.set_yticklabels([str(d) for d in d_keys])
    # cbar.set_label("d value")
    
    # New smooth gradiated colorbar version:
    sm = ScalarMappable(cmap=cmap, norm=LogNorm2(vmin=min(d_keys), vmax=max(d_keys)))
    sm.set_array([])
    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    ticks = d_keys
    tick_labels = [str(d) for d in d_keys]
    cbar_ax = fig.add_axes([0.93, 0.07, 0.02, 0.88])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("d value")
    
    fig.savefig(save_path)
    plt.close(fig)

def plot_loss_and_accuracy(data, save_directory, x_label="L", show_std=True, file_name="best_histogram"):
    """
    For a given data dictionary, generate and save two figures:
      - One for loss metrics.
      - One for accuracy metrics.
      
    Each figure will contain a 2x2 grid of subplots with the following layout:
         Train           Test
      ----------------------------
      Best Value    |  (row 0)
      Best Step     |  (row 1)
    
    Parameters:
      data (dict): Two-level dictionary as described.
      save_directory (str): Directory to save the figures.
      x_label (str): Label for the x-axis.
      show_std (bool): Whether to display error bars.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    for metric in ["loss", "accuracy"]:
        file_name = f"{file_name}_{metric}.png"
        save_path = os.path.join(save_directory, file_name)
        plot_all_metric_subplots(data, metric, save_path, x_label=x_label, show_std=show_std)

def plot_all_configs_best_metrics_sliced_by_steps(data, save_directory, file_name="sliced", x_label_title="Configuration",
                                                  show_std=True, alphas=[0.5, 0.75, 1.0], x_ticks=None, loss_y_scale=False):
    """
    For each configuration in 'data', slice the metrics according to t_max = α*T (for each α in alphas),
    then combine the sliced metrics from all configurations per α and extract the best metrics.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Slice each configuration by steps using slice_by_steps.
    sliced_data = {config: slice_by_steps(data[config], alphas) for config in data}
    
    # Build a dict per alpha.
    data_by_alpha = []
    for k in range(len(alphas)):
        slice_k = {config: sliced_data[config][k] for config in sliced_data}
        data_by_alpha.append(slice_k)
    
    # --- STEP 3: For each alpha slice, extract best metrics.
    best_by_alpha = {}
    for idx, alpha in enumerate(alphas):
        best_by_alpha[alpha] = extract_config_best_metrics(data_by_alpha[idx], compute_std=show_std)
    
    # Prepare x-axis labels.
    x_labels = [data[config].get('label', config) for config in data.keys()]
    
    # --- STEP 5: Setup color mapping for alphas.
    n_alphas = len(alphas)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(alphas), vmax=max(alphas))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # --- STEP 6: Plot for each metric type.
    for metric in ['loss', 'accuracy']:
        fig, (ax_val, ax_step) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for alpha in alphas:
            best_metrics = best_by_alpha[alpha]
            # Plot training curve as dotted, test as solid.
            ax_val.plot(x_labels, best_metrics['train'][metric]['value'],
                        linestyle=":", marker='o', label=f"Train, α={alpha:.2f}",
                        color=cmap(norm(alpha)))
            ax_val.plot(x_labels, best_metrics['test'][metric]['value'],
                        linestyle="-", marker='o', label=f"Test, α={alpha:.2f}",
                        color=cmap(norm(alpha)))
            ax_step.plot(x_labels, best_metrics['train'][metric]['step'],
                        linestyle=":", marker='o', label=f"Train, α={alpha:.2f}",
                        color=cmap(norm(alpha)))
            ax_step.plot(x_labels, best_metrics['test'][metric]['step'],
                        linestyle="-", marker='o', label=f"Test, α={alpha:.2f}",
                        color=cmap(norm(alpha)))
            if show_std:
                ax_val.fill_between(x_labels,
                                    np.array(best_metrics['train'][metric]['value']) - np.array(best_metrics['train'][metric]['std']),
                                    np.array(best_metrics['train'][metric]['value']) + np.array(best_metrics['train'][metric]['std']),
                                    alpha=0.2, color=cmap(norm(alpha)))
                ax_val.fill_between(x_labels,
                                    np.array(best_metrics['test'][metric]['value']) - np.array(best_metrics['test'][metric]['std']),
                                    np.array(best_metrics['test'][metric]['value']) + np.array(best_metrics['test'][metric]['std']),
                                    alpha=0.2, color=cmap(norm(alpha)))
                ax_step.fill_between(x_labels,
                                     np.array(best_metrics['train'][metric]['step']) - np.array(best_metrics['train'][metric]['std']),
                                     np.array(best_metrics['train'][metric]['step']) + np.array(best_metrics['train'][metric]['std']),
                                     alpha=0.2, color=cmap(norm(alpha)))
                ax_step.fill_between(x_labels,
                                     np.array(best_metrics['test'][metric]['step']) - np.array(best_metrics['test'][metric]['std']),
                                     np.array(best_metrics['test'][metric]['step']) + np.array(best_metrics['test'][metric]['std']),
                                     alpha=0.2, color=cmap(norm(alpha)))
        if metric == 'loss' and loss_y_scale:
            ax_val.set_yscale("log")
        ax_val.set_ylabel(f"Best {metric.capitalize()}")
        ax_val.grid(True)
        ax_step.set_ylabel("Step")
        ax_step.set_xlabel(x_label_title)
        if x_ticks is not None:
            ax_step.set_xticks(x_ticks)
        ax_step.grid(True)
        fig.align_ylabels([ax_val, ax_step])
        plt.tight_layout(rect=[0, 0, 0.90, 1])
        cbar_ax = fig.add_axes([0.93, 0.07, 0.02, 0.85])
        fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label="α (Fraction of Training Steps)")
        
        fig.savefig(os.path.join(save_directory, f"{file_name}_{metric}.png"))
        plt.close()

def visualize_attention_matshow(model, samples, tokenizer, file_path, file_name):
    """
    Visualize attention heatmaps using matshow with overlaid numerical values.

    Parameters:
        model      : The GPT model.
        samples    : Tuple of tensors, where samples[0] is a tensor of shape (B, sequence_length)
                     containing the token indices to decode.
        tokenizer  : Tokenizer with .decode() to convert token indices to strings.
        file_path  : Directory path to save the plots.
        file_name  : Base file name to use when saving plots.
    """
    inputs = torch.stack([data[0] for data in samples])
    model.eval()
    with torch.no_grad():
        logits, (hidden_states, attentions) = model(inputs)
    
    batch_size, num_layers, num_heads, S, _ = attentions.shape

    for i in range(batch_size):
        attention = attentions[i]
        equation = tokenizer.decode(inputs[i])
        tokens = equation.split(" ")
        
        fig, axs = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))

        if num_layers == 1 and num_heads == 1:
            axs = np.array([[axs]])
        elif num_layers == 1:
            axs = np.expand_dims(axs, axis=0)
        elif num_heads == 1:
            axs = np.expand_dims(axs, axis=1)
        
        for layer in range(num_layers):
            for head in range(num_heads):
                ax = axs[layer, head]
                cax = ax.matshow(attention[layer, head].cpu().numpy(), cmap='plasma')
                ax.set_xticks(range(S))
                ax.set_yticks(range(S))
                ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
                if layer == 0:
                    ax.set_title(f"Head {head+1}", fontsize=10)
                if head == 0:
                    ax.set_ylabel(f"Layer {layer+1}", fontsize=10)
                # Overlay numerical values.
                attn_matrix = attention[layer, head].cpu().numpy()
                for (j, k), val in np.ndenumerate(attn_matrix):
                    ax.text(k, j, f"{val:.2f}", ha='center', va='center', fontsize=6, color='white')
        
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.027, 0.02, 0.82])
        fig.colorbar(cax, cax=cbar_ax)
        fig.suptitle(f"Attention for : {equation}", fontsize=10, fontweight='bold')

        os.makedirs(file_path, exist_ok=True)
        fig.savefig(f"{file_path}/{file_name}_matshow_sample_{i+1}.png")
        plt.close(fig)