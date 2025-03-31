import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from collections import defaultdict, OrderedDict

# -----------------------------
# Custom Piecewise Scale Classes
# -----------------------------
class CustomPiecewiseTransform(mtransforms.Transform):
    """
    This transform expands the x-axis between x_break1 and x_break2 by a factor m
    and compresses values beyond x_break2 by a factor c.
    """
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, m=5, c=0.2, x_break1=500, x_break2=1500):
        super().__init__()
        self.m = m
        self.c = c
        self.x_break1 = x_break1
        self.x_break2 = x_break2

    def transform_non_affine(self, x):
        x = np.array(x, dtype=float)
        y = np.empty_like(x)
        mask1 = x < self.x_break1
        y[mask1] = x[mask1]
        mask2 = (x >= self.x_break1) & (x <= self.x_break2)
        y[mask2] = self.x_break1 + self.m * (x[mask2] - self.x_break1)
        mask3 = x > self.x_break2
        y[mask3] = (self.x_break1 + self.m * (self.x_break2 - self.x_break1) +
                    self.c * (x[mask3] - self.x_break2))
        return y

    def inverted(self):
        return InvertedCustomPiecewiseTransform(self.m, self.c, self.x_break1, self.x_break2)

class InvertedCustomPiecewiseTransform(mtransforms.Transform):
    """
    Inverts the CustomPiecewiseTransform.
    """
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, m=5, c=0.2, x_break1=500, x_break2=1500):
        super().__init__()
        self.m = m
        self.c = c
        self.x_break1 = x_break1
        self.x_break2 = x_break2

    def transform_non_affine(self, y):
        y = np.array(y, dtype=float)
        x = np.empty_like(y)
        mask1 = y < self.x_break1
        x[mask1] = y[mask1]
        y_break2_val = self.x_break1 + self.m * (self.x_break2 - self.x_break1)
        mask2 = (y >= self.x_break1) & (y <= y_break2_val)
        x[mask2] = self.x_break1 + (y[mask2] - self.x_break1) / self.m
        mask3 = y > y_break2_val
        x[mask3] = self.x_break2 + (y[mask3] - y_break2_val) / self.c
        return x

    def inverted(self):
        return CustomPiecewiseTransform(self.m, self.c, self.x_break1, self.x_break2)

class CustomPiecewiseScale(mscale.ScaleBase):
    """
    Registers a custom scale named 'custompiecewise' that uses the above transform.
    Custom breakpoints and scale factors (m, c) can be specified.
    """
    name = 'custompiecewise'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.m = kwargs.get('m', 5)
        self.c = kwargs.get('c', 0.2)
        self.x_break1 = kwargs.get('x_break1', 500)
        self.x_break2 = kwargs.get('x_break2', 1500)
        self._transform = CustomPiecewiseTransform(self.m, self.c, self.x_break1, self.x_break2)

    def get_transform(self):
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        from matplotlib.ticker import AutoLocator, ScalarFormatter
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

mscale.register_scale(CustomPiecewiseScale)

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

def find_best_metric(metrics, merge_ops=True, optimum="min"):
    """
    Find the best (optimal) metric value.
    
    Parameters:
      metrics : NumPy array of shape (n_runs, n_op_orders, n_evals)
      merge_ops : bool, if True, compute a single best value after merging all ops and runs;
                  if False, compute the best value for each op order (averaging over runs).
      optimum : str, "min" for loss or "max" for accuracy.
      
    Returns:
      If merge_ops is True:
         best_idx: index along the evaluation axis.
         best_value: the optimal value.
         best_std: the std at that index.
      Else:
         best_idx: list of best indices for each op order.
         best_value: list of best values per op order.
         best_std: list of best std values per op order.
    """
    curve, stds = compute_plot_data(metrics, merge_ops=merge_ops)
    if merge_ops:
        optimum_fn = np.argmin if optimum == "min" else np.argmax
        best_idx = optimum_fn(curve)
        return best_idx, curve[best_idx], stds[best_idx]
    else:
        best_idx_list = []
        best_value_list = []
        best_std_list = []
        optimum_fn = np.argmin if optimum == "min" else np.argmax
        for i in range(curve.shape[0]):
            idx = optimum_fn(curve[i])
            best_idx_list.append(idx)
            best_value_list.append(curve[i, idx])
            best_std_list.append(stds[i, idx])
        return best_idx_list, best_value_list, best_std_list

def extract_config_best_metrics(all_metrics, metrics_to_extract=['loss', 'accuracy'], merge_ops=True, compute_std=True):
    """
    Extract best metrics (and the step at which they occur) for each configuration.
    
    Parameters:
      all_metrics : dict with configuration names as keys. Each value should contain:
                    - "train": {"loss": array, "accuracy": array}
                    - "test": {"loss": array, "accuracy": array}
                    - "steps": 1D array of evaluation steps.
      metrics_to_extract : list of metric names to extract (default: ['loss', 'accuracy']).
      merge_ops : bool, if True, best metrics are computed on merged ops (averaging over runs and op orders),
                  else computed separately for each op order.
      compute_std : bool, if True, also include the standard deviation with the best value and the step.
      
    Returns:
      best_metrics : dict structured as:
         {
           "train": {"loss": {"value": [...], "step": [...], "std": [...] (if stds True)},
                     "accuracy": {"value": [...], "step": [...], "std": [...] (if stds True)}},
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
                best_idx, best_val, best_std = find_best_metric(config[mode][metric],
                                                                merge_ops=merge_ops,
                                                                optimum=optimum)
                best_metrics[mode][metric]['value'].append(best_val)
                step = config['steps'][best_idx] if merge_ops else [config['steps'][i] for i in best_idx]
                best_metrics[mode][metric]['step'].append(step)
                if compute_std:
                    best_metrics[mode][metric]['std'].append(best_std)
    return best_metrics

# -----------------------------
# Standardized Plotting Functions
# All functions now have parameters:
#   (data, save_directory, file_name, show_std=True, merge_ops=True,
#    train_x_scale=(35,0.2,10,300), x_ticks=None, loss_y_scale=False)
# -----------------------------

def plot_config_loss_accs(data, save_directory, file_name=None,
                          show_std=True, train_x_scale=False, x_ticks=None,
                          loss_y_scale=False):
    """
    Plot metrics for a single configuration.
    
    Parameters:
      data : dict with keys "steps", "train": {"loss", "accuracy"}, "test": {"loss", "accuracy"}
      save_directory : Directory to save the plot.
      file_name : Output file name.
      show_std : If True, show standard deviation as an error band.
      merge_ops : If True, merge over runs and ops; if False, show separate curves for each op.
      train_x_scale : bool, if True, set x-axis to log scale for subplots where mode is 'train'.
      x_ticks : Optional custom x-tick values.
      loss_y_scale : If True, set y-axis to log scale for loss metrics.
    """
    os.makedirs(save_directory, exist_ok=True)
    steps = data['steps']
    for metric in ['loss', 'accuracy']:
        plt.figure(figsize=(8, 6))
        curve_train, stds_train = compute_plot_data(data['train'][metric])
        curve_test, stds_test = compute_plot_data(data['test'][metric])
        plt.plot(steps, curve_train, label="Train", color='blue')
        plt.plot(steps, curve_test, label="Test", color='red')
        if show_std and stds_train is not None and stds_test is not None:
            plt.fill_between(steps, curve_train - stds_train, curve_train + stds_train,
                             color='blue', alpha=0.2)
            plt.fill_between(steps, curve_test - stds_test, curve_test + stds_test,
                             color='red', alpha=0.2)
        plt.xlabel("Steps")
        plt.ylabel(metric.capitalize())
        if loss_y_scale and metric == "loss":
            plt.yscale("log")
        if train_x_scale:
            plt.xscale('log')
        if x_ticks is not None:
            plt.xticks(x_ticks)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        file_name = f"{metric}_{file_name}.png" if file_name else f"{metric}.png"
        plt.savefig(os.path.join(save_directory, file_name))
        plt.close()

def plot_config_ops(data, save_directory, file_name=None,
                    show_std=True, train_x_scale=False, x_ticks=None,
                    loss_y_scale=False):
    """
    Plot metrics for a single configuration, showing each operation order individually.
    
    Instead of plotting curves for different configurations, this function plots
    the metrics for each op order (using data['operation_orders'] for the labels)
    in a 2x2 grid:
      - Top Left: Train Loss, Top Right: Test Loss,
      - Bottom Left: Train Accuracy, Bottom Right: Test Accuracy.
      
    The subplots share the x-axis (only the bottom row is labeled) and the left column
    displays y-axis labels. Legends and grids are applied to each subplot.
    
    Parameters:
      data : dict
          Contains:
            - "steps": 1D array of evaluation steps.
            - "train": dict with keys (e.g. "loss", "accuracy") whose values are arrays
              of shape (n_runs, n_op_orders, n_evals).
            - "test": dict with similar structure as "train".
            - "operation_orders": list of labels for each op.
      save_directory : str
          Directory to save the plots.
      file_name : str
          Output file name.
      show_std : bool, default True
          If True, show standard deviation as an error band.
      train_x_scale : bool, if True, set x-axis to log scale for subplots where mode is 'train'.
      x_ticks : list or None
          Optional custom x-tick values.
      loss_y_scale : bool, default False
          If True, set y-axis to log scale for loss metrics.
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
    
    # Apply custom train x-scale to left column subplots.
    if train_x_scale:
        for ax in [axs[0, 0], axs[1, 0]]:
            ax.set_xscale('log')
    
    if x_ticks is not None:
        for ax in axs.flat:
            ax.set_xticks(x_ticks)
    
    plt.tight_layout()
    file_name = f"{file_name}.png" if file_name else f"operation_orders.png"
    plt.savefig(os.path.join(save_directory, file_name))
    plt.close()

def plot_all_configs_metrics(data, save_directory, file_name=None,
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
    file_name = f"{file_name}.png" if file_name else "all_configs.png"
    plt.savefig(os.path.join(save_directory, file_name))
    plt.close()

def plot_all_configs_best_metrics(data, save_directory, file_name=None,
                                  show_std=True, x_label_title="Configuration",
                                  x_ticks=None, loss_y_scale=False):
    """
    Plot comparative best metrics (Loss and Accuracy) across configurations.
    Two figures are generated (one for loss and one for accuracy).
    
    Parameters:
      data : dict with configuration names as keys. Each config should include:
             - "train": {"loss": array, "accuracy": array}
             - "test": {"loss": array, "accuracy": array}
             - "steps": 1D array of evaluation steps.
      show_std : bool, if True, error bands representing std are shown.
      x_label_title : str, label for the x-axis.
      x_ticks : optional custom x-tick values.
      loss_y_scale : bool, if True, set the y-axis to log scale for loss metrics.
      file_name : str, output file name for the loss figure; the accuracy figure will use a similar name.
    """
    os.makedirs(save_directory, exist_ok=True)
    # Use the 'label' key if available, otherwise the key name.
    x_labels = [data[config].get('label', config) for config in data.keys()]
    best_metrics = extract_config_best_metrics(data, merge_ops=True, compute_std=show_std)
    
    # ---- Loss Figure ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot best loss values with markers
    ax1.plot(x_labels, best_metrics['train']['loss']['value'], marker='o', label='Best Train Loss')
    ax1.plot(x_labels, best_metrics['test']['loss']['value'], marker='o', label='Best Test Loss')
    if show_std:
        # Use fill_between to show error bands.
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
    ax1.set_ylabel("Best Loss (log scale)")
    ax1.set_title("Comparative Best Loss Across Configurations")
    ax1.legend()
    ax1.grid(True)
    
    # Plot best loss steps
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
    ax2.set_ylabel("Step of Best Loss")
    ax2.set_xlabel(x_label_title)
    if x_ticks is not None:
        ax2.set_xticks(x_ticks)
    ax2.legend()
    ax2.grid(True)
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    file_name = f"{file_name}_loss.png" if file_name else f"comparative_best_loss.png"
    plt.savefig(os.path.join(save_directory, file_name))
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
    ax1.set_title("Comparative Best Accuracy Across Configurations")
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
    file_name = f"{file_name}_accuracy.png" if file_name else f"comparative_best_accuracy.png"
    plt.savefig(os.path.join(save_directory, file_name))
    plt.close()