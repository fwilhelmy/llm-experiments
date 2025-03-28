import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

def get_plot_data(metrics, mode="mean", seed_index=0):
    # metrics has shape (n_runs, n_op_orders, n_evals)
    if mode in ["mean", "std", "specific"]:
        curve = np.mean(metrics[seed_index], axis=0) if mode == "specific" else np.mean(metrics, axis=(0, 1))
        stds = np.std(metrics, axis=(0, 1)) if mode == "std" else None
        return curve, stds
    else: raise ValueError("Mode not recognized. Use 'mean', 'std', or 'specific'.")

def extract_best_for_metric(metrics, mode):
    # For loss, lower is better; for accuracy, higher is better.
    best_idx = np.unravel_index(np.argmin(metrics) if mode == 'min' else np.argmax(metrics), metrics.shape)
    best_value = metrics[best_idx]
    return best_value, best_idx

def extract_configs_best_metrics(all_metrics):
    best_metrics = { k: { m: {'value':[],  'step': []}
        for m in ['loss', 'accuracy'] } for k in ['train', 'test']}
    
    for config_metrics in all_metrics.values():
        for mode in ['train', 'test']:
            # Compute best loss for each configuration
            best_loss, best_loss_idx = extract_best_for_metric(config_metrics[mode]['loss'], 'min')
            best_metrics[mode]['loss']['value'].append(best_loss)
            best_metrics[mode]['loss']['step'].append(config_metrics['steps'][best_loss_idx[2]])
            
            # Compute best accuracy for each configuration
            best_acc, best_acc_idx = extract_best_for_metric(config_metrics[mode]['accuracy'], 'max')
            best_metrics[mode]['accuracy']['value'].append(best_acc)
            best_metrics[mode]['accuracy']['step'].append(config_metrics['steps'][best_acc_idx[2]])
    
    return best_metrics

def plot_configuration_metrics(data, save_directory, mode="mean", seed_index=0):
    os.makedirs(save_directory, exist_ok=True)
    steps = data['steps']
    for metric in ['loss', 'accuracy']:
        plt.figure(figsize=(8, 6))

        curve_train, stds_train = get_plot_data(data['train'][metric], mode, seed_index)
        curve_test, stds_test = get_plot_data(data['test'][metric], mode, seed_index)

        plt.plot(steps, curve_train, label="Train", color='blue')
        plt.plot(steps, curve_test, label="Test", color='red')
        if mode == "std":
            plt.fill_between(steps, curve_train - stds_train, curve_train + stds_train, color='blue', alpha=0.2)
            plt.fill_between(steps, curve_test - stds_test, curve_test + stds_test, color='red', alpha=0.2)

        plt.xlabel("Steps")
        plt.ylabel(f"{metric.capitalize()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f"{metric}.png"))
        plt.close()

def plot_all_configurations(data, save_directory, mode="mean", seed_index=0, config_labels=None, symlog=False):
    os.makedirs(save_directory, exist_ok=True)
    figures = [('train', 'loss'),
               ('train', 'accuracy'),
               ('test', 'loss'),
               ('test', 'accuracy')]

    for m, k in figures:
        plt.figure(figsize=(8, 6))
        for config_name, config_data in zip((config_labels or data.keys()), data.values()):
            steps = config_data['steps']  # 1D array of evaluation steps
            curve, stds = get_plot_data(config_data[m][k], mode, seed_index) # (n_runs, n_op_orders, n_evals)
            plt.plot(steps, curve, label=config_name)
            if mode == "std" and stds is not None:
                plt.fill_between(steps, curve - stds, curve + stds, alpha=0.2)

        # For train plots, use a symlog scale with a linear threshold of 1000
        if symlog and m == "train":
            #plt.xscale('symlog', linthresh=250, linscale=2)

            # Do this:
            # ax.set_xscale('symlog', linthresh=10, base=10)
            # ax.set_xlim(1, 10000)
            plt.xscale('log')
            plt.xlim(1, 10000)
        
            

        plt.xlabel("Steps")
        plt.title(f"{m.capitalize()} {k.capitalize()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f"{m}_{k}.png"))
        plt.close()

def plot_all_configurations_grouped(data, save_directory, mode="mean", seed_index=0, config_labels=None, symlog=True):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    os.makedirs(save_directory, exist_ok=True)
    
    # (Changed) Sorted list of configuration names and use config_labels if provided.
    config_names = sorted(data.keys())
    x_labels = config_labels if config_labels is not None else config_names

    # (Changed) Create a 2x2 grid of subplots:
    # Top left: Train Loss, Top right: Test Loss, Bottom left: Train Accuracy, Bottom right: Test Accuracy
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex='col', sharey='row')
    
    # Loop over each configuration and plot its curves in the appropriate subplot.
    for config_name, config_data in zip(x_labels, data.values()):
        steps = config_data['steps'] # 1D array of evaluation steps
        
        # Plot for Train Loss (top left)
        curve, stds = get_plot_data(config_data['train']['loss'], mode, seed_index)
        axs[0, 0].plot(steps, curve, label=config_name)
        if mode == "std" and stds is not None:
            axs[0, 0].fill_between(steps, curve - stds, curve + stds, alpha=0.2)
        
        # Plot for Test Loss (top right)
        curve, stds = get_plot_data(config_data['test']['loss'], mode, seed_index)
        axs[0, 1].plot(steps, curve, label=config_name)
        if mode == "std" and stds is not None:
            axs[0, 1].fill_between(steps, curve - stds, curve + stds, alpha=0.2)
        
        # Plot for Train Accuracy (bottom left)
        curve, stds = get_plot_data(config_data['train']['accuracy'], mode, seed_index)
        axs[1, 0].plot(steps, curve, label=config_name)
        if mode == "std" and stds is not None:
            axs[1, 0].fill_between(steps, curve - stds, curve + stds, alpha=0.2)
        
        # Plot for Test Accuracy (bottom right)
        curve, stds = get_plot_data(config_data['test']['accuracy'], mode, seed_index)
        axs[1, 1].plot(steps, curve, label=config_name)
        if mode == "std" and stds is not None:
            axs[1, 1].fill_between(steps, curve - stds, curve + stds, alpha=0.2)
    
    # (Changed) For train plots (left column), apply log scale if requested.
    if symlog:
        axs[0, 0].set_xscale('log')
        axs[1, 0].set_xscale('log')
    
    # (Changed) Set y-axis to log scale for loss plots (top row).
    axs[0, 0].set_yscale('log')
    axs[0, 1].set_yscale('log')
    
    # (Changed) Set subplot titles.
    axs[0, 0].set_title("Train", fontweight='bold')
    axs[0, 1].set_title("Test", fontweight='bold')
    
    # (Changed) Set x-labels for bottom row only.
    axs[1, 0].set_xlabel("Steps")
    axs[1, 1].set_xlabel("Steps")
    
    # Set y-labels for left column.
    axs[0, 0].set_ylabel("Loss")
    axs[1, 0].set_ylabel("Accuracy")
    
    # Add legends and grids to each subplot.
    for ax in axs.flat:
        ax.legend()
        ax.grid(True)
    
    # (Changed) Save one combined figure.
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "comparative_configurations_depth.png"))
    plt.close()

def plot_comparative_best_metrics(data, save_directory, config_title=None, config_labels=None):
    os.makedirs(save_directory, exist_ok=True)
    x_labels = config_labels or data.keys()
    
    # Containers for extracted data
    best_metrics = extract_configs_best_metrics(data)

    # --- Plot Loss Figures ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top subplot: best loss values (log scale)
    ax1.plot(x_labels, best_metrics['train']['loss']['value'], marker='o', label='Train Loss')
    ax1.plot(x_labels, best_metrics['test']['loss']['value'], marker='o', label='Test Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel("Best Loss (log scale)")
    ax1.set_title("Comparative Best Loss Across Configurations")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom subplot: steps at which best loss was first reached
    ax2.plot(x_labels, best_metrics['train']['loss']['step'], marker='o', label='Train Loss Step')
    ax2.plot(x_labels, best_metrics['test']['loss']['step'], marker='o', label='Test Loss Step')
    ax2.set_ylabel("Step of Best Loss")
    ax2.set_xlabel(config_title or "Configuration")
    ax2.legend()
    ax2.grid(True)

    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "comparative_best_loss.png"))
    plt.close()
    
    # --- Plot Accuracy Figures ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top subplot: best accuracy values
    ax1.plot(x_labels, best_metrics['train']['accuracy']['value'], marker='o', label='Best Train Accuracy')
    ax1.plot(x_labels, best_metrics['test']['accuracy']['value'], marker='o', label='Best Test Accuracy')
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Comparative Best Accuracy Across Configurations")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom subplot: steps at which best accuracy was first reached
    ax2.plot(x_labels, best_metrics['train']['accuracy']['step'], marker='o', label='Best Train Accuracy Step')
    ax2.plot(x_labels, best_metrics['test']['accuracy']['step'], marker='o', label='Best Test Accuracy Step')
    ax2.set_ylabel("Step")
    ax2.set_xlabel(config_title or "Configuration")
    ax2.legend()
    ax2.grid(True)

    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "comparative_best_accuracy.png"))
    plt.close()


def plot_exp4_comparative(data, save_directory, config_title=None, config_labels=None):
    os.makedirs(save_directory, exist_ok=True)
    x_labels = config_labels or data.keys()
    
    
    # Containers for extracted data
    best_metrics = {}
    for config_name, config_metrics in data.items():
        best_metrics[config_name] = {}
        for mode in ['train', 'test']:
            best_metrics[config_name][mode] = {'loss':{}, 'accuracy': {}}
            target_metric = best_metrics[config_name][mode]
            # Compute best loss for each configuration
            best_loss, best_loss_idx = extract_best_for_metric(config_metrics[mode]['loss'], 'min')
            target_metric['loss']['value'] = best_loss
            target_metric['loss']['step'] = config_metrics['steps'][best_loss_idx[2]]
            
            # Compute best accuracy for each configuration
            best_acc, best_acc_idx = extract_best_for_metric(config_metrics[mode]['accuracy'], 'max')
            target_metric['accuracy']['value'] = best_acc
            target_metric['accuracy']['step'] = config_metrics['steps'][best_acc_idx[2]]

    def sort_key(config):
        config_specs = config[0].split("_")
        return int(config_specs[2]), int(config_specs[4])
    sorted_best_metrics = dict(sorted(best_metrics.items(), key=sort_key))
    # extract best accuracy for training
    best_train_acc = [v['train']['accuracy']['value'] for v in sorted_best_metrics.values()]
    best_train_acc_steps = [v['train']['accuracy']['step'] for v in sorted_best_metrics.values()]
    # extract best accuracy for testing
    best_test_acc = [v['test']['accuracy']['value'] for v in sorted_best_metrics.values()]
    best_test_acc_steps = [v['test']['accuracy']['step'] for v in sorted_best_metrics.values()]
    # extract best loss for training
    best_train_loss = [v['train']['loss']['value'] for v in sorted_best_metrics.values()]
    best_train_loss_steps = [v['train']['loss']['step'] for v in sorted_best_metrics.values()]
    # extract best loss for testing
    best_test_loss = [v['test']['loss']['value'] for v in sorted_best_metrics.values()]
    best_test_loss_steps = [v['test']['loss']['step'] for v in sorted_best_metrics.values()]

    def sort_labels(config):
        L = int(config[2])
        d = int(config[7:])
        return L, d
    sorted_x_labels = sorted(x_labels, key=sort_labels)

    # --- Plot Loss Figures ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top subplot: best loss values (log scale)
    ax1.plot(sorted_x_labels, best_train_loss, marker='o', label='Train Loss')
    ax1.plot(sorted_x_labels, best_test_loss, marker='o', label='Test Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel("Best Loss (log scale)")
    ax1.set_title("Comparative Best Loss Across Configurations")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom subplot: steps at which best loss was first reached
    ax2.plot(sorted_x_labels, best_train_loss_steps, marker='o', label='Train Loss Step')
    ax2.plot(sorted_x_labels, best_test_loss_steps, marker='o', label='Test Loss Step')
    ax2.set_ylabel("Step of Best Loss")
    ax2.set_xlabel(config_title or "Configuration")
    ax2.legend()
    ax2.grid(True)

    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "comparative_best_loss.png"))
    plt.close()
    
    # --- Plot Accuracy Figures ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top subplot: best accuracy values
    ax1.plot(sorted_x_labels, best_train_acc, marker='o', label='Best Train Accuracy')
    ax1.plot(sorted_x_labels, best_test_acc, marker='o', label='Best Test Accuracy')
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Comparative Best Accuracy Across Configurations")
    ax1.legend()
    ax1.grid(True)
    
    # Bottom subplot: steps at which best accuracy was first reached
    ax2.plot(sorted_x_labels, best_train_acc_steps, marker='o', label='Best Train Accuracy Step')
    ax2.plot(sorted_x_labels, best_test_acc_steps, marker='o', label='Best Test Accuracy Step')
    ax2.set_ylabel("Step")
    ax2.set_xlabel(config_title or "Configuration")
    ax2.legend()
    ax2.grid(True)

    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "comparative_best_accuracy.png"))
    plt.close()

    import os
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

def parse_L_d(config_key):
    """
    Example parser that expects config_key strings like: 'model_L1_d64' or 'run_L=2_d=128', etc.
    Adjust this logic to match your naming convention.
    """
    # Simple example: split by '_' and look for segments starting with 'L' or 'd'
    parts = config_key.split('_')
    L, d = parts[2], parts[4]
    return L, d

def plot_comparative_best_metricsv4(data, save_directory, config_title=None, config_labels=None):
    """
    Plots two figures (Loss and Accuracy).
      - Each figure has 2 stacked subplots (top: best value, bottom: step).
      - x-axis: all configs sorted by L first, then d.
      - Each L group is a separate (disconnected) line.
    """
    os.makedirs(save_directory, exist_ok=True)

    # 1) Collect (L, d, config_key) tuples
    configs_list = []
    for config_key in data.keys():
        L, d = parse_L_d(config_key)
        configs_list.append((L, d, config_key))
    
    # 2) Sort by (L, d)
    configs_list.sort(key=lambda x: (x[0], x[1]))  # first by L, then by d
    
    # 3) Create an OrderedDict in that sorted order, so extract_configs_best_metrics
    #    returns arrays in exactly this sequence.
    sorted_data = OrderedDict()
    for (L, d, ckey) in configs_list:
        sorted_data[ckey] = data[ckey]
    
    # 4) Extract the best metrics in sorted order
    best_metrics = extract_configs_best_metrics(sorted_data)
    
    # 5) Build x-axis labels and positions
    x_labels = [f"L={L}, d={d}" for (L, d, _) in configs_list]
    x_positions = list(range(len(x_labels)))
    
    # 6) Group indices by L, so we can plot them in disconnected segments
    l_indices = defaultdict(list)
    for i, (L, d, _) in enumerate(configs_list):
        l_indices[L].append(i)
    
    # ----------------------------------------------------------------------
    # Figure 1: Best Loss (top) and Step of Best Loss (bottom)
    # ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Arrays for best loss
    train_loss_values = best_metrics['train']['loss']['value']
    test_loss_values  = best_metrics['test']['loss']['value']
    
    # -- Top subplot: Best Loss (log scale)
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train = [train_loss_values[i] for i in indices]
        y_test  = [test_loss_values[i]  for i in indices]
        
        ax1.plot(indices, y_train, marker='o', label=f"Train Loss (L={L})")
        ax1.plot(indices, y_test,  marker='o', label=f"Test Loss (L={L})")
    
    ax1.set_yscale('log')
    ax1.set_ylabel("Best Loss (log scale)")
    ax1.set_title("Comparative Best Loss Across Configurations")
    ax1.grid(True)
    ax1.legend()
    
    # -- Bottom subplot: Step at which best Loss was reached
    train_loss_steps = best_metrics['train']['loss']['step']
    test_loss_steps  = best_metrics['test']['loss']['step']
    
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train_step = [train_loss_steps[i] for i in indices]
        y_test_step  = [test_loss_steps[i]  for i in indices]
        
        ax2.plot(indices, y_train_step, marker='o', label=f"Train Loss Step (L={L})")
        ax2.plot(indices, y_test_step,  marker='o', label=f"Test Loss Step (L={L})")
    
    ax2.set_ylabel("Step of Best Loss")
    ax2.set_xlabel(config_title or "Configuration")
    ax2.grid(True)
    ax2.legend()
    
    # -- Shared x-axis ticks/labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "v4_comparative_best_loss.png"))
    plt.close()
    
    # ----------------------------------------------------------------------
    # Figure 2: Best Accuracy (top) and Step of Best Accuracy (bottom)
    # ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Arrays for best accuracy
    train_acc_values = best_metrics['train']['accuracy']['value']
    test_acc_values  = best_metrics['test']['accuracy']['value']
    
    # -- Top subplot: Best Accuracy
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train = [train_acc_values[i] for i in indices]
        y_test  = [test_acc_values[i]  for i in indices]
        
        ax1.plot(indices, y_train, marker='o', label=f"Best Train Accuracy (L={L})")
        ax1.plot(indices, y_test,  marker='o', label=f"Best Test Accuracy (L={L})")
    
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Comparative Best Accuracy Across Configurations")
    ax1.grid(True)
    ax1.legend()
    
    # -- Bottom subplot: Step at which best Accuracy was reached
    train_acc_steps = best_metrics['train']['accuracy']['step']
    test_acc_steps  = best_metrics['test']['accuracy']['step']
    
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train_step = [train_acc_steps[i] for i in indices]
        y_test_step  = [test_acc_steps[i]  for i in indices]
        
        ax2.plot(indices, y_train_step, marker='o', label=f"Best Train Acc Step (L={L})")
        ax2.plot(indices, y_test_step,  marker='o', label=f"Best Test Acc Step (L={L})")
    
    ax2.set_ylabel("Step")
    ax2.set_xlabel(config_title or "Configuration")
    ax2.grid(True)
    ax2.legend()
    
    # -- Shared x-axis ticks/labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "v4_comparative_best_accuracy.png"))
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms

# -----------------------
# Define the custom piecewise transform & scale with customizable breakpoints
# -----------------------
class CustomPiecewiseTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, m=5, c=0.2, x_break1=500, x_break2=1500):
        super().__init__()
        self.m = m      # Expansion factor for x in [x_break1, x_break2]
        self.c = c      # Compression factor for x > x_break2
        self.x_break1 = x_break1
        self.x_break2 = x_break2

    def transform_non_affine(self, x):
        x = np.array(x, dtype=float)
        y = np.empty_like(x)
        # For x < x_break1: identity transformation
        mask1 = x < self.x_break1
        y[mask1] = x[mask1]
        # For x between x_break1 and x_break2: expanded with slope = m
        mask2 = (x >= self.x_break1) & (x <= self.x_break2)
        y[mask2] = self.x_break1 + self.m * (x[mask2] - self.x_break1)
        # For x > x_break2: compressed with slope = c
        mask3 = x > self.x_break2
        y[mask3] = (self.x_break1 +
                    self.m * (self.x_break2 - self.x_break1) +
                    self.c * (x[mask3] - self.x_break2))
        return y

    def inverted(self):
        return InvertedCustomPiecewiseTransform(self.m, self.c, self.x_break1, self.x_break2)

class InvertedCustomPiecewiseTransform(mtransforms.Transform):
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
        # For y < x_break1: identity
        mask1 = y < self.x_break1
        x[mask1] = y[mask1]
        # For y between x_break1 and y_break2_val (expanded region)
        y_break2_val = self.x_break1 + self.m * (self.x_break2 - self.x_break1)
        mask2 = (y >= self.x_break1) & (y <= y_break2_val)
        x[mask2] = self.x_break1 + (y[mask2] - self.x_break1) / self.m
        # For y > y_break2_val: inverse of compression
        mask3 = y > y_break2_val
        x[mask3] = self.x_break2 + (y[mask3] - y_break2_val) / self.c
        return x

    def inverted(self):
        return CustomPiecewiseTransform(self.m, self.c, self.x_break1, self.x_break2)

class CustomPiecewiseScale(mscale.ScaleBase):
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

# Register the custom scale so it can be used by its name.
mscale.register_scale(CustomPiecewiseScale)

# -----------------------
# Updated Plotting Function for Experiment 3 (Combined Figure)
# -----------------------
def plot_exp3_ops_combined(data, save_directory, config_title=None):
    """
    Creates a combined figure with 2 rows x 2 columns arranged as:
      Top row: Loss figures
         - Left: Train Loss with a custom x-axis scale (using customizable breakpoints)
           and bold title "TRAIN"
         - Right: Test Loss with bold title "TEST"
      Bottom row: Accuracy figures (left: Train Accuracy, right: Test Accuracy; no titles)
      
    Each subplot plots n_order curves (one per operation order) with error bands (mean ± std)
    computed over runs (averaging over axis 1) from arrays of shape (n_order, n_run, n_evaluations).
    Loss subplots use a logarithmic y-axis.
    
    The custom x-axis scale expands values between x_break1 and x_break2, and compresses values beyond x_break2.
    
    Expected data structure:
      data["train"]["loss"], data["train"]["accuracy"],
      data["test"]["loss"], data["test"]["accuracy"]:
         NumPy arrays of shape (n_order, n_run, n_evaluations).
      data["operation_orders"]: list of labels (e.g., ["order2", "order3"]).
      data["steps"]: (optional) list/array of step values for the x-axis.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Determine x-axis steps
    if "steps" in data:
        steps = data["steps"]
    else:
        n_evals = data["train"]["loss"].shape[2]
        steps = list(range(n_evals))
    
    # Compute mean and standard deviation over runs (axis=1)
    train_loss_avg = np.mean(data["train"]["loss"], axis=1)
    train_loss_std = np.std(data["train"]["loss"], axis=1)
    
    train_acc_avg = np.mean(data["train"]["accuracy"], axis=1)
    train_acc_std = np.std(data["train"]["accuracy"], axis=1)
    
    test_loss_avg = np.mean(data["test"]["loss"], axis=1)
    test_loss_std = np.std(data["test"]["loss"], axis=1)
    
    test_acc_avg = np.mean(data["test"]["accuracy"], axis=1)
    test_acc_std = np.std(data["test"]["accuracy"], axis=1)
    
    op_order_labels = data["operation_orders"]
    n_order = len(op_order_labels)
    
    # Create a 2x2 subplot figure with shared x-axis for columns and shared y-axis for rows.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex='col', sharey='row')
    
    # --- Top Row: Loss Figures ---
    # Left: Train Loss, Right: Test Loss
    ax_loss_train = axes[0, 0]
    ax_loss_test  = axes[0, 1]
    
    for i in range(n_order):
        # Plot Train Loss curves (with error band)
        ax_loss_train.plot(steps, train_loss_avg[i, :], label=op_order_labels[i])
        ax_loss_train.fill_between(steps,
                                   train_loss_avg[i, :] - train_loss_std[i, :],
                                   train_loss_avg[i, :] + train_loss_std[i, :],
                                   alpha=0.3)
        # Plot Test Loss curves (with error band)
        ax_loss_test.plot(steps, test_loss_avg[i, :], label=op_order_labels[i])
        ax_loss_test.fill_between(steps,
                                  test_loss_avg[i, :] - test_loss_std[i, :],
                                  test_loss_avg[i, :] + test_loss_std[i, :],
                                  alpha=0.3)
    
    # Bold titles for the top row subplots
    ax_loss_train.set_title(r"$\bf{TRAIN}$", fontsize=14)
    ax_loss_test.set_title(r"$\bf{TEST}$", fontsize=14)
    
    ax_loss_train.set_ylabel("Loss")
    # Use logarithmic y-axis for loss subplots
    ax_loss_train.set_yscale("log")
    ax_loss_test.set_yscale("log")
        
    ax_loss_train.grid(True)
    ax_loss_test.grid(True)
    
    # --- Bottom Row: Accuracy Figures (No titles) ---
    ax_acc_train = axes[1, 0]
    ax_acc_test  = axes[1, 1]
    
    for i in range(n_order):
        ax_acc_train.plot(steps, train_acc_avg[i, :], label=op_order_labels[i])
        ax_acc_train.fill_between(steps,
                                  train_acc_avg[i, :] - train_acc_std[i, :],
                                  train_acc_avg[i, :] + train_acc_std[i, :],
                                  alpha=0.3)
        ax_acc_test.plot(steps, test_acc_avg[i, :], label=op_order_labels[i])
        ax_acc_test.fill_between(steps,
                                 test_acc_avg[i, :] - test_acc_std[i, :],
                                 test_acc_avg[i, :] + test_acc_std[i, :],
                                 alpha=0.3)
    
    ax_acc_train.set_ylabel("Accuracy")
    xlabel = "Steps" if config_title is None else config_title
    ax_acc_train.set_xlabel(xlabel)
    ax_acc_test.set_xlabel(xlabel)
    
    # Also apply the custom piecewise scale to the training accuracy subplot
    ax_acc_train.set_xscale('custompiecewise', m=7, c=0.5, x_break1=0, x_break2=700)
    ax_loss_train.set_xscale('custompiecewise', m=7, c=0.5, x_break1=0, x_break2=700)
    ax_acc_train.grid(True)
    ax_loss_train.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "exp3_ops_combined.png"))
    plt.close()
