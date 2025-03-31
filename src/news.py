import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mpl_colors

#from plotter import FIGSIZE, FONTSIZE, LINEWIDTH
FIGSIZE = (6, 4)
LINEWIDTH = 2.0
FONTSIZE = 12

# Plot #1: Training Curves vs r_train
all_r_train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
all_seed = [0, 42, 100, 200, 300]

rows, cols = 1, 1
figsize = FIGSIZE
fontsize = FONTSIZE
linewidth = LINEWIDTH
fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))
axs = fig.subplots(rows, cols)
ax = axs # only one axis (in pratice, you will have axis for each model and each metrics)

color_indices = np.linspace(0, 1, len(all_r_train))
colors = plt.cm.viridis(color_indices)

for j, r_train in enumerate(all_r_train):

    T = 10**4+1
    eval_first = 10**2
    eval_period = 10**2
    all_steps = list(range(eval_first)) + list(range(eval_first, T, eval_period))

    mean = np.log2(r_train)
    std = np.sqrt(r_train+1)
    train_losses = [np.abs( np.random.rand(len(all_steps)) * std  + mean).tolist() for _ in range(len(all_seed))]
    eval_losses = [np.abs(np.random.rand(len(all_steps)) * std  + mean*2).tolist() for _ in range(len(all_seed))]

    train_loss_mean = np.mean(train_losses, axis=0)
    train_loss_std = np.std(train_losses, axis=0)
    eval_loss_mean = np.mean(eval_losses, axis=0)
    eval_loss_std = np.std(eval_losses, axis=0)

    ax.plot(all_steps, train_loss_mean, '-', color=colors[j], label=r_train)
    ax.fill_between(all_steps, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2, color=colors[j])
    ax.plot(all_steps, eval_loss_mean, '--', color=colors[j], label=r_train)
    ax.fill_between(all_steps, eval_loss_mean - eval_loss_std, eval_loss_mean + eval_loss_std, alpha=0.2, color=colors[j])

ax.tick_params(axis='y', labelsize='x-large')
ax.tick_params(axis='x', labelsize='x-large')
ax.set_xlabel('Training Steps (t)', fontsize=fontsize)
ax.set_ylabel('Loss', fontsize=fontsize)
ax.set_title("LSTM", fontsize=fontsize*2)
axs.grid(True)
legend_elements = [Line2D([0], [0], color='k', lw=linewidth, linestyle='-', label='Train'),
                Line2D([0], [0], color='k', lw=linewidth, linestyle='--', label='Eval')]
ax.legend(handles=legend_elements, fontsize=fontsize)


# Normal color bar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(all_r_train), vmax=max(all_r_train)))
sm.set_array([])  # We only need the colormap here, no actual data
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label('$r_{train}$', fontsize=fontsize)
# Set the ticks to correspond to the values in `all_r_train`
cbar.set_ticks(all_r_train)  # Sets tick positions based on `all_r_train`
cbar.set_ticklabels([str(r_train) for r_train in all_r_train])  # Sets tick labels to match `all_r_train`



########################### After plotting for all axis
# Adjust layout and add padding
fig.tight_layout(pad=2)  # Adjust padding between plots
plt.subplots_adjust(right=0.85)  # Adjust right boundary of the plot to fit color bar

#plt.savefig(f"{LOG_DIR}/training_curves_vs_r_train"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

plt.show()




def plot_all_configs_metrics(data, save_directory, mode="mean", seed_index=0,
                                    train_scale=None, loss_scale=False,
                                    figures=[('train', 'loss'), ('train', 'accuracy'),
                                             ('test', 'loss'), ('test', 'accuracy')],
                                    config_labels=None, file_name="all_configs.png"):
    """
    Plot metrics for all configurations.
    
    Parameters:
      data : dict with configuration names as keys; each value is a dict with configuration data.
      figures : list of tuples (mode, metric) indicating which plots to generate.
      config_labels : list of custom labels for the x-axis; if None, configuration keys are used.
      train_scale : Either None or a tuple (m, c, x_break1, x_break2) to apply custom x-axis scale for train curves.
      loss_scale : bool, if True apply log scale to y-axis for loss metrics.
      file_name : str, output file name.
    """
    os.makedirs(save_directory, exist_ok=True)
    config_names = sorted(data.keys())
    x_labels_final = config_labels if config_labels is not None else config_names
    for (mode_key, metric) in figures:
        plt.figure(figsize=(8, 6))
        for config_name, config_data in zip(x_labels_final, data.values()):
            steps = config_data['steps']
            curve, stds = compute_plot_data(config_data[mode_key][metric], mode, seed_index)
            plt.plot(steps, curve, label=config_name)
            if mode == "std" and stds is not None:
                plt.fill_between(steps, curve - stds, curve + stds, alpha=0.2)
        plt.xlabel("Steps")
        plt.title(f"{mode_key.capitalize()} {metric.capitalize()}")
        if loss_scale and metric == "loss":
            plt.yscale("log")
        if train_scale is not None and mode_key == "train":
            m, c, x_break1, x_break2 = train_scale
            plt.gca().set_xscale('custompiecewise', m=m, c=c, x_break1=x_break1, x_break2=x_break2)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_file = f"{mode_key}_{metric}_{file_name}"
        plt.savefig(os.path.join(save_directory, out_file))
        plt.close()



def plot_comparative_best_metrics_v4(data, save_directory, mode="mean", seed_index=0,
                                     config_labels=None, file_name="v4_comparative_best_metrics.png"):
    """
    Version 4: Plot comparative best metrics after sorting configurations by L then d.
    Generates two figures (one for loss and one for accuracy) with disconnected line segments per L group.
    
    Parameters are as standardized.
    """
    os.makedirs(save_directory, exist_ok=True)
    x_labels_final = config_labels if config_labels is not None else list(data.keys())
    configs_list = []
    for config_key in data.keys():
        parts = config_key.split("_")
        L, d = parts[2], parts[4]
        configs_list.append((int(L), int(d), config_key))
    configs_list.sort(key=lambda x: (x[0], x[1]))
    sorted_data = OrderedDict()
    for L, d, config_key in configs_list:
        sorted_data[config_key] = data[config_key]
    best_metrics = extract_best_metrics_for_configs(sorted_data)
    x_labels_sorted = [f"L={L}, d={d}" for L, d, _ in configs_list]
    x_positions = list(range(len(x_labels_sorted)))
    l_indices = defaultdict(list)
    for i, (L, d, _) in enumerate(configs_list):
        l_indices[L].append(i)
    
    # Loss Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train = [best_metrics['train']['loss']['value'][i] for i in indices]
        y_test = [best_metrics['test']['loss']['value'][i] for i in indices]
        ax1.plot(indices, y_train, marker='o', label=f"Train Loss (L={L})")
        ax1.plot(indices, y_test, marker='o', label=f"Test Loss (L={L})")
    ax1.set_yscale('log')
    ax1.set_ylabel("Best Loss (log scale)")
    ax1.set_title("Comparative Best Loss Across Configurations")
    ax1.legend()
    ax1.grid(True)
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train_step = [best_metrics['train']['loss']['step'][i] for i in indices]
        y_test_step = [best_metrics['test']['loss']['step'][i] for i in indices]
        ax2.plot(indices, y_train_step, marker='o', label=f"Train Loss Step (L={L})")
        ax2.plot(indices, y_test_step, marker='o', label=f"Test Loss Step (L={L})")
    ax2.set_ylabel("Step of Best Loss")
    ax2.set_xlabel("Configuration")
    ax2.legend()
    ax2.grid(True)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels_sorted, rotation=45, ha='right')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels_sorted, rotation=45, ha='right')
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, file_name))
    plt.close()
    
    # Accuracy Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train = [best_metrics['train']['accuracy']['value'][i] for i in indices]
        y_test = [best_metrics['test']['accuracy']['value'][i] for i in indices]
        ax1.plot(indices, y_train, marker='o', label=f"Train Accuracy (L={L})")
        ax1.plot(indices, y_test, marker='o', label=f"Test Accuracy (L={L})")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Comparative Best Accuracy Across Configurations")
    ax1.legend()
    ax1.grid(True)
    for L in sorted(l_indices.keys()):
        indices = l_indices[L]
        y_train_step = [best_metrics['train']['accuracy']['step'][i] for i in indices]
        y_test_step = [best_metrics['test']['accuracy']['step'][i] for i in indices]
        ax2.plot(indices, y_train_step, marker='o', label=f"Train Acc Step (L={L})")
        ax2.plot(indices, y_test_step, marker='o', label=f"Test Acc Step (L={L})")
    ax2.set_ylabel("Step")
    ax2.set_xlabel("Configuration")
    ax2.legend()
    ax2.grid(True)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels_sorted, rotation=45, ha='right')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels_sorted, rotation=45, ha='right')
    fig.align_ylabels([ax1, ax2])
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "v4_comparative_best_accuracy.png"))
    plt.close()