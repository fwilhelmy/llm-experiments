import os
import numpy as np
import matplotlib.pyplot as plt

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

def plot_all_configurations(data, save_directory, mode="mean", seed_index=0, config_labels=None):
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
        if m == "train":
            plt.xscale('symlog', linthresh=250)

        plt.xlabel("Steps")
        plt.title(f"{m.capitalize()} {k.capitalize()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f"symlog1_{m}_{k}.png"))
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
    ax2.set_xlabel("Configuration")
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