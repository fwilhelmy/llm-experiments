import os
import numpy as np
import matplotlib.pyplot as plt
from metrics import get_plot_data, extract_best_train_and_test_metrics

def plot_metrics_across_configurations(data, save_directory, mode="mean", seed_index=0):
    """
    Generate plots comparing a single metric across multiple configurations.
    
    This function produces one figure per metric (train loss, train accuracy, test loss,
    test accuracy) where each configuration is represented by its corresponding curve.
    
    Parameters:
        data (dict): Dictionary where each key is a configuration name and the value is a dict containing:
                     - 'train': {'loss': [...], 'accuracy': [...]}
                     - 'test': {'loss': [...], 'accuracy': [...]}
                     - 'all_steps': list of lists (step values for each run)
        save_directory (str): Directory where the plots will be saved.
        mode (str): 'mean', 'std', or 'specific' to determine how the data is aggregated.
        seed_index (int): Run index to use if mode is 'specific'.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    metrics_to_plot = [('train', 'loss'),
                       ('train', 'accuracy'),
                       ('test', 'loss'),
                       ('test', 'accuracy')]
    
    for data_type, metric in metrics_to_plot:
        plt.figure(figsize=(8, 6))
        for config_name, config_data in data.items():
            steps_runs = config_data['all_steps']
            y_values_runs = config_data[data_type][metric]
            steps, curve, stds = get_plot_data(y_values_runs, steps_runs, mode, seed_index)
            if steps is None or curve is None:
                print(f"Warning: Data not available for {data_type} {metric} in config {config_name}")
                continue
            plt.plot(steps, curve, label=config_name)
            if mode == "std" and stds is not None:
                plt.fill_between(steps,
                                 np.array(curve) - np.array(stds),
                                 np.array(curve) + np.array(stds),
                                 alpha=0.2)
        plt.xlabel("Steps")
        plt.title(f"{data_type.capitalize()} {metric.capitalize()} Across Configurations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f"{data_type}_{metric}.png"))
        plt.close()

def plot_single_configuration_metrics(config_name, metrics, save_directory, mode="mean", seed_index=0):
    """
    Generate plots for a single configuration, comparing train and test curves for loss and accuracy.
    
    Two figures are produced: one for loss and one for accuracy.
    
    Parameters:
        config_name (str): Name of the configuration.
        metrics (dict): Dictionary with keys 'train', 'test', and 'all_steps'.
        save_directory (str): Directory where the plots will be saved.
        mode (str): 'mean', 'std', or 'specific' for selecting the plot data.
        seed_index (int): Run index if mode is 'specific'.
    """
    os.makedirs(save_directory, exist_ok=True)
    steps_runs = metrics['all_steps']
    
    for metric in ['loss', 'accuracy']:
        plt.figure(figsize=(8, 6))
        # Get train data
        steps_train, curve_train, stds_train = get_plot_data(metrics['train'][metric], steps_runs, mode, seed_index)
        # Get test data
        steps_test, curve_test, stds_test = get_plot_data(metrics['test'][metric], steps_runs, mode, seed_index)
        if steps_train is None or curve_train is None or steps_test is None or curve_test is None:
            print(f"Warning: Missing data for {metric} in configuration {config_name}")
            continue
        plt.plot(steps_train, curve_train, label="Train", color='blue')
        plt.plot(steps_test, curve_test, label="Test", color='red')
        if mode == "std":
            plt.fill_between(steps_train,
                             np.array(curve_train) - np.array(stds_train),
                             np.array(curve_train) + np.array(stds_train),
                             color='blue', alpha=0.2)
            plt.fill_between(steps_test,
                             np.array(curve_test) - np.array(stds_test),
                             np.array(curve_test) + np.array(stds_test),
                             color='red', alpha=0.2)
        plt.xlabel("Steps")
        plt.ylabel(metric.capitalize())
        plt.title(f"Configuration: {config_name} - {metric.capitalize()} Over Steps")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f"{config_name}_{metric}.png"))
        plt.close()

def plot_best_across_configurations(results, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    
    best_metrics_data = {config_name: extract_best_train_and_test_metrics(config_data) for config_name, config_data in results.items()}
    config_names = sorted(best_metrics_data.keys())
    
    train_best_losses = [best_metrics_data[config]['train']['loss']['best_value'] for config in config_names]
    test_best_losses = [best_metrics_data[config]['test']['loss']['best_value'] for config in config_names]
    train_loss_steps = [best_metrics_data[config]['train']['loss']['best_step'] for config in config_names]
    test_loss_steps = [best_metrics_data[config]['test']['loss']['best_step'] for config in config_names]
    train_best_accuracies = [best_metrics_data[config]['train']['accuracy']['best_value'] for config in config_names]
    test_best_accuracies = [best_metrics_data[config]['test']['accuracy']['best_value'] for config in config_names]
    train_accuracy_steps = [best_metrics_data[config]['train']['accuracy']['best_step'] for config in config_names]
    test_accuracy_steps = [best_metrics_data[config]['test']['accuracy']['best_step'] for config in config_names]
    
    # Figure 1: Best Loss vs. Configuration (logarithmic y-axis)
    plt.figure(figsize=(8, 6))
    plt.plot(config_names, train_best_losses, marker='o', label='Train Loss')
    plt.plot(config_names, test_best_losses, marker='o', label='Test Loss')
    plt.yscale('log')
    plt.xlabel("Configuration")
    plt.ylabel("Best Loss (log scale)")
    plt.title("Best Loss vs. Configuration")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "best_loss.png"))
    plt.close()
    
    # Figure 2: Step at which Best Loss is reached vs. Configuration
    plt.figure(figsize=(8, 6))
    plt.plot(config_names, train_loss_steps, marker='o', label='Train Loss Step')
    plt.plot(config_names, test_loss_steps, marker='o', label='Test Loss Step')
    plt.xlabel("Configuration")
    plt.ylabel("Step of Best Loss")
    plt.title("Best Loss Step vs. Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "best_loss_step.png"))
    plt.close()
    
    # Figure 3: Best Accuracy vs. Configuration
    plt.figure(figsize=(8, 6))
    plt.plot(config_names, train_best_accuracies, marker='o', label='Train Accuracy')
    plt.plot(config_names, test_best_accuracies, marker='o', label='Test Accuracy')
    plt.xlabel("Configuration")
    plt.ylabel("Best Accuracy")
    plt.title("Best Accuracy vs. Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "best_accuracy.png"))
    plt.close()
    
    # Figure 4: Step at which Best Accuracy is reached vs. Configuration
    plt.figure(figsize=(8, 6))
    plt.plot(config_names, train_accuracy_steps, marker='o', label='Train Accuracy Step')
    plt.plot(config_names, test_accuracy_steps, marker='o', label='Test Accuracy Step')
    plt.xlabel("Configuration")
    plt.ylabel("Step of Best Accuracy")
    plt.title("Best Accuracy Step vs. Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "best_accuracy_step.png"))
    plt.close()