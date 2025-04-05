import os
import json
import numpy as np
import torch
from torch import nn

MODES = ['train', 'test']

def save_model(logdir, file_name, model, verbose=True) -> None:
    """ Save the model's state dictionary """
    os.makedirs(logdir, exist_ok=True)
    file_path = os.path.join(logdir, f"{file_name}.pth")
    torch.save(model, file_path)        
    if verbose: print(f"Model saved to {logdir}")

def load_model(logdir, model_name, map_location=None, verbose=True) -> nn.Module:
    """ Load the model's state dictionary """
    load_path = os.path.join(logdir, f"{model_name}.pth")
    # If map_location is None, will load onto the devices originally saved from. 
    model = torch.load(load_path, map_location=map_location, weights_only=False)
    if verbose: print(f"Model loaded from {logdir}")
    return model

def save_metrics(metrics, filename, verbose=True):
    with open(filename, 'w') as f: json.dump(metrics, f, indent=4)
    if verbose: print(f"Metrics saved to {filename}")

def load_metrics(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f: data = json.load(f)

    # Convert lists to numpy arrays
    for mode in MODES:
        if mode not in data: continue
        for metric, array in data[mode].items():
            data[mode][metric] = np.array(array)

    return data

def load_configuration(config_path, config_name):
    run_paths = sorted([
        os.path.join(config_path, d)
        for d in os.listdir(config_path)
        if os.path.isdir(os.path.join(config_path, d))
    ]) # Sort to keep the order of runs consistent
    if not run_paths: raise ValueError(f"No run directories found in {config_path}")
    
    runs_metrics = []
    for run_path in run_paths:
        metrics_file = f"{config_name}_metrics.json"
        metrics_path = os.path.join(run_path, metrics_file)
        if not os.path.exists(metrics_path): continue
        metrics = load_metrics(metrics_path)
        runs_metrics.append(metrics)
    if not runs_metrics: raise ValueError(f"No metrics found in any run within {config_path}")
    
    # Metrics that are the same for all runs
    merged = {
        key: runs_metrics[0][key] 
        for key in ['epochs', 'steps', 'operation_orders']
        if key in runs_metrics[0]
    }
    
    # Metrics that are unique per run
    merged.update({
        key: [m[key] for m in runs_metrics]
        for key in ['total_time', 'avg_step_time']
        if key in runs_metrics[0]
    })
    
    # Evaluation metrics
    for mode in MODES:
        merged[mode] = {}
        for metric_name in runs_metrics[0][mode]:
            # Stack the arrays along the first axis
            arrays = [m[mode][metric_name] for m in runs_metrics]
            merged[mode][metric_name] = np.stack(arrays, axis=0)

    return merged

def load_experiment(experiment_path, has_configs=True, verbose=True):
    metrics = {}
    for model_name in os.listdir(experiment_path):
        model_path = os.path.join(experiment_path, model_name)
        if not os.path.isdir(model_path): continue
        if has_configs:
            metrics[model_name] = {}
            for config_name in os.listdir(model_path):
                config_path = os.path.join(model_path, config_name)
                if not os.path.isdir(config_path): continue
                try:
                    metrics[model_name][config_name] = load_configuration(config_path, config_name)
                except Exception as e:
                    if verbose: print("Error loading configuration metrics:", e)

                if verbose: print(f"Loaded configuration {config_name} metrics")
        else:
            metrics[model_name] = load_configuration(model_path, model_name)
            if verbose: print(f"Loaded model {model_name} metrics")
    return metrics

# Use "min" for loss (lower is better) or "max" for accuracy
def to_best_metrics(metrics, mode='max'):
    best_fn = np.argmax if mode == 'max' else np.argmin
    best_idx = best_fn(metrics)
    return (best_idx, metrics[best_idx])

from pprint import pprint
if __name__ == '__main__':
    experiment_dir = "logs/experiment1"
    metrics = load_experiment(experiment_dir, has_configs=False)
    pprint(metrics, depth=2)

    lstm_train_metrics = metrics['lstm']['train']['accuracy']
    mean_per_op = np.mean(lstm_train_metrics, axis=0)
    std_per_op  = np.std(lstm_train_metrics, axis=0)
    mean_overall = np.mean(lstm_train_metrics, axis=(0, 1))
    std_overall  = np.std(lstm_train_metrics, axis=(0, 1))

    best_idx = np.unravel_index(np.argmax(lstm_train_metrics), lstm_train_metrics.shape)
    best_acc = lstm_train_metrics[best_idx]
    best_eval = best_idx[2]
    print(f"Best training accuracy: {best_acc:.2f} at evaluation {best_eval}")