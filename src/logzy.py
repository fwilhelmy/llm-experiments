import os
import json
import numpy as np

def save_metrics(metrics, filename, verbose=True):
    # Convert numpy arrays to lists for serialization
    for mode in ['test', 'train']:
        if mode not in metrics: continue
        for metric, array in metrics[mode].items():
            metrics[mode][metric] = array.tolist()

    with open(filename, 'w') as f: json.dump(metrics, f, indent=4)
    if verbose: print(f"Metrics saved to {filename}")

def load_metrics(filepath):
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as f: data = json.load(f)

    # Convert lists to numpy arrays
    for mode in ['train', 'test']:
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
    for mode in ['train', 'test']:
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
                metrics[model_name][config_name] = load_configuration(config_path, config_name)
                if verbose: print(f"Loaded configuration {config_name} metrics")
        else:
            metrics[model_name] = load_configuration(model_path, model_name)
            if verbose: print(f"Loaded model {model_name} metrics")
    return metrics

from pprint import pprint
if __name__ == '__main__':
    experiment_dir = "logs/experiment1"
    metrics = load_experiment(experiment_dir, has_configs=False)
    pprint(metrics, depth=2)