import numpy as np

def compute_mean_and_std_across_runs(metric_values):
    """
    Compute the mean and standard deviation across runs for each time step.
    
    Parameters:
        metric_values (list of lists): Each inner list is a metric curve from one run.
        
    Returns:
        tuple: (mean_array, std_array) computed element-wise.
    """
    metric_array = np.array(metric_values)
    mean_array = np.mean(metric_array, axis=0)
    std_array = np.std(metric_array, axis=0)
    return mean_array, std_array

def find_best_metric_across_runs(metric_values, steps_values, mode='min'):
    """
    Find the best (lowest for loss or highest for accuracy) metric value and its corresponding step
    across all runs.
    
    Parameters:
        metric_values (list of lists): Each inner list is a metric curve for one run.
        steps_values (list of lists): Each inner list contains step values for one run.
        mode (str): Use 'min' for loss (lower is better) or 'max' for accuracy.
        
    Returns:
        dict: Dictionary with keys 'best_value', 'best_step', and 'run_index'.
    """
    best_value = None
    best_step = None
    best_run_index = None

    for i, (values, steps) in enumerate(zip(metric_values, steps_values)):
        values = np.array(values)
        steps = np.array(steps)
        idx = np.argmin(values) if mode == 'min' else np.argmax(values)
        candidate = values[idx]
        if best_value is None or (mode == 'min' and candidate < best_value) or (mode == 'max' and candidate > best_value):
            best_value = candidate
            best_step = steps[idx]
            best_run_index = i

    return {'best_value': best_value, 'best_step': best_step, 'run_index': best_run_index}

def extract_best_train_and_test_metrics(config_metrics):
    """
    Extract the best loss and accuracy (and their corresponding steps) for both the train and test sets.
    
    Parameters:
        config_metrics (dict): Expected to have keys 'train', 'test', and 'all_steps', where
                               'train' and 'test' are dicts with keys 'loss' and 'accuracy'.
    
    Returns:
        dict: Best metrics structured as:
              {'train': {'loss': {...}, 'accuracy': {...}},
               'test': {'loss': {...}, 'accuracy': {...}}}
    """
    best_results = {}
    best_results['train'] = {
        'loss': find_best_metric_across_runs(config_metrics['train']['loss'], config_metrics['all_steps'], mode='min'),
        'accuracy': find_best_metric_across_runs(config_metrics['train']['accuracy'], config_metrics['all_steps'], mode='max')
    }
    best_results['test'] = {
        'loss': find_best_metric_across_runs(config_metrics['test']['loss'], config_metrics['all_steps'], mode='min'),
        'accuracy': find_best_metric_across_runs(config_metrics['test']['accuracy'], config_metrics['all_steps'], mode='max')
    }
    return best_results

def compute_convergence_steps(metric_values, steps_values, mode='min', threshold=0.95):
    """
    Compute the convergence step for each run. For a given run, convergence is defined as the
    step at which the metric first reaches a specified fraction of the total improvement.
    
    Parameters:
        metric_values (list of lists): Metric curves for each run.
        steps_values (list of lists): Corresponding step values for each run.
        mode (str): 'min' for loss or 'max' for accuracy.
        threshold (float): Fraction (e.g., 0.95) of the total improvement.
        
    Returns:
        list: Convergence step for each run (or None if the threshold is never reached).
    """
    convergence_steps = []
    for values, steps in zip(metric_values, steps_values):
        values = np.array(values)
        steps = np.array(steps)
        initial = values[0]
        if mode == 'min':
            best = np.min(values)
            target = initial - threshold * (initial - best)
            indices = np.where(values <= target)[0]
        else:
            best = np.max(values)
            target = initial + threshold * (best - initial)
            indices = np.where(values >= target)[0]
        convergence_steps.append(steps[indices[0]] if len(indices) > 0 else None)
    return convergence_steps

def convert_to_float(val):
    """
    Convert a tensor-like object to a float if necessary.
    
    Parameters:
        val: A numeric value or tensor.
        
    Returns:
        float: The value as a float.
    """
    return val.item() if hasattr(val, "item") else val

def compute_mean_and_std_for_runs(y_values_runs):
    """
    Compute the mean and standard deviation for a metric across runs at each time step.
    
    Parameters:
        y_values_runs (list of lists): Metric values for each run.
        
    Returns:
        tuple: (list of means, list of standard deviations)
    """
    n_points = len(y_values_runs[0])
    means = []
    stds = []
    for i in range(n_points):
        values = [convert_to_float(run[i]) for run in y_values_runs]
        means.append(np.mean(values))
        stds.append(np.std(values))
    return means, stds

def get_plot_data(y_values_runs, steps_runs, mode, seed_index=0):
    """
    Prepare data for plotting a metric given a particular mode.
    
    Parameters:
        y_values_runs (list of lists): Metric curves for each run.
        steps_runs (list of lists): Step curves for each run.
        mode (str): One of 'mean', 'std', or 'specific'.
        seed_index (int): Run index to use in 'specific' mode.
        
    Returns:
        tuple: (steps, curve, stds) where stds is None if mode is not 'std'.
    """
    if mode in ['mean', 'std']:
        if not steps_runs:
            return None, None, None
        steps = steps_runs[0]  # Assumes all runs share the same step values.
        means, stds = compute_mean_and_std_for_runs(y_values_runs)
        return (steps, means, stds) if mode == 'std' else (steps, means, None)
    elif mode == 'specific':
        if seed_index < len(y_values_runs):
            steps = steps_runs[seed_index]
            specific_values = [convert_to_float(val) for val in y_values_runs[seed_index]]
            return steps, specific_values, None
        else:
            return None, None, None
    else:
        raise ValueError("Mode must be 'mean', 'std', or 'specific'.")
