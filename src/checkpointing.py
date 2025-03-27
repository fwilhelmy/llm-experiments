# Old code to manage saves, everything is not managed in metrics.py

import torch
import numpy as np
import re
import os
from tqdm import tqdm
from logzy import save_metrics

MODEL_FILE_NAME_REGEX = rf'_state\.pth$'
MODEL_FILE_NAME_REGEX_MATCH = rf'_state_step=(\d+)_acc=([\d.eE+-]+)_loss=([\d.eE+-]+)\.pth$'

def sorted_nicely(l): 
    """ 
    Sort the given iterable in the way that humans expect.
    https://stackoverflow.com/a/2669120/11814682
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)

def extract_metrics(file_name, exp_name=None):
    """
    Extract step, accuracy, and loss from the file name
    Args:
        file_name (str): The name of the file.
        exp_name (str): The name of the experiment.
    """
    pattern = '^' + ('(.*)' if exp_name is None else re.escape(exp_name)) + MODEL_FILE_NAME_REGEX_MATCH
    match_info = re.match(pattern, file_name)
    if match_info:
        i = 2 if exp_name is None else 1
        step = int(match_info.group(i))
        test_acc = float(match_info.group(i+1))
        test_loss = float(match_info.group(i+2))
        return step, test_acc, test_loss
    return None, None, None

def get_model_files(checkpoint_path, exp_name=None):
    """
    Get the model files in the given directory.
    Args:
        checkpoint_path (str): The path to the directory containing the checkpoints.
        exp_name (str): The name of the experiment.
    """
    pattern = f'^' + ('.*' if exp_name is None else re.escape(exp_name)) + MODEL_FILE_NAME_REGEX
    model_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)) and  re.match(pattern, f)]
    model_files = sorted_nicely(model_files)
    return model_files

def get_all_checkpoints(checkpoint_path, exp_name, just_files=False):
    """
    Load all the checkpoints from the given directory.
    Args:
        checkpoint_path (str): The path to the directory containing the checkpoints.
        exp_name (str): The name of the experiment.
        just_files (bool): If True, only the file paths will be returned. Otherwise, the models will be loaded.
    """

    model_files = get_model_files(checkpoint_path, exp_name)
    # Extract metrics from the file names
    metrics_dict = {f: extract_metrics(f, exp_name) for f in model_files} # {file : (step, test_acc, test_loss)}

    if exp_name is not None :
        statistics = torch.load(os.path.join(checkpoint_path, f"{exp_name}_metrics.pth"), map_location='cpu')
    else :
        statistics = None
    
    model_files = list(metrics_dict.keys())
    if just_files:
        return [os.path.join(checkpoint_path, f) for f in model_files], statistics

    all_models = {metrics_dict[f][0] : torch.load(os.path.join(checkpoint_path, f), map_location='cpu') for f in tqdm(model_files)}

    return all_models, statistics

def get_all_checkpoints_per_trials(all_checkpoint_paths, exp_name, just_files=False, verbose=False):
    
    all_models_per_trials = []
    all_statistics = []

    n_model = len(all_checkpoint_paths)
    for i, checkpoint_path in enumerate(all_checkpoint_paths) :
        if verbose : 
            print(checkpoint_path)
            #print(os.listdir(checkpoint_path))

        all_models, statistics = get_all_checkpoints(checkpoint_path, exp_name, just_files)
        all_models_per_trials.append(all_models)
        all_statistics.append(statistics)
    
    #metrics_names = list(statistics.keys())

    if len(all_statistics) > 0 :
        all_statistics_dic = {}
        #
        for key_1 in ['train', 'test']:
            all_statistics_dic[key_1] = {}
            for key in statistics[key_1].keys():
                all_statistics_dic[key_1][key] = [statistics[key_1][key] for statistics in all_statistics ]
        #
        for key in statistics.keys():
            if key in ['train', 'test'] : continue
            all_statistics_dic[key] = [statistics[key] for statistics in all_statistics ]
    else :
        all_statistics_dic = {}
    
    return all_models_per_trials, all_statistics_dic

def get_extrema_performance_steps(all_metrics, T_max=None):
    """
    Analyze the training and testing metrics to find the minimum train & test losses and 
    the maximum train & test accuracies, along with the steps at which they were achieved.
    
    Args:
        all_metrics (dict): A dictionary containing training and testing loss and accuracy data.
                            - all_metrics["train"]['loss'] : list of size T (training loss over time)
                            - all_metrics["test"]['loss'] : list of size T (testing loss over time)
                            - all_metrics["train"]['accuracy'] : list of size T (training accuracy over time)
                            - all_metrics["test"]['accuracy'] : list of size T (testing accuracy over time)
                            - all_metrics["all_steps"] : list of size T (list of step numbers)
        T_max (int): The maximum step to consider in the analysis. If None, all steps are considered.

    Returns:
        dict: A dictionary containing:
            - Minimum train loss and the first step at which it was achieved
            - Minimum test loss and the first step at which it was achieved
            - Maximum train accuracy and the first step at which it was achieved
            - Maximum test accuracy and the first step at which it was achieved
    """
    # Extract data from the dictionary
    steps = all_metrics["all_steps"]
    if T_max is not None :
        # The model is not evaluate every step, so if we fix a step T_max in range(n_steps),
        # We need to find the index i of T_max in "steps" : min i such that steps[i] <= T_max
        # Then we will consider the data from 0 to i
        #T_max_index  = next((i for i, step in enumerate(steps) if step > T_max), len(steps))
        T_max_index = (np.array(steps) <= T_max).sum()
    else :
        T_max_index = len(steps)
    steps = steps[:T_max_index]
    train_loss = all_metrics["train"]['loss'][:T_max_index]
    test_loss = all_metrics["test"]['loss'][:T_max_index]
    train_accuracy = all_metrics["train"]['accuracy'][:T_max_index]
    test_accuracy = all_metrics["test"]['accuracy'][:T_max_index]

    # Find the minimum train loss and the step at which it was achieved
    min_train_loss = min(train_loss)
    min_train_loss_step = steps[train_loss.index(min_train_loss)]

    # Find the minimum test loss and the step at which it was achieved
    min_test_loss = min(test_loss)
    min_test_loss_step = steps[test_loss.index(min_test_loss)]

    # Find the maximum train accuracy and the step at which it was achieved
    max_train_accuracy = max(train_accuracy)
    max_train_accuracy_step = steps[train_accuracy.index(max_train_accuracy)]

    # Find the maximum test accuracy and the step at which it was achieved
    max_test_accuracy = max(test_accuracy)
    max_test_accuracy_step = steps[test_accuracy.index(max_test_accuracy)]

    # Return the results in a dictionary
    return {
        "min_train_loss": min_train_loss,
        "min_train_loss_step": min_train_loss_step,
        "min_test_loss": min_test_loss,
        "min_test_loss_step": min_test_loss_step,
        "max_train_accuracy": max_train_accuracy,
        "max_train_accuracy_step": max_train_accuracy_step,
        "max_test_accuracy": max_test_accuracy,
        "max_test_accuracy_step": max_test_accuracy_step,
        "T_max_index" : T_max_index
    }


def get_extrema_performance_steps_per_trials(all_metrics, T_max=None):
    """
    Similar to get_extrema_performance_steps, but for multiple trials (compute the mean and std).

    Args:
        all_metrics (dict): A dictionary containing training and testing loss and accuracy data.
                            - all_metrics["train"]['loss'] : list of size T (training loss over time) x number of trials
                            - all_metrics["test"]['loss'] : list of size T (testing loss over time) x number of trials
                            - all_metrics["train"]['accuracy'] : list of size T (training accuracy over time) x number of trials
                            - all_metrics["test"]['accuracy'] : list of size T (testing accuracy over time) x number of trials
                            - all_metrics["all_steps"] : list of size T (list of step numbers) x number of trials
        T_max (int): The maximum step to consider in the analysis. If None, all steps are considered.
    """
    
    # Extract data from the dictionary
    all_steps = all_metrics["all_steps"] + []
    train_losses = all_metrics["train"]['loss'] + []
    test_losses = all_metrics["test"]['loss'] + []
    train_accuracies = all_metrics["train"]['accuracy'] + []
    test_accuracies = all_metrics["test"]['accuracy'] + []
    

    if T_max is not None :
        # The model is not evaluate every step, so if we fix a step T_max in range(n_steps),
        # We need to find the index i of T_max in "steps" : min i such that steps[i] <= T_max
        # Then we will consider the data from 0 to i
        T_max_indexes =[]
        for k in range(len(all_steps)) :
            T_max_index = (np.array(all_steps[k]) <= T_max).sum()
            T_max_indexes.append(T_max_index)
            all_steps[k] = all_steps[k][:T_max_index]
            test_losses[k] = test_losses[k][:T_max_index]
            train_losses[k] = train_losses[k][:T_max_index]
            test_accuracies[k] = test_accuracies[k][:T_max_index]
            train_accuracies[k] = train_accuracies[k][:T_max_index]
    else :
        T_max_indexes = [len(steps) for steps in all_steps]

    # Find the minimum train loss and the step at which it was achieved
    min_train_losses = [min(losses) for losses in train_losses]
    min_train_loss_steps = [steps[losses.index(min_loss)] for losses, min_loss, steps in zip(train_losses, min_train_losses, all_steps)]
    min_train_loss_mean = np.mean(min_train_losses)
    min_train_loss_std = np.std(min_train_losses)
    min_train_loss_step_mean = np.mean(min_train_loss_steps)
    min_train_loss_step_std = np.std(min_train_loss_steps)

    # Find the minimum test loss and the step at which it was achieved
    min_test_losses = [min(losses) for losses in test_losses]
    min_test_loss_steps = [steps[losses.index(min_loss)] for losses, min_loss, steps in zip(test_losses, min_test_losses, all_steps)]
    min_test_loss_mean = np.mean(min_test_losses)
    min_test_loss_std = np.std(min_test_losses)
    min_test_loss_step_mean = np.mean(min_test_loss_steps)
    min_test_loss_step_std = np.std(min_test_loss_steps)

    # Find the maximum train accuracy and the step at which it was achieved
    max_train_accuracies = [max(accs) for accs in train_accuracies]
    max_train_accuracy_steps = [steps[accs.index(max_acc)] for accs, max_acc, steps in zip(train_accuracies, max_train_accuracies, all_steps)]
    max_train_accuracy_mean = np.mean(max_train_accuracies)
    max_train_accuracy_std = np.std(max_train_accuracies)
    max_train_accuracy_step_mean = np.mean(max_train_accuracy_steps)
    max_train_accuracy_step_std = np.std(max_train_accuracy_steps)

    # Find the maximum test accuracy and the step at which it was achieved
    max_test_accuracies = [max(accs) for accs in test_accuracies]
    max_test_accuracy_steps = [steps[accs.index(max_acc)] for accs, max_acc, steps in zip(test_accuracies, max_test_accuracies, all_steps)]
    max_test_accuracy_mean = np.mean(max_test_accuracies)
    max_test_accuracy_std = np.std(max_test_accuracies)
    max_test_accuracy_step_mean = np.mean(max_test_accuracy_steps)
    max_test_accuracy_step_std = np.std(max_test_accuracy_steps)

    # Return the results in a dictionary
    return {
        "min_train_loss": min_train_loss_mean,
        "min_train_loss_std": min_train_loss_std,
        "min_train_loss_step": min_train_loss_step_mean,
        "min_train_loss_step_std": min_train_loss_step_std,
        "min_test_loss": min_test_loss_mean,
        "min_test_loss_std": min_test_loss_std,
        "min_test_loss_step": min_test_loss_step_mean,
        "min_test_loss_step_std": min_test_loss_step_std,
        "max_train_accuracy": max_train_accuracy_mean,
        "max_train_accuracy_std": max_train_accuracy_std,
        "max_train_accuracy_step": max_train_accuracy_step_mean,
        "max_train_accuracy_step_std": max_train_accuracy_step_std,
        "max_test_accuracy": max_test_accuracy_mean,
        "max_test_accuracy_std": max_test_accuracy_std,
        "max_test_accuracy_step": max_test_accuracy_step_mean,
        "max_test_accuracy_step_std": max_test_accuracy_step_std,
        "T_max_indexes" : T_max_indexes
    }

# Code to convert to the new metrics system the old data
def convert_old_metrics(runs_dirs, configuration_name):
    _, old_metrics = get_all_checkpoints_per_trials(runs_dirs, configuration_name, just_files=True, verbose=True)

    for run_id, run_dir in enumerate(runs_dirs):
        new_metrics = {}
        new_metrics["operation_orders"] = [2]
        new_metrics["steps"] = old_metrics["all_steps"][run_id]
        new_metrics["epochs"] = list(old_metrics["steps_epoch"][run_id].values())
        if "performance" in old_metrics:
            new_metrics["total_time"] = old_metrics["performance"][run_id]["total_elapsed"]
            new_metrics["avg_step_time"] = old_metrics["performance"][run_id]["step_time_avg"]
        new_metrics["train"] = {}
        new_metrics["test"] = {}
        for mode in ["train", "test"]:
            for key, data in old_metrics[mode].items():
                new_metrics[mode][key] = np.array([data[run_id]])  # shape: (1, n_evals)
        save_metrics(new_metrics, os.path.join(run_dir,f"{configuration_name}_metrics.json"))

def convert_old_experiments(experiments_dir, has_configs=True):
    for model_name in os.listdir(experiments_dir):
        model_path = os.path.join(experiments_dir, model_name)
        if not os.path.isdir(model_path): continue
        if has_configs:
            for config_name in os.listdir(model_path):
                config_path = os.path.join(model_path, config_name)
                if not os.path.isdir(config_path): continue
                runs_dirs = []
                old_filename = f"{config_name}_metrics.pth"
                for run in os.listdir(config_path):
                    run_path = os.path.join(config_path, run)
                    if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, old_filename)):
                        runs_dirs.append(run_path)
                convert_old_metrics(runs_dirs, config_name)
                print(f"Converted {config_name} metrics")
        else:
            runs_dirs = []
            old_filename = f"{model_name}_metrics.pth"
            for run in os.listdir(model_path):
                run_path = os.path.join(model_path, run)
                if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, old_filename)):
                    runs_dirs.append(run_path)
            convert_old_metrics(runs_dirs, model_name)
            print(f"Converted {model_name} metrics")