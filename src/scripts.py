#import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import random
import math

from data import get_arithmetic_dataset
from lstm import LSTMLM
from gpt import GPT
from train import train
from utils import seed_experiment
from arguments import Arguments
from schedulers import DummyScheduler
from logzy import load_experiment
from plots import *

def plot_exp2():
    logdir = "logs/experiment2"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        configs_label = []
        for config_name, config_metrics in model_metrics.items():
            # Extracting r value from config name
            configs_label.append(float(config_name.split("_")[-1]))
            config_path = os.path.join(model_path, config_name)
            plot_configuration_metrics(config_metrics, config_path, "std")
        
        plot_all_configurations(model_metrics, model_path, "std", config_labels=configs_label, symlog=True)
        plot_comparative_best_metrics(model_metrics, model_path, config_title="Configuration (r_train)", config_labels=configs_label)

def plot_exp3():
    logdir = "logs/experiment3"
    all_metrics = load_experiment(logdir, has_configs=False, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        plot_configuration_metrics(model_metrics, model_path, "std")
        plot_exp3_ops_combined(model_metrics, model_path)
    plot_all_configurations(all_metrics, logdir, "std")

def plot_exp4():
    logdir = "logs/experiment4"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        configs_label = []
        formated_metrics = {}
        for config_name, config_metrics in model_metrics.items():
            config_specs = config_name.split("_")
            L, d = int(config_specs[2]), int(config_specs[4])
            if L not in formated_metrics: formated_metrics[L] = {}
            formated_metrics[L][d] = config_metrics
            configs_label.append(f"L={L}, d={d}")
            
            config_path = os.path.join(model_path, config_name)
            plot_configuration_metrics(config_metrics, config_path, "std")
        
        for L, d_metrics in formated_metrics.items():
            # sort the d_metrics by the key
            d_metrics = dict(sorted(d_metrics.items(), key=lambda item: int(item[0])))
            plot_all_configurations_grouped(d_metrics, os.path.join(model_path,f"plots/{L}"), "std", config_labels=configs_label, symlog=True)

        # write a sorting function to sort by l and d
        plot_exp4_comparative(model_metrics, model_path, config_title="Configuration", config_labels=configs_label)
    
def plot_exp5():
    logdir = "logs/experiment5"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        configs_label = []
        for config_name, config_metrics in model_metrics.items():
            # Extracting batch size from config name formatted as 'modelName_B_BatchSize'
            parts = config_name.split("_")
            if len(parts) != 3:
                print(f"Unexpected config name format: {config_name}")
                continue
            batch_size = int(parts[2])
            configs_label.append(batch_size)
            
            config_path = os.path.join(model_path, config_name)
            plot_configuration_metrics(config_metrics, config_path, "std")
        
        plot_all_configurations(model_metrics, model_path, "std", config_labels=configs_label, symlog=True)
        plot_comparative_best_metrics(model_metrics, model_path, config_title="Configuration (Batch Size)", config_labels=configs_label)

if __name__ == "__main__":
    plot_exp5()
