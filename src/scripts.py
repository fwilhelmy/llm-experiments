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
from itertools import groupby

default_train_scale = (35, 0.2, 10, 300)
default_figures = [(m, k) for m in ['test', 'train'] for k in ['loss', 'accuracy']]

def plot_exp2():
    logdir = "logs/experiment2"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        for config_name, config_metrics in model_metrics.items():
            r = float(config_name.split("_")[-1])
            config_metrics['sorting_key'] = r
            config_metrics['label'] = f"r={r}"
            config_path = os.path.join(model_path, config_name)

            plot_config_loss_accs(config_metrics, config_path,
                                  file_name="single_config.png",
                                  loss_y_scale=False)
            
        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))

        plot_all_configs_metrics(sorted_model_metrics, model_path,
                                 file_name="all_configs.png",
                                 loss_y_scale=False)
        plot_all_configs_metrics(sorted_model_metrics, model_path,
                                figures=[[('train', 'accuracy'), ('test', 'accuracy')]],
                                axis_labels=[['Train', 'Test'], ['Accuracy']],
                                file_name="only_accs.png",
                                loss_y_scale=False)
        plot_all_configs_best_metrics(sorted_model_metrics, model_path,
                                      file_name="comparative_best_metrics.png",
                                      x_label_title="Configuration (r_train)",
                                      loss_y_scale=True)


def plot_exp3():
    logdir = "logs/experiment3"
    # For experiment 3, the metrics have no configuration keys.
    all_metrics = load_experiment(logdir, has_configs=False, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_metrics['label'] = model_name
        model_path = os.path.join(logdir, model_name)
        
        # Plot a single configuration.
        plot_config_loss_accs(model_metrics, model_path,
                              file_name="single_config_metrics.png",
                              loss_y_scale=True)
        
        plot_config_ops(model_metrics, model_path,
                          file_name="single_config_ops.png",
                          loss_y_scale=True)
        
    # Plot all configurations.
    plot_all_configs_metrics(all_metrics, logdir,
                             file_name="all_configs.png",
                             loss_y_scale=True)

def plot_exp4():
    logdir = "logs/experiment4"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        for config_name, config_metrics in model_metrics.items():
            config_specs = config_name.split("_")
            L, d = int(config_specs[2]), int(config_specs[4])
            config_metrics['sorting_key'] = (L, d)
            config_metrics['label'] = f"L={L}, d={d}"
            config_path = os.path.join(model_path, config_name)
            plot_config_loss_accs(config_metrics, config_path,
                                  file_name="single_config.png",
                                  loss_y_scale=True)
        
        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))

        # For each L, plot grouped configurations with different d values.
        grouping_key = lambda item: item[1]['sorting_key'][0]
        grouped_model_metrics = groupby(sorted_model_metrics.items(), key=grouping_key)
        for L, d_metrics in grouped_model_metrics:
            plot_all_configs_metrics(dict(d_metrics), model_path,
                                     file_name=f"L_{L}_grouped_metrics.png",
                                     loss_y_scale=True)
            
        plot_all_configs_best_metrics(model_metrics, model_path,
                                      file_name="exp4_comparative.png",
                                      loss_y_scale=True)

def plot_exp5():
    logdir = "logs/experiment5"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        for config_name, config_metrics in model_metrics.items():
            batch_size = int(config_name.split("_")[2])
            config_metrics['sorting_key'] = batch_size
            config_metrics['label'] = f"B={batch_size}"
            config_path = os.path.join(model_path, config_name)
            plot_config_loss_accs(config_metrics, config_path,
                                  file_name="single_config.png",
                                  loss_y_scale=True)

        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))

        plot_all_configs_metrics(sorted_model_metrics, model_path,
                                 file_name="all_configs.png", loss_y_scale=True)
        plot_all_configs_best_metrics(sorted_model_metrics, model_path,
                                      file_name="comparative_best_metrics.png",
                                      x_label_title="Configuration",
                                      loss_y_scale=True)

def plot_exp6():
    logdir = "logs/experiment6"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        for config_name, config_metrics in model_metrics.items():
            weight_decay = float(config_name.split("_")[2])
            config_metrics['sorting_key'] = weight_decay
            config_metrics['label'] = f"W={weight_decay}"
            config_path = os.path.join(model_path, config_name)
            plot_config_loss_accs(config_metrics, config_path,
                                  file_name="single_config.png",
                                  loss_y_scale=True)
            
        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))
        
        plot_all_configs_metrics(sorted_model_metrics, model_path,
                                 file_name="all_configs.png",
                                 loss_y_scale=True)
        plot_all_configs_best_metrics(sorted_model_metrics, model_path,
                                      file_name="comparative_best_metrics.png",
                                      x_label_title="Configuration",
                                      loss_y_scale=True)

if __name__ == "__main__":
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
    plot_exp6()
