import os
from logzy import load_experiment, load_metrics, save_metrics
from plots import *
from itertools import groupby
from utils import slice_by_steps
from data import get_arithmetic_dataset
import math
from lstm import LSTMLM
from gpt import GPT
from arguments import Arguments
import random

# make a graphic as a function of parameters for exp4 ?????
# make the grouped batch size at different T for exp5 SO ONE PLOT PER T?
# There is no variance for the step part of plot_all_configs_best_metrics_sliced_by_steps

import os
import json
import numpy as np

def process_and_save_metrics_as_json(original_data, logdir, out_filename="processed_metrics.json", save=True):
    new_data = {}
    shared_keys = ['epochs', 'steps', 'operation_orders']
    shared = {}
    models_dict = {}
    first_shared_extracted = False

    for model_name, configs in original_data.items():
        models_dict[model_name] = {}
        for config_name, metrics in configs.items():
            # Extract shared keys from the first configuration encountered.
            if not first_shared_extracted:
                for key in shared_keys:
                    if key in metrics:
                        value = metrics[key]
                        shared[key] = value.tolist() if isinstance(value, np.ndarray) else value
                first_shared_extracted = True

            new_metrics = {}
            # Average avg_step_time and total_time over runs.
            for key in ['avg_step_time', 'total_time']:
                if key in metrics:
                    value = metrics[key]
                    # Assume value is a numpy array; compute the mean over the first axis.
                    new_metrics[key] = float(np.mean(value))
            
            # Process evaluation metrics in 'train' and 'test'.
            for mode in ['train', 'test']:
                new_metrics[mode] = {}
                if mode not in metrics:
                    continue
                for metric_name, metric_arr in metrics[mode].items():
                    # Compute mean over runs and op orders.
                    mean_metric = np.mean(metric_arr, axis=(0, 1))
                    new_metrics[mode][metric_name] = mean_metric.tolist()
                    # Determine best index: lower for loss, higher otherwise.
                    if "loss" in metric_name.lower():
                        best_idx = np.argmin(mean_metric)
                    else:
                        best_idx = np.argmax(mean_metric)
                    best_value = float(mean_metric[best_idx])
                    steps = metrics['steps']
                    best_step = steps[best_idx]
                    if isinstance(best_step, np.generic):
                        best_step = best_step.item()
                    new_metrics[mode]["best_" + metric_name] = {"value": best_value, "step": best_step}
            
            models_dict[model_name][config_name] = new_metrics

    new_data.update(shared)
    new_data["models"] = models_dict

    if save:
        os.makedirs(logdir, exist_ok=True)
        out_path = os.path.join(logdir, out_filename)
        with open(out_path, "w") as f:
            json.dump(new_data, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o, indent=2)
    return new_data

import os
import json
import numpy as np

import os
import json
import numpy as np

def process_and_save_metrics_as_json_keep_ops(original_data, logdir, out_filename="processed_metrics.json"):
    new_data = {}
    shared_keys = ['epochs', 'steps', 'operation_orders']
    shared = {}
    models_dict = {}
    first_shared_extracted = False

    for model_name, configs in original_data.items():
        models_dict[model_name] = {}
        for config_name, metrics in configs.items():
            if not first_shared_extracted:
                for key in shared_keys:
                    if key in metrics:
                        value = metrics[key]
                        shared[key] = value.tolist() if isinstance(value, np.ndarray) else value
                first_shared_extracted = True

            new_metrics = {}
            for key in ['avg_step_time', 'total_time']:
                if key in metrics:
                    value = metrics[key]
                    new_metrics[key] = float(np.mean(value))
            for mode in ['train', 'test']:
                new_metrics[mode] = {}
                if mode not in metrics:
                    continue
                for metric_name, metric_arr in metrics[mode].items():
                    mean_metric = np.mean(metric_arr, axis=0)  # shape: (ops, n_eval)
                    new_metrics[mode][metric_name] = mean_metric.tolist()
                    num_ops = mean_metric.shape[0]
                    best_values = []
                    best_steps = []
                    steps = metrics['steps']
                    for i in range(num_ops):
                        curve = mean_metric[i]
                        if "loss" in metric_name.lower():
                            best_idx = int(np.argmin(curve))
                        else:
                            best_idx = int(np.argmax(curve))
                        best_values.append(float(curve[best_idx]))
                        step_val = steps[best_idx]
                        if isinstance(step_val, np.generic):
                            step_val = step_val.item()
                        best_steps.append(step_val)
                    new_metrics[mode]["best_" + metric_name] = {"value": best_values, "step": best_steps}
            models_dict[model_name][config_name] = new_metrics

    new_data.update(shared)
    new_data["models"] = models_dict
    os.makedirs(logdir, exist_ok=True)
    out_path = os.path.join(logdir, out_filename)
    with open(out_path, "w") as f:
        json.dump(new_data, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o, indent=2)
    return new_data

# Example usage:
# processed = process_and_save_metrics_as_json_keep_ops(original_data, "/path/to/logdir")

figures_keys = [['train', 'test'], ['loss', 'accuracy', 'l2_norm']]

def plot_exp1():
    logdir = "logs/experiment1"
    all_metrics = load_experiment(logdir, has_configs=False, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        model_metrics['label'] = model_name

        # Question 4.1
        plot_config_loss_accs(model_metrics, model_path)

    # Question 4.2
    best_metrics = extract_config_best_metrics(all_metrics)
    best_metrics_path = os.path.join(logdir, "best_metrics.json")
    save_metrics(best_metrics, best_metrics_path, False)

    # Extra
    plot_all_configs_metrics2(all_metrics, logdir)

def plot_exp2():
    logdir = "logs/experiment2"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        # r 0.1 to 0.9
        for config_name, config_metrics in model_metrics.items():
            config_path = os.path.join(model_path, config_name)
            r = float(config_name.split("_")[-1])
            config_metrics['sorting_key'] = r
            config_metrics['label'] = f"r={r}"
            config_metrics['label_value'] = r

            # Plot for a single configuration.
            plot_config_loss_accs(config_metrics, config_path)
            
        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))

        # Question 4.3.a
        plot_all_configs_metrics_colobar(sorted_model_metrics, model_path, 
                                         colorbar_scale=False, colorbar=True, colorbar_title="r_train",
                                         file_name="all_metrics")
        # Question 4.3.b
        plot_all_configs_best_metrics(sorted_model_metrics, model_path, loss_y_scale=True,
                                      x_label_title="Configuration (r_train)",
                                      file_name="best_metrics")
        
    process_and_save_metrics_as_json(all_metrics, logdir, out_filename="processed_metrics.json")
        


def plot_exp3():
    logdir = "logs/experiment3"
    # For experiment 3, the metrics have no configuration keys.
    all_metrics = load_experiment(logdir, has_configs=False, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        model_metrics['label'] = model_name
        model_path = os.path.join(logdir, model_name)

        # Extra
        plot_config_loss_accs(model_metrics, model_path, file_name="extra_model_metrics")
        
        # Question 4.4.b
        # TODO CHOOSE WHICH ONE YOU WANT
        plot_config_ops(model_metrics, model_path, file_name="ops_metrics_v1")
        plot_config_ops2(model_metrics, model_path, file_name="ops_metrics_v2")
        
    # Question 4.4.a
    plot_all_configs_metrics2(all_metrics, logdir, file_name="all_metrics")

    # TODO
    
    process_and_save_metrics_as_json_keep_ops({model_name:{'multiops':model_metrics for model_name, model_metrics in all_metrics.items()}}, logdir, out_filename="processed_metrics.json")

def plot_exp4():
    logdir = "logs/experiment4"
    all_metrics = load_experiment(logdir, verbose=False)
    models_params = load_metrics(os.path.join(logdir, "models_params.json"))

    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        # L [1, 2, 3] X d [64, 128, 256]
        for config_name, config_metrics in model_metrics.items():
            config_path = os.path.join(model_path, config_name)
            config_specs = config_name.split("_")
            L, d = int(config_specs[2]), int(config_specs[4])
            params_num = models_params[model_name][config_name]
            config_metrics['sorting_key'] = (L, d)
            config_metrics['label'] = f"L={L}, d={d}\n({params_num})"
            config_metrics['label_value'] = d

            # Plot for a single configuration.
            plot_config_loss_accs(config_metrics, config_path)        
        
        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))

        # For each L, plot grouped configurations with different d values.
        grouping_key = lambda item: item[1]['sorting_key'][0]
        grouped_model_metrics = groupby(sorted_model_metrics.items(), key=grouping_key)
        grouped_model_metrics = {L: dict(d_metrics) for L, d_metrics in grouped_model_metrics}

        for L, d_metrics in grouped_model_metrics.items():
            # Question 5.a
            plot_all_configs_metrics_colobar_scale(d_metrics, model_path, colorbar_scale=False, colorbar=True, colorbar_title="d value", file_name=f"L_{L}_metrics")
        
        # Question 5.b
        # TODO CHOOSE WHICH ONE YOU WANT
        plot_all_configs_best_metrics(sorted_model_metrics, model_path, loss_y_scale=True,
                                      x_label_title="Configuration (L, d)", file_name="best_metrics_v1")
        plot_loss_and_accuracy(grouped_model_metrics, model_path, file_name="best_metrics_v2")
        
        for config_name, config_metrics in sorted_model_metrics.items():
            params_num = models_params[model_name][config_name]
            config_metrics['label'] = f"{params_num}"
            config_metrics['sorting_key'] = params_num
            config_metrics['label_value'] = params_num

        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))

        # Question 5.b
        # TODO WHAT TO SHOW??? IS THIS GOOD??
        plot_all_configs_best_metrics(sorted_model_metrics, model_path,
                                      file_name="params_all_metrics",
                                      x_label_title="Configuration (# Parameters)",
                                      loss_y_scale=True)
        
    process_and_save_metrics_as_json(all_metrics, logdir, out_filename="processed_metrics.json")

def plot_exp5():
    logdir = "logs/experiment5"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        # batch_size [32, 64, 128, 256, 512]
        for config_name, config_metrics in model_metrics.items():
            config_path = os.path.join(model_path, config_name)
            batch_size = int(config_name.split("_")[2])
            config_metrics['sorting_key'] = batch_size
            config_metrics['label'] = f"B={batch_size}"
            config_metrics['label_value'] = batch_size

            # Plot for a single configuration.
            plot_config_loss_accs(config_metrics, config_path, loss_y_scale=True)

        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))

        # Question 6.a
        plot_all_configs_metrics_colobar_scale(sorted_model_metrics, model_path,
                                               colorbar_scale=False, colorbar=True, colorbar_title="Batch Size",
                                               file_name="all_metrics")
        
        # Question 6.b
        plot_all_configs_best_metrics_sliced_by_steps(sorted_model_metrics, model_path, file_name="sliced_best_metrics",
                                      x_label_title="Configuration (Batch Size)", alphas=np.arange(0.1,1.1,0.1))
        
        # Extra
        plot_all_configs_best_metrics(sorted_model_metrics, model_path, file_name="extra_best_metrics",
                                      x_label_title="Configuration (Batch Size)")
        
        # Extra
        figures_keys = [['train'], ['l2_norm']]
        plot_all_configs_metrics_colobar_scale(sorted_model_metrics, model_path,
                                                colorbar_scale=False, colorbar=True, colorbar_title="Batch Size",
                                                file_name="l2norms_only", figures=[[(m, k) for m in figures_keys[0]] for k in figures_keys[1]],
                                                axis_labels=[[''], ['l2_norm']])
    sliced_data = {model_name:
                   {f"{config_name}_{round(slice_name,1)}": slice_metrics
                    for slice_name, slice_metrics 
                    in zip(
                        np.arange(0.1,1.1,0.1), 
                        slice_by_steps(config_metrics, np.arange(0.1,1.1,0.1))
                        ) 
                    for config_name, config_metrics in model_metrics.items()} 
                for model_name, model_metrics in all_metrics.items()}
    processed_data = process_and_save_metrics_as_json(sliced_data, logdir, out_filename="processed_metrics.json")
    [processed_data['models'][model_name][config_name][mode].pop(metric) 
                    for mode in ['train', 'test']
                    for metric in ['accuracy', 'loss', 'l2_norm']
                    for model_name, model_metrics in processed_data['models'].items() 
                    for config_name, config_metrics in model_metrics.items()]
    processed_data.pop('steps')
    processed_data.pop('epochs')
    processed_data.pop('operation_orders')
    #save
    out_path = os.path.join(logdir, "processed_metrics.json")
    with open(out_path, "w") as f:
        json.dump(processed_data, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o, indent=2)



def plot_exp6(): 
    logdir = "logs/experiment6"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        # weight_decay [0.25, 0.5, 0.75, 1.0]
        for config_name, config_metrics in model_metrics.items():
            config_path = os.path.join(model_path, config_name)
            weight_decay = float(config_name.split("_")[2])
            config_metrics['sorting_key'] = weight_decay
            config_metrics['label'] = f"W={weight_decay}"
            config_metrics['label_value'] = weight_decay

            # Plot for a single configuration.
            plot_config_loss_accs(config_metrics, config_path, loss_y_scale=True)
            
        # Sort config labels for consistent ordering in plots
        sorting_key = lambda item: item[1]['sorting_key']
        sorted_model_metrics = dict(sorted(model_metrics.items(), key=sorting_key))
        
        # Question 7.a
        plot_all_configs_metrics_colobar(sorted_model_metrics, model_path, file_name="all_metrics",
                                 figures=[[(m, k) for m in figures_keys[0]] for k in figures_keys[1]],
                                 axis_labels=figures_keys, colorbar_scale=False, colorbar=True, colorbar_title="Weight Decay")
        
        # Question 7.b
        plot_all_configs_best_metrics(sorted_model_metrics, model_path, loss_y_scale=True,
                                      x_label_title="Configuration (Weight Decay)",
                                      file_name="best_metrics")
        
    process_and_save_metrics_as_json(all_metrics, logdir, out_filename="processed_metrics.json")

def load_model(filepath, args, map_location='cpu'):
    # Data
    (dataset, _), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(args.p, args.p, args.operator, 1.0, args.operation_orders, is_symmetric=False, shuffle=False, seed=args.seed)

    dataset_per_oders = {}
    for op in args.operation_orders:
        dataset_per_oders[op] = [dataset[i] for i in range(len(dataset)) if dataset[i][2] == math.ceil(op/2)+op]

    # Model initialization
    vocabulary_size = len(tokenizer)
    if args.model == "lstm": model = LSTMLM(vocabulary_size=vocabulary_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, padding_index=padding_index, bias_lstm=True, bias_classifier=args.bias_classifier, share_embeddings=args.share_embeddings)
    elif args.model == "gpt": model = GPT(num_heads=args.num_heads, num_layers=args.num_layers, embedding_size=args.embedding_size, vocabulary_size=vocabulary_size, sequence_length=MAX_LENGTH, multiplier=4, dropout=args.dropout, non_linearity="gelu", padding_index=padding_index, bias_attention=True, bias_classifier=args.bias_classifier, share_embeddings=args.share_embeddings)
    else: raise ValueError("Unknown model {0}".format(args.model))
    model = model.to(args.device)

    # Loading the model
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['model_state'])
    return model, dataset_per_oders, tokenizer

def plot_exp7():
    # Model #1
    model_args1 = Arguments() # GPT - Sanity Check
    model_args1.model = "gpt"

    # Model #2
    model_args2 = Arguments() # GTP - Operation orders [2,3]
    model_args2.model = "gpt"
    model_args2.p = 11
    model_args2.operation_orders = [2, 3]

    # Model #3
    model_args3 = Arguments() # GTP - Deep Network
    model_args3.model = "gpt"
    model_args3.num_layers = 3
    model_args3.embedding_size = 2**8
    model_args3.hidden_size = 2**8

    models = {
        "Baseline": ("logs/experiment1/gpt/0/gpt_state.pth", model_args1),
        "BigData": ("logs/experiment3/gpt/0/gpt_state.pth", model_args2),
        "BigModel": ("logs/experiment4/gpt/gpt_L_3_d_256/0/gpt_L_3_d_256_state.pth", model_args3),
    }

    random.seed(42)
    samples_idx = random.sample(range(0, 100), 3)

    for model_name, model_data in models.items():
        model_path, model_args = model_data
        model, dataset_per_oders, tokenizer = load_model(model_path, model_args)

        for op in model_args.operation_orders:
            samples = [dataset_per_oders[op][idx] for idx in samples_idx]

            visualize_attention_matshow(
                model, 
                samples, 
                tokenizer, f"logs/experiment7/gpt/{model_name}", 
                f"{model_name}_{op}_attention")

if __name__ == "__main__":
    # plot_exp1()
    # plot_exp2()
    # plot_exp3()
    plot_exp4()
    # plot_exp5()
    # plot_exp6()
    # plot_exp7()