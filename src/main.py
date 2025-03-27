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
from plots import plot_all_configurations, plot_configuration_metrics, plot_comparative_best_metrics

import torch

def train_model(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)
    
    # Create a directory to save the experiment results
    checkpoint_path = os.path.join(args.log_dir, str(args.exp_name), str(args.exp_id))
    os.makedirs(checkpoint_path, exist_ok=True)

    # Data
    (dataset, _), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(args.p, args.p, args.operator, 1.0, args.operation_orders, is_symmetric=False, shuffle=True, seed=args.seed)
    
    dataset_per_oders = {}
    for op in args.operation_orders:
        dataset_per_oders[op] = torch.utils.data.Subset(dataset,[i for i in range(len(dataset)) if dataset[i][2] == math.ceil(op/2)+op])
        dataset_per_oders[op] = torch.utils.data.random_split(dataset_per_oders[op], [args.r_train, 1-args.r_train])

    train_dataset = torch.utils.data.ConcatDataset([dataset_per_oders[op][0] for op in args.operation_orders])
    train_dataloader = DataLoader(train_dataset, batch_size=min(args.train_batch_size, len(train_dataset)), shuffle=True, num_workers=args.num_workers)

    eval_train_loaders = {}
    eval_test_loaders = {}
    for op in args.operation_orders:
        eval_train_loaders[op] = DataLoader(dataset_per_oders[op][0], batch_size=min(args.eval_batch_size, len(dataset_per_oders[op][0])), shuffle=False, num_workers=args.num_workers)
        eval_test_loaders[op] = DataLoader(dataset_per_oders[op][1], batch_size=min(args.eval_batch_size, len(dataset_per_oders[op][1])), shuffle=False, num_workers=args.num_workers)

    # Model
    vocabulary_size = len(tokenizer)
    if args.model == "lstm": model = LSTMLM(vocabulary_size=vocabulary_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, padding_index=padding_index, bias_lstm=True, bias_classifier=args.bias_classifier, share_embeddings=args.share_embeddings)
    elif args.model == "gpt": model = GPT(num_heads=args.num_heads, num_layers=args.num_layers, embedding_size=args.embedding_size, vocabulary_size=vocabulary_size, sequence_length=MAX_LENGTH, multiplier=4, dropout=args.dropout, non_linearity="gelu", padding_index=padding_index, bias_attention=True, bias_classifier=args.bias_classifier, share_embeddings=args.share_embeddings)
    else: raise ValueError("Unknown model {0}".format(args.model))
    model = model.to(args.device)

    # Optimizer
    if args.optimizer == "adamw": optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam": optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd": optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum": optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else: raise ValueError("Unknown optimizer {0}".format(args.optimizer))

    # Learning rate scheduler
    if args.scheduler == "dummy": scheduler = DummyScheduler(optimizer) # Dummy scheduler that does nothing
    elif args.scheduler == "step": scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif args.scheduler == "cosine": scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.n_steps + len(train_dataloader) - 1) // len(train_dataloader), eta_min=1e-5)
    elif args.scheduler == "plateau": scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    else: raise ValueError("Unknown scheduler {0}".format(args.scheduler))

    # Print parameters
    if args.verbose :
        print("=="*20)
        for k, v in vars(args).items(): print(k, ":", v)
        print("checkpoint_path:", checkpoint_path)
        print("dataset_size:", dataset.tensors[0].shape[0])
        print("=="*20)

    # Train    
    all_metrics = train(model, args, checkpoint_path, optimizer, scheduler,
        train_dataloader, eval_train_loaders, eval_test_loaders)
    
    return all_metrics, checkpoint_path

def train_models(args, seeds:list=[0, 42], rseeds:int=0):
    """Train a model with different seeds and plot the loss and accuracies of each separately."""
    assert seeds is not None or rseeds > 0, "Either M or seeds should be provided."

    # If rseeds > 0 then generate rseeds random seeds and concatenate them with seeds
    if rseeds > 0: seeds = seeds + [random.randint(0, 10000) for _ in range(rseeds)]

    print(f"Training model {args.exp_name}")
    run_paths = []
    for m, seed in enumerate(seeds):
        print(f"({m+1}/{len(seeds)}) Running training for seed {seed}")
        args.exp_id = m # Set the experiment id
        args.seed = seed # Set the seed
        _, run_path = train_model(args)
        run_paths.append(run_path)

    return run_paths

if __name__ == "__main__":
    logdir = "logs/new/experiment2"
    all_metrics = load_experiment(logdir, verbose=False)
    for model_name, model_metrics in all_metrics.items():
        model_path = os.path.join(logdir, model_name)
        configs_label = []
        for config_name, config_metrics in model_metrics.items():
            # Extracting r value from config name
            configs_label.append(float(config_name.split("_")[-1]))
            config_path = os.path.join(model_path, config_name)
            plot_configuration_metrics(config_metrics, config_path, "std")
        plot_all_configurations(model_metrics, model_path, "std", config_labels=configs_label)
        plot_comparative_best_metrics(model_metrics, model_path, config_title="Configuration (r_train)", config_labels=configs_label)