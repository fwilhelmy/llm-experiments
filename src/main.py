#import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import random

from data import get_arithmetic_dataset
from lstm import LSTMLM
from gpt import GPT
from train import train
from checkpointing import get_all_checkpoints_per_trials
from plotter import plot_configuration, plot_all
from utils import seed_experiment
from arguments import Arguments
from schedulers import DummyScheduler

import torch

def train_model(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)
    
    # Create a directory to save the experiment results
    checkpoint_path = os.path.join(args.log_dir, str(args.exp_name), str(args.exp_id))
    os.makedirs(checkpoint_path, exist_ok=True)

    # Data
    if isinstance(args.operation_orders, int):
        (train_dataset, valid_dataset), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(args.p, args.p, args.operator, args.r_train, args.operation_orders, is_symmetric=False, shuffle=True, seed=args.seed)
    else:
        (dataset, _), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(args.p, args.p, args.operator, 1.0, args.operation_orders, seed=args.seed)
        vocabulary_size = len(tokenizer)
        dataset_per_oders = {
            2 : torch.utils.data.Subset(dataset,[i for i in range(len(dataset)) if dataset[i][2] == 3]), # a + b = r EOS PAD PAD
            3 : torch.utils.data.Subset(dataset,[i for i in range(len(dataset)) if dataset[i][2] == 5]) # a + b + c = r EOS
        }
        train_dataset_2, valid_dataset_2 = torch.utils.data.random_split(dataset_per_oders[2], [args.r_train, 1-args.r_train])
        train_dataset_3, valid_dataset_3 = torch.utils.data.random_split(dataset_per_oders[3], [args.r_train, 1-args.r_train])
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_2, train_dataset_3])
        valid_dataset = torch.utils.data.ConcatDataset([valid_dataset_2, valid_dataset_3])
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)))
        valid_dataset = torch.utils.data.Subset(valid_dataset, torch.randperm(len(valid_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=min(args.train_batch_size, len(train_dataset)), shuffle=True, num_workers=args.num_workers)
    train_dataloader_for_eval = DataLoader(train_dataset, batch_size=min(args.eval_batch_size, len(train_dataset)), shuffle=False, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=min(args.eval_batch_size, len(valid_dataset)), shuffle=False, num_workers=args.num_workers)

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
        print("train_dataset_size:", train_dataset.tensors[0].shape[0])
        print("valid_dataset_size:", train_dataset.tensors[0].shape[0])
        print("=="*20)

    # Train    
    all_metrics = train(
        model,
        train_dataloader, train_dataloader_for_eval, valid_dataloader,
        optimizer, scheduler, args.device, 
        args.exp_name, checkpoint_path, n_steps=args.n_steps,
        eval_step=args.eval_step, save_step=args.save_step,
        verbose=args.verbose
    )
    
    return all_metrics, checkpoint_path

def train_models(args, seeds:list=[0, 42], rseeds:int=0):
    """Train a model with different seeds and plot the loss and accuracies of each separately."""
    assert seeds is not None or rseeds > 0, "Either M or seeds should be provided."

    # If rseeds > 0 then generate rseeds random seeds and concatenate them with seeds
    if rseeds > 0: seeds = seeds + [random.randint(0, 10000) for _ in range(rseeds)]

    all_checkpoint_paths = []
    all_metrics = []
    print(f"Training model {args.exp_name}")
    for m, seed in enumerate(seeds):
        print(f"({m+1}/{len(seeds)}) Running training for seed {seed}")
        args.exp_id = m # Set the experiment id
        args.seed = seed # Set the seed
        _, checkpoint_path = train_model(args)
        all_checkpoint_paths.append(checkpoint_path)
        
    all_models_paths, all_metrics = get_all_checkpoints_per_trials(all_checkpoint_paths, args.exp_name, just_files=True, verbose=args.verbose)
    # Plots for all runs of one configuration
    save_path = os.path.join(args.log_dir, args.exp_name)
    plot_configuration(all_metrics, save_path=f"{save_path}", mode="std")

    return all_models_paths, all_metrics

if __name__ == "__main__":
    args = Arguments()
    args.log_dir = "logs/experiment1"
    args.n_steps = 50
    args.operation_orders = 2
    models = ["lstm"]
    results = {}
    for model in models:
        args.model = model
        
        args.exp_name = model
        results[model], _ = train_models(args, [0])

    plot_all(results, save_path=args.log_dir, mode="std")