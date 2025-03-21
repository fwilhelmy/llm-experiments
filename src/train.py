#import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import time
import argparse

from data import get_arithmetic_dataset
from lstm import LSTMLM
from gpt import GPT
from trainer import train as train_model
from checkpointing import get_all_checkpoints_per_trials
from plotter import plot_loss_accs
from utils import seed_experiment
from arguments import Arguments
from schedulers import DummyScheduler

def train(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)
    
    # Create a directory to save the experiment results
    checkpoint_path = os.path.join(args.log_dir, str(args.exp_name), str(args.exp_id))
    os.makedirs(checkpoint_path, exist_ok=True)

    # Data
    (train_dataset, valid_dataset), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(args.p, args.p, args.operator, args.r_train, args.operation_orders, is_symmetric=False, shuffle=True, seed=args.seed)
    
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

    # Learning rate scheduler
    scheduler = DummyScheduler(optimizer) # Dummy scheduler that does nothing
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-4)

        ## Print parameters
    if args.verbose :
        print("=="*20)
        for k, v in vars(args).items(): print(k, ":", v)
        print("checkpoint_path:", checkpoint_path)
        print("dataset_size:", train_dataset.tensors[0].shape[0])
        print("=="*20)

    # Train    
    all_metrics = train_model(
        model, train_dataloader, train_dataloader_for_eval, valid_dataloader, optimizer, scheduler,
        args.device, 
        args.exp_name, checkpoint_path, 
        n_epochs=args.n_epochs,
        eval_step=args.eval_step,
        save_step=args.save_step,
        verbose=args.verbose
    )
    
    # Plots for the model
    plot_loss_accs(all_metrics, multiple_runs=False, log_x=False, log_y=False, fileName=args.exp_name, filePath=checkpoint_path, show=False)

    return all_metrics, checkpoint_path

def train_m_models(args, M:int=None, seeds:list=None):
    """Train M models and plot the loss and accuracies of each model separately."""
    assert M is not None or seeds is not None, "Either M or seeds should be provided."
    if seeds is not None: M = len(seeds)
    else: seeds = [args.seed + m if args.seed is not None else None for m in range(M)]
    
    all_checkpoint_paths = []

    for seed, m in zip(seeds, range(M)):
        print(f"Model {m+1}/{M}")
        args.exp_id = m # Set the experiment id
        args.seed = seed # Set the seed
        all_metrics, checkpoint_path = train(args) # Train the model
        all_checkpoint_paths.append(checkpoint_path)

    all_models_per_trials, all_metrics = get_all_checkpoints_per_trials(all_checkpoint_paths, args.exp_name, just_files=True, verbose=args.verbose)

    # Plots for all models
    plot_loss_accs(all_metrics, multiple_runs=True, log_x=False, log_y=False,fileName=f'{args.exp_name}_M={M}', filePath=args.log_dir, show=False)

    return all_models_per_trials, all_metrics, all_checkpoint_paths

if __name__ == "__main__":
    args = Arguments()
    args.n_epochs = 10000
    args.p = 53
    args.exp_name = "gpt3_53p"
    args.model = "gpt"

    all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=None)