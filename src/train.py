import torch

from tqdm import tqdm
import os
from collections import defaultdict
from utils import get_loss_and_accuracy
from logzy import save_checkpoint, save_metrics
import numpy as np
import json

# From: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    """Early stopping with slope-based detection."""
    def __init__(self, patience=5, min_delta=0.001, warmup_epochs=100, window_size=10, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.window_size = window_size
        self.verbose = verbose
        self.counter = 0
        self.loss_history = []

    def update_history(self, validation_loss):
        self.loss_history.append(validation_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

    def compute_slope(self):
        if len(self.loss_history) < self.window_size: return None
        x = np.arange(self.window_size) # Epochs
        y = np.array(self.loss_history) # Validation loss
        A = np.vstack([x, np.ones(len(x))]).T # y = mx + c
        return np.linalg.lstsq(A, y, rcond=None)[0][0]

    def early_stop(self, validation_loss, epoch):
        self.update_history(validation_loss)

        # Skip early stopping during warmup.
        if epoch < self.warmup_epochs: return False

        if len(self.loss_history) == self.window_size:
            current_slope = self.compute_slope()
            if abs(current_slope) < self.min_delta:
                self.counter += 1
                if self.verbose: print(f"\nEpoch {epoch}: Plateau detected (slope {current_slope:.6f}). Counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience: return True
            else: self.counter = 0  # Reset counter if meaningful progress resumes.
        return False

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc = 0
    loss = 0
    l2_norm = 0
    n = 0

    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x) # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        n += batch_x.shape[0]
        loss += batch_loss.item() * batch_x.shape[0]
        acc += batch_acc * batch_x.shape[0]

        for p in model.parameters():
            if p.requires_grad:
                l2_norm += p.norm(2).item() ** 2  # Square the norm for each parameter tensor
        l2_norm = l2_norm ** 0.5  # Take the square root to get the overall ℓ₂ norm

    return loss / n, acc / n, l2_norm

def run_evaluation(all_metrics, model, train_loaders, test_loaders, device, step, epoch):
    all_metrics["steps"].append(step)
    all_metrics["epochs"].append(epoch)

    index_step = len(all_metrics["steps"]) - 1
    for index_op, op in enumerate(train_loaders):
        loss, acc, l2_norm = evaluate(model, train_loaders[op], device)
        all_metrics['train']['loss'][index_op, index_step] = loss
        all_metrics['train']['acc'][index_op, index_step] = acc.item()
        all_metrics['train']['l2_norm'][index_op, index_step] = l2_norm

    for index_op, op in enumerate(test_loaders):
        loss, acc, l2_norm = evaluate(model, train_loaders[op], device)
        all_metrics['test']['loss'][index_op, index_step] = loss
        all_metrics['test']['acc'][index_op, index_step] = acc.item()
        all_metrics['test']['l2_norm'][index_op, index_step] = l2_norm
    
def train(model, args, logdir, optimizer, scheduler, train_loader, eval_train_loaders, eval_test_loaders):   
    # Create checkpoint directory if it doesn't exist.
    os.makedirs(logdir, exist_ok=True)

    # Determine total epochs based on n_steps and the number of batches per epoch.
    n_epochs = (args.n_steps + len(train_loader) - 1) // len(train_loader)
    
    if args.verbose: print(f"Number of training epochs ({n_epochs}) & steps ({n_epochs * len(train_loader)})")

    # Lambda functions to compute the mean of a metric over all operation orders for one evaluation set.
    mean = lambda metrics: np.mean([metrics[op][len(all_metrics['steps']) - 1] for op in range(len(args.operation_orders))])
    # Lambda function to initialize the metrics array.
    n_evals = n_epochs * len(train_loader) // args.eval_step + 3
    init_metrics = lambda: {
        'loss': np.empty((len(args.operation_orders), n_evals)),
        'acc': np.empty((len(args.operation_orders), n_evals)),
        'l2_norm': np.empty((len(args.operation_orders), n_evals))
    }

    all_metrics = {'train': init_metrics(), 'test': init_metrics(), 'steps': [], 'epochs': [], 'operation_orders': args.operation_orders}

    run_evaluation(all_metrics, model, eval_train_loaders, eval_test_loaders, args.device, 0, 0)

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    cur_step = 1
    cur_metrics = {'train': {}, 'test': {}} # Store the last evaluation metrics for each set

    # Early stopping configuration
    # early_stopping = EarlyStopper(verbose=verbose)

    # Main training loop (per epoch)
    pbar = tqdm(range(1, n_epochs+1), desc="Training", total=n_epochs)
    for epoch in pbar:
        for i, batch in enumerate(train_loader):
            batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x) # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
              
            # Evaluate model on training and test sets periodically.
            if cur_step == 1 or (cur_step % args.eval_step == 0 and cur_step != args.n_steps):
                run_evaluation(all_metrics, model, eval_train_loaders, eval_test_loaders, args.device, cur_step, epoch)
                cur_metrics = {k: {l: mean(all_metrics[k][l]) for l in ['acc', 'loss']} for k in ['train', 'test']}

                if args.verbose:
                    to_print = "\n" 
                    to_print += " | ".join(f"Train {k} : {v:.5f}" for k, v in cur_metrics['train'].items())
                    to_print += " || "
                    to_print += " | ".join(f"Test {k} : {v:.5f}" for k, v in cur_metrics['test'].items())
                    to_print += f" || lr = {current_lr:.5f}"
                    print(to_print)

            # Save model statistics & checkpoint periodically.
            if cur_step == 1 or (cur_step % args.save_step == 0 and cur_step != args.n_steps):
                file_name = f"{args.exp_name}_state_step={cur_step}_acc={cur_metrics['test']['acc']:.5f}_loss={cur_metrics['test']['loss']:.5f}.pth"
                save_checkpoint(model, optimizer, os.path.join(logdir, file_name), verbose=args.verbose)

            # update tqdm progress bar
            pbar.set_postfix({'step': cur_step, 'loss': f"{loss:.5f}", 'lr': f"{current_lr:.3f}"})

            cur_step += 1

        # We update the learning rate scheduler once per epoch.
        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]["lr"]

        # if early_stopping.early_stop(evaluate(model, test_loader, device)['loss'], epoch): break

    # Compute the global average step time.
    total_time = pbar.format_dict["elapsed"] # total seconds for all epochs
    all_metrics["total_time"], all_metrics["avg_step_time"] = total_time, total_time / args.n_steps

    # Save final model state after training.
    save_checkpoint(model, optimizer, f"{logdir}/{args.exp_name}_state.pth", verbose=args.verbose)
    
    run_evaluation(all_metrics, model, eval_train_loaders, eval_test_loaders, args.device, cur_step, epoch)
    all_metrics['train'] = {k: np.array(v) for k, v in all_metrics['train'].items()}
    all_metrics['test'] = {k: np.array(v) for k, v in all_metrics['test'].items()}
    save_metrics(all_metrics, f"{logdir}/{args.exp_name}_metrics.pth", verbose=args.verbose)

    return all_metrics