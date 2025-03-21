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
    n = 0

    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x) # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        n += batch_x.shape[0]
        loss += batch_loss.item() * batch_x.shape[0]
        acc += batch_acc * batch_x.shape[0]

    # Additional metrics can be added here (e.g., L2 norm of parameters)

    return {"loss" : loss / n, "accuracy": acc / n}

def run_evaluation(metrics, model, train_loader, test_loader, device, step = 0, epoch = 0):
    # Evaluate model on training and test sets before training starts.
    train_statistics = evaluate(model, train_loader, device)
    for k, v in train_statistics.items(): metrics["train"][k].append(v)

    test_statistics = evaluate(model, test_loader, device)
    for k, v in test_statistics.items(): metrics["test"][k].append(v)

    metrics["all_steps"].append(step)
    metrics["steps_epoch"][step] = epoch

    return train_statistics, test_statistics
    
def train(model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device, exp_name: str, checkpoint_path: str, n_steps: int, eval_step: int = 1000, save_step: int = 1000, verbose=True):
    # Create checkpoint directory if it doesn't exist.
    os.makedirs(checkpoint_path, exist_ok=True)

    # Determine total epochs based on n_steps and the number of batches per epoch.
    n_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    
    if verbose: print(f"Number of training epochs ({n_epochs}) & steps ({n_epochs * len(train_loader)})")

    all_metrics = defaultdict(lambda: [])
    all_metrics["train"] = defaultdict(lambda: [])
    all_metrics["test"] = defaultdict(lambda: [])
    all_metrics["steps_epoch"] = {}

    run_evaluation(all_metrics, model, train_loader_for_eval, test_loader, device)

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    cur_step = 1

    # Early stopping configuration
    # early_stopping = EarlyStopper(verbose=verbose)

    # Main training loop (per epoch)
    pbar = tqdm(range(1, n_epochs+1), desc="Training", total=n_epochs, bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]")
    for epoch in pbar:
        for i, batch in enumerate(train_loader):
            batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x) # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
              
            # Evaluate model on training and test sets periodically.
            if cur_step == 1 or (cur_step % eval_step == 0 and cur_step != n_steps):
                train_statistics, test_statistics = run_evaluation(all_metrics, model, train_loader_for_eval, test_loader, device, cur_step, epoch)

                if verbose:
                    to_print = "\n" + " | ".join(f"Train {k} : {v:.5f}" for k, v in train_statistics.items())
                    to_print += " | " + " | ".join(f"Test {k} : {v:.5f}" for k, v in test_statistics.items())
                    to_print += f" | lr = {current_lr:.5f}"
                    print(to_print)

            # Save model statistics & checkpoint periodically.
            if cur_step == 1 or (cur_step % save_step == 0 and cur_step != n_steps):
                file_name = f"{exp_name}_state_step={cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth"
                save_checkpoint(model, optimizer, os.path.join(checkpoint_path, file_name))

            # update tqdm progress bar
            pbar.set_postfix({'step': cur_step, 'loss': f"{loss.item():.5f}", 'lr': f"{current_lr:.3f}"})

            cur_step += 1

        # We update the learning rate scheduler once per epoch.
        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]["lr"]

        # if early_stopping.early_stop(evaluate(model, test_loader, device)['loss'], epoch): break

    # Save final model state after training.
    save_checkpoint(model, optimizer, f"{checkpoint_path}/{exp_name}_state.pth")
    
    run_evaluation(all_metrics, model, train_loader_for_eval, test_loader, device, cur_step-1, epoch)
    save_metrics(all_metrics, f"{checkpoint_path}/{exp_name}_metrics.pth")

    return all_metrics