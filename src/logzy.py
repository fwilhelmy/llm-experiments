import torch
import os
from collections import defaultdict

# --- Checkpoint Functions (Model + Optimizer) ---

def save_checkpoint(model, optimizer, filepath, verbose=True):
    """Save model and optimizer states to a file."""
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    if verbose: print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer, map_location='cpu', verbose=True):
    """Load model and optimizer states from a file and update them."""
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if verbose: print(f"Checkpoint loaded from {filepath}")
    return model, optimizer

# --- Metrics Functions ---

def save_metrics(metrics, filepath, verbose=True):
    """Save training metrics to a file."""
    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in metrics.items()}  # to avoid lambda issues
    torch.save(to_save, filepath)
    if verbose: print(f"Metrics saved to {filepath}")

def load_metrics(filepath, map_location='cpu', verbose=True):
    """Load training metrics from a file, or return None if missing."""
    if os.path.exists(filepath):
        metrics = torch.load(filepath, map_location=map_location)
        if verbose: print(f"Metrics loaded from {filepath}")
        return metrics
    else:
        if verbose: print(f"Metrics file not found at {filepath}")
        return None

# --- How to Update Your Code ---
#
# In your training loop, save metrics like this:
#
#     # Update your metrics dictionary
#     train_statistics = evaluate(model, train_loader_for_eval, device)
#     for k, v in train_statistics.items():
#         all_metrics["train"][k].append(v)
#
#     test_statistics = evaluate(model, test_loader, device)
#     for k, v in test_statistics.items():
#         all_metrics["test"][k].append(v)
#
#     all_metrics["all_steps"].append(cur_step)
#     all_metrics["steps_epoch"][cur_step] = epoch
#
#     # Save metrics separately:
#     save_metrics(all_metrics, f"{checkpoint_path}/{exp_name}_metrics.pth")
#
# To save a checkpoint (model & optimizer), call:
#
#     save_checkpoint(model, optimizer, f"{checkpoint_path}/checkpoint_step_{cur_step}.pth")
#
# To load, simply use:
#
#     model, optimizer = load_checkpoint(checkpoint_filepath, model, optimizer)
#     metrics = load_metrics(f"{checkpoint_path}/{exp_name}_metrics.pth")
