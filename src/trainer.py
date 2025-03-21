import torch

from tqdm import tqdm
import os
from collections import defaultdict
from utils import get_loss_and_accuracy

@torch.no_grad()
def eval_model(model, loader, device):
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
    
def train(
    model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device, 
    exp_name: str, checkpoint_path: str,
    n_epochs: int, eval_step: int = 1000, save_step: int = 1000, verbose=True
):
    # Create checkpoint directory if it doesn't exist.
    os.makedirs(checkpoint_path, exist_ok=True)

    # Determine total epochs based on n_steps and the number of batches per epoch.
    n_steps = n_epochs * len(train_loader)
    
    if verbose: print(f"Number of training epochs & steps: {n_epochs} {n_steps}")

    all_metrics = defaultdict(lambda: [])
    all_metrics["train"] = defaultdict(lambda: [])
    all_metrics["test"] = defaultdict(lambda: [])
    all_metrics["steps_epoch"] = {}

    # Evaluate model on training and test sets before training starts.
    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items():
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items():
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(0)
    all_metrics["steps_epoch"][0] = 0

    # Save initial model state.
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
  
    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    cur_step = 1

    # Early stopping parameters:
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    patience = 50  # Stop if no improvement for X epochs
    early_stopping = False

    # Main training loop (per epoch)
    pbar = tqdm(range(1, n_epochs+1), desc="Training", total=n_epochs)
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

            # (No per-batch scheduler update for StepLR; update scheduler once per epoch instead.)
            # if desired, you could update per batch for other scheduler types.
            # scheduler.step()
            # current_lr = scheduler.optimizer.param_groups[0]["lr"]
              
            # Evaluate model on training and test sets periodically.
            if cur_step in [1, n_steps] or cur_step % eval_step == 0:
                train_statistics = eval_model(model, train_loader_for_eval, device)
                for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, device)
                for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step)
                all_metrics["steps_epoch"][cur_step] = epoch

                if verbose:
                    to_print = "\n" + " | ".join(f"Train {k} : {v:.5f}" for k, v in train_statistics.items())
                    to_print += " | " + " | ".join(f"Test {k} : {v:.5f}" for k, v in test_statistics.items())
                    to_print += f" | lr = {current_lr:.5f}"
                    print(to_print)

            # Save model statistics & checkpoint periodically.
            if cur_step in [1, n_steps] or cur_step % save_step == 0:
                state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
                torch.save(state, f"{checkpoint_path}/{exp_name}_step={cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")

                to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()}  # to avoid lambda issues
                torch.save(to_save, f"{checkpoint_path}/{exp_name}_stats.pth")

            # update cur step to pbar
            pbar.set_postfix({'step': cur_step, 'loss': loss.item(), 'lr': current_lr})

            cur_step += 1

        # We update the learning rate scheduler once per epoch.
        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]["lr"]

        # -------------------------
        # Early Stopping Logic
        # -------------------------
        if early_stopping:
            # Evaluate on validation set to check for improvement
            test_statistics = eval_model(model, test_loader, device)
            current_val_loss = test_statistics["loss"]

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # If no improvement for 'patience' epochs, break
            if epochs_since_improvement >= patience:
                if verbose:
                    print(f"Early stopping triggered: No improvement in validation loss for {patience} epochs.")
                break
        # If early_stopping=False, we simply continue training through all epochs

    # Save final model state after training.
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
    
    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(cur_step)
    all_metrics["steps_epoch"][cur_step] = epoch

    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

    return all_metrics