import torch

class Arguments:
    # Data
    p: int = 31 # Must be a prime number if operator is "/"
    operator : str = "+" # ["+", "-", "*", "/"]
    r_train : float = .5
    operation_orders : int = [2] # [2], [3] or [2, 3]

    # Model
    model: str = 'lstm' # [lstm, gpt]
    num_heads: int = 4
    num_layers: int = 2
    embedding_size: int = 2**7
    hidden_size: int = 2**7
    dropout : float = 0.0
    share_embeddings : bool = False
    bias_classifier : bool = True

    # Optimization
    optimizer: str = 'adamw'  # [sgd, momentum, adam, adamw]
    scheduler: str = 'dummy'  # [dummy, step, cosine, plateau]
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-0

    # Training
    n_steps : int = 10**4 + 1
    eval_step: int = 100
    save_step: int = 1000

    # Dataloaders
    train_batch_size: int = 512
    eval_batch_size: int = 2**12
    num_workers: int = 0

    # Experiment & Miscellaneous
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id: int = 0
    exp_name: str = "default"
    log_dir: str = 'logs'
    seed: int = 42    
    verbose: bool = True