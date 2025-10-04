# LLM Experiments

## Project overview
This repository contains the cleaned-up training code that Felix Wilhelmy used to study how lightweight language models learn arithmetic operations under different curricula.  The project trains either LSTM or GPT sequence models on synthetic datasets of arithmetic expressions and tracks accuracy, loss, and parameter norms for each operation order.  Sanitized reports documenting the experiments are available in the repository root as `FelixWilhelmy_IFT6135_HW2_Practical.pdf`, `FelixWilhelmy_IFT6135_HW2_Theory.pdf`, `IFT6135___HW__2_practical.pdf`, `IFT6135___HW__2_theory.pdf`, `LoRA.pdf`, and `QLoRA.pdf`.

## Getting started

### Clone the repository
```bash
git clone https://github.com/mila-iqia/llm-experiments.git
cd llm-experiments
```

### Set up a virtual environment and install dependencies
The training scripts in `src/` depend on PyTorch, NumPy, tqdm, and Gradescope utilities.  Install them with:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training workflows

### Train a single model run
Use the high-level helper in `src/main.py` to configure and launch one training run:
```python
from src.arguments import Arguments
from src.main import train_model

args = Arguments()
args.model = "gpt"          # or "lstm"
args.operator = "+"         # choose among ["+", "-", "*", "/"]
args.operation_orders = [2]  # operation depths to include
args.exp_name = "demo"
args.log_dir = "logs"

metrics, checkpoint_dir = train_model(args)
print("Metrics keys:", metrics.keys())
print("Checkpoints saved under:", checkpoint_dir)
```
This call constructs the datasets, initializes the selected model, and runs the optimizer loop defined in `src/train.py`, persisting checkpoints and metrics to `logs/<exp_name>/<exp_id>/`.

### Launch multiple seeds
For reproducibility studies, sweep over several random seeds with `train_models`:
```python
from src.arguments import Arguments
from src.main import train_models

args = Arguments()
args.exp_name = "seed_sweep"
run_paths = train_models(args, seeds=[0, 13, 42])
print("Saved runs:", run_paths)
```
Each entry in `run_paths` points to a directory populated by `src/train.py` with model snapshots and JSON metrics for later analysis.

## Reports and artifacts
The sanitized write-ups and ablation notes live alongside the source tree.  After running experiments you will also find generated checkpoints and metric JSON files in the `logs/` directory created by the training helpers above.

## Credits
Felix Wilhelmy thanks Professor Aaron Courville for his guidance and lab assistants Pascal Tikeng and Alireza Dizaji, who provided the initial project code that was heavily modified for this repository.
