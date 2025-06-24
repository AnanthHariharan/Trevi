# meta_hrl/main.py
from trainers.baseline_trainer import train_baseline

if __name__ == "__main__":
    # For now, we just run the baseline trainer directly.
    # Later, this will parse configs to decide what to run.
    train_baseline()