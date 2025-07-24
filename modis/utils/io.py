import json
from pathlib import Path

import torch


def load_checkpoint(checkpoint_file: Path) -> dict:
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} doesn't exist.")
    checkpoint = torch.load(checkpoint_file)
    return checkpoint

def load_log(checkpoint_file: Path) -> list:
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Log file {checkpoint_file} doesn't exist.")
    with open(checkpoint_file, 'r', encoding='utf-8') as file:
        log = json.load(file)
    return log

