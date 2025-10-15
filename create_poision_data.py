# create_poisoned_data.py
import os
import random
import shutil
from pathlib import Path
from typing import List

def copy_clean_to_attack(clean_root: Path, attack_root: Path):
    if attack_root.exists():
        print(f"Removing existing attack folder: {attack_root}")
        shutil.rmtree(attack_root)
    shutil.copytree(clean_root, attack_root)
    print(f"Copied clean dataset {clean_root} -> {attack_root}")

def poison_client_folder(client_folder: Path, flip_rate: float, num_labels: int = 10):
    """Flip labels for a fraction of images in a client folder on disk.
    Moves selected png files from their true-label subfolder into a wrong-label subfolder.
    """
    # Gather label directories that are numeric (0..9)
    label_dirs = [d for d in client_folder.iterdir() if d.is_dir() and d.name.isdigit()]
    for label_dir in label_dirs:
        img_files = [f for f in label_dir.iterdir() if f.suffix.lower() == ".png"]
        if not img_files:
            continue
        k = int(len(img_files) * flip_rate)
        if k == 0:
            continue
        to_flip = random.sample(img_files, k)
        orig_label = int(label_dir.name)
        for img_path in to_flip:
            # choose a wrong label uniformly at random
            wrong_label = random.choice([l for l in range(num_labels) if l != orig_label])
            dst_dir = client_folder / str(wrong_label)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / img_path.name
            # Move file into the wrong label folder
            img_path.rename(dst_path)

def create_attack_dataset(clean_root: str, attack_root: str, poisoned_clients: List[int], flip_rate: float):
    random.seed(42)  # reproducibility
    clean_root = Path(clean_root)
    attack_root = Path(attack_root)
    if not clean_root.exists():
        raise FileNotFoundError(f"Clean dataset root not found: {clean_root}")

    # Copy entire dataset
    copy_clean_to_attack(clean_root, attack_root)

    # Poison selected clients inside the copied folder
    for cid in poisoned_clients:
        client_folder = attack_root / f"client_{cid}"
        if not client_folder.exists():
            raise FileNotFoundError(f"Client folder not found in attack copy: {client_folder}")
        print(f"Poisoning client_{cid} at rate {flip_rate}")
        poison_client_folder(client_folder, flip_rate)

    print(f"✅ Attack dataset created at {attack_root} with poisoned clients {poisoned_clients} at flip_rate={flip_rate}")

if __name__ == "__main__":
    # CONFIGURE THESE
    CLEAN_ROOT = "femnist/dataset"         # path to your original dataset (keep original safe)
    POISONED_CLIENTS = [2, 5, 7, 10]       # four clients to poison (change as desired)
    FLIP_RATES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]  # rates to generate datasets for

    for rate in FLIP_RATES:
        # create attack folder name e.g. dataset_attack_25 for 25%
        pct = int(rate * 100)
        attack_root = f"femnist/dataset_attack_{pct}"
        create_attack_dataset(CLEAN_ROOT, attack_root, POISONED_CLIENTS, flip_rate=rate)
