import random
from pathlib import Path

# Paths in your deit/ folder
in_list    = Path("deit/clipart_train.txt")
train_out  = Path("deit/clipart_train_split.txt")
val_out    = Path("deit/clipart_val_split.txt")
val_frac   = 0.10  # 10% validation

# load & shuffle
lines = [l.strip() for l in in_list.read_text().splitlines() if l.strip()]
random.seed(42)
random.shuffle(lines)

# split
n_val = int(len(lines) * val_frac)
val, train = lines[:n_val], lines[n_val:]

# write out
train_out.write_text("\n".join(train) + "\n")
val_out.write_text(  "\n".join(val)   + "\n")

print(f"→ {len(train)} train, {len(val)} val")