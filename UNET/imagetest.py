import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

label_dir = Path("./keras_png_slices_data/keras_png_slices_seg_train")
counts = Counter()
for p in list(label_dir.rglob("*.png"))[:100]:
    arr = np.array(Image.open(p).convert("L"))
    for val, c in zip(*np.unique(arr, return_counts=True)):
        counts[int(val)] += int(c)
total = sum(counts.values())
for val, c in sorted(counts.items()):
    print(f"{val}: {c/total:.4%}")