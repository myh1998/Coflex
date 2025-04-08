import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.double
else:
    DEVICE = "cpu"
    DTYPE = torch.double

DEVICE = "cpu"
DTYPE = torch.double

tkwargs = {
    "dtype": DTYPE,
    "device": DEVICE,
}

def min_max_norm(array: np.ndarray, global_min: float = None, global_max: float = None) -> np.ndarray:
    if global_min is None:
        global_min = np.min(array)
    if global_max is None:
        global_max = np.max(array)
    if global_max == global_min:
        return np.zeros_like(array)
    return (array - global_min) / (global_max - global_min)

def log_transform(data):
    return np.log1p(data)

algo_files = {
    "qnehvi": (
        "your-path-to/Saving/qnehvi/DeFiNES_SSS_os_ALGO_qNEHVI_TIME__02_13_15_55_imagenet_/train_output.csv", 
        "green"
    ),
    "coflex": (
        "your-path-to/Saving/coflex_gen_2/DeFiNES_SSS_os_ALGO_Coflex_TIME__02_07_13_31_imagenet_/train_output.csv", 
        "blue"
    ),
    "qehvi": (
        "your-path-to/Saving/qehvi/DeFiNES_SSS_os_ALGO_qEHVI_TIME__02_15_09_54_imagenet_/train_output.csv", 
        "orange"
    ),
    "qnpargeo": (
        "your-path-to/Saving/qnpargeo/DeFiNES_SSS_os_ALGO_qNParEGO_TIME__02_09_20_01_imagenet_/train_output.csv", 
        "red"
    ),
    "pabo": (
        "your-path-to/Saving/pabo/DeFiNES_SSS_os_ALGO_pabo_TIME__03_03_10_25_imagenet_/train_output.csv", 
        "gray"
    ),
    "random": (
        "your-path-to/Saving/random/DeFiNES_SSS_os_ALGO_random_TIME__02_16_01_17_imagenet_/train_output.csv", 
        "gray"
    ),
}

all_err = []
all_eng = []
for algo, (file_path, color) in algo_files.items():
    df = pd.read_csv(file_path, header=None)
    all_err.extend(df[0].values)
    all_eng.extend(df[1].values)
global_min_err = np.min(all_err)
global_max_err = np.max(all_err)
global_min_eng = np.min(all_eng)
global_max_eng = np.max(all_eng)

print(f"Global Err: min = {global_min_err}, max = {global_max_err}")
print(f"Global EDP: min = {global_min_eng}, max = {global_max_eng}")

plt.figure(figsize=(10, 6))
global_min_distance = float("inf")
global_min_algo = None
global_min_point = None
min_point_list = []

for algo, (file_path, color) in algo_files.items():
    print(f"Processing algorithm: {algo}")
    df = pd.read_csv(file_path, header=None)
    y_err = df[0].values  
    y_eng = df[1].values
    y_err_norm = min_max_norm(y_err, global_min_err, global_max_err)
    y_eng_norm = min_max_norm(y_eng, global_min_eng, global_max_eng)

    distances_norm = np.sqrt(y_err_norm**2 + y_eng_norm**2)

    min_idx = np.argmin(distances_norm)
    min_distance = distances_norm[min_idx]
    min_point_original = (y_err[min_idx], y_eng[min_idx])

    if min_distance < global_min_distance:
        global_min_distance = min_distance
        global_min_algo = algo
        global_min_point = min_point_original

    plt.scatter(y_err, y_eng, color=color, label=algo, alpha=0.6)

    sorted_indices = np.argsort(distances_norm)
    top5_indices = sorted_indices[:5]
    print(f"Top 5 results for {algo}:")
    for rank, idx in enumerate(top5_indices, 1):
        err_val = y_err[idx]
        eng_val = y_eng[idx]
        dist_val = distances_norm[idx]
        print(f"  Rank {rank}: Index {idx}, Err: {err_val:.4f}, Energy: {eng_val:.4f}, Norm Distance: {dist_val:.4f}")

    min_point_list.append([min_point_original, color, algo, min_distance, min_idx])

for min_data in min_point_list:
    point_xy, c, name, dist_val, idx_val = min_data
    plt.scatter(*point_xy, color=c, edgecolors='black', s=50, marker='s',
                label=f"{name} min - {dist_val:.3f}")
    print(" ** ", point_xy, name, dist_val, idx_val)

if global_min_point:
    plt.scatter(*global_min_point, color='gold', edgecolors='black', s=200, marker='*',
                label=f"Global Min ({global_min_algo})")

plt.xlabel("Err (%)")
plt.ylabel("EDP (µJ*s)")
plt.title("Pareto Analysis with Global Normalized Distances")
plt.grid(True)
plt.legend()
plt.show()

print(f"Global Minimum Distance in normalized space: {global_min_distance:.4f}")
print(f"Global Min Point (original space): {global_min_point} from {global_min_algo}")
