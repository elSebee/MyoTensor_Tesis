import pandas as pd
import numpy as np
from pathlib import Path

# Parameters
W = 300
S = 100
BASE_DIR = Path("/home/cbe/Proyectos/MyoTensor_Tesis/intelligence/datasets/raw/myotensor_proto")

X_list = []
y_list = []

# Find S07 session CSV
csv_files = sorted(list(BASE_DIR.glob("S07/session_*.csv")))
print(f"Found {len(csv_files)} S07 session CSV files.\n")

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    
    # 1. Filter valid samples
    df_valid = df[df["valid_flag"] == 1].copy()
    
    # 2. Identify contiguous blocks of restimulus
    # Cumsum of difference detects any change in class
    block_id = (df_valid["restimulus"] != df_valid["restimulus"].shift()).cumsum()
    
    groups = df_valid.groupby(block_id)
    
    for _, group_df in groups:
        gesture_class = group_df["restimulus"].iloc[0]
        sig_col = "emg_norm" if "emg_norm" in group_df.columns else "filtered"
        signal_block = group_df[sig_col].values
        
        L = len(signal_block)
        if L < W:
            continue
            
        # 3. Sliding window
        for start in range(0, L - W + 1, S):
            window = signal_block[start:start + W]
            X_list.append(window)
            y_list.append(gesture_class)

X = np.array(X_list)
X = np.expand_dims(X, axis=-1)
y = np.array(y_list)

print("Tensors processed successfully!")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nWindow counts per class:")
classes, counts = np.unique(y, return_counts=True)
gesture_names = {0: "Reposo (0)", 1: "Palma (1)", 2: "Puño (2)", 3: "Paz (3)"}
for c, cnt in zip(classes, counts):
    name = gesture_names.get(c, f"Class {c}")
    print(f"  * {name:<12} : {cnt} windows")
