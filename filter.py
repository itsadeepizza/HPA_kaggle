import pandas as pd
TRAIN_DF_PATH = "dataset/short_train.csv"
train_df = pd.read_csv(TRAIN_DF_PATH)
N=100
from shutil import copyfile
import os



for i in range(N):
    id = train_df.iloc[i,1]
    print(id)
    copyfile(f"dataset/mask/hpa_nuclei_mask/{id}.npz", f"dataset/short_mask/hpa_nuclei_mask/{id}.npz")
    copyfile(f"dataset/mask/hpa_cell_mask/{id}.npz", f"dataset/short_mask/hpa_cell_mask/{id}.npz")
