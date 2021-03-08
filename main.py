import torch
from loader import CellDataset

dataset = CellDataset("dataset/short_train.csv", "dataset/train", "dataset/short_mask/hpa_cell_mask")
for img, tgt in dataset:
    pass
    #print(tgt)