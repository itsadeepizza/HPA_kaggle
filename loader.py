from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd


class CellDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv

    def __getitem__(self, item):
        # FIXME get id from csv at line item
        id = item
        num_rois = 10
        boxes = torch.zeros((num_rois, 4), dtype=torch.float)
        gt_classes = torch.zeros(num_rois, dtype=torch.int64)
        masks = []

        b = Image.open(F'{id}_blue.png').convert("L")
        g = Image.open(F'{id}_green.png').convert("L")
        r = Image.open(F'{id}_red.png').convert("L")
        y = Image.open(F'{id}_yellow.png').convert("L")

        ten = transforms.ToTensor()
        b = ten(b)
        g = ten(g)
        r = ten(r)
        y = ten(y)

        tensor_image = torch.stack([b, g, r, y])

        target = {
            "boxes": boxes,
            # FIXME labels need to go. We have to use global ones instead.
            "labels": gt_classes,
            "image_id": id,
            "masks": masks,
        }
        return tensor_image, target

    def parsing(self, csv):
        files_and_id = []
        for i in csv:


    def __len__(self):
        pass

