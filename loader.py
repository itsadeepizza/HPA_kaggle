from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import csv

path = "dataset/short_train.csv"

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

    def parse_label(label):
        vec_ind = label.split("|")
        bool_out = [str(x) in vec_ind for x in range(19)]
        out = [int(bool) for bool in bool_out]
        return out

    def parsing_csv(path):
        with open(path, newline='') as f:
            out = []
            reader = csv.reader(f)
            data = list(reader)
            for row in data[1:]:
                out.append((row[1], parse_label(row[2])))
        return out


    def __len__(self):
        pass

