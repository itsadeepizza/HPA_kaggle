from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import itertools

def bbox2_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])

    y1, y2, x1, x2 = out
    return [x1, y1, x2, y2]

path = "dataset/short_train.csv"

class CellDataset(Dataset):
    def __init__(self, csv, img_path, mask_path):
        self.csv = self._parse_csv(csv)
        self.img_path = img_path
        self.mask_path = mask_path

    def _get_binary_masks(self, masks, num_cells):
        binary_masks = [masks == idx for idx in range(1, num_cells + 1)]
        return binary_masks

    def _get_bounding_boxes(self, binary_masks):
        bbs = [bbox2_ND(binmask) for binmask in binary_masks]
        return bbs

    def __getitem__(self, item):
        # FIXME get id from csv at line item
        id, weak_labels = self.csv[item]

        masks = np.load("{}/{}.npz".format(self.mask_path, id))['arr_0']
        num_cells = np.max(masks)
        binary_masks = self._get_binary_masks(masks, num_cells)
        bbs = self._get_bounding_boxes(binary_masks)

        '''
        fig, ax = plt.subplots()

        ax.imshow(masks)
        rect = patches.Rectangle((0, 100), 100, 200, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        for x1, y1, x2, y2 in bbs:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()
        print(num_cells, bbs)
        '''

        img_root = F'{self.img_path}/{id}'
        b = Image.open(F'{img_root}_blue.png').convert("L")
        g = Image.open(F'{img_root}_green.png').convert("L")
        r = Image.open(F'{img_root}_red.png').convert("L")
        y = Image.open(F'{img_root}_yellow.png').convert("L")

        ten = transforms.ToTensor()
        b = ten(b)
        g = ten(g)
        r = ten(r)
        y = ten(y)

        boxes = torch.as_tensor(bbs, dtype=torch.float)
        # FIXME: What to specify here?
        gt_classes = torch.zeros(num_cells, dtype=torch.int64)

        masks = np.stack(binary_masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        tensor_image = torch.stack([b, g, r, y])

        target = {
            "boxes": boxes,
            # FIXME labels need to go. We have to use global ones instead.
            "labels": gt_classes,
            "image_id": id,
            "masks": masks,
        }
        return tensor_image, target

    def _parse_label(self, label):
        vec_ind = label.split("|")
        bool_out = [str(x) in vec_ind for x in range(19)]
        out = [int(bool) for bool in bool_out]
        return out

    def _parse_csv(self, path):
        with open(path, newline='') as f:
            out = []
            reader = csv.reader(f)
            data = list(reader)
            for row in data[1:]:
                out.append((row[1], self._parse_label(row[2])))
        return out


    def __len__(self):
        pass

