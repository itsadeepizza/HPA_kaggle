from .custom_mask_rcnn import maskrcnn_resnet50_fpn
import torch.nn as nn

class CustomBoxHead(nn.Module):
    def __init__(self):
        super(CustomBoxHead, self).__init__()



def get_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    return model

