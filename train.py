import torch
from loader import CellDataset
import torchvision
from model.model import get_model
import torch.nn.functional as F

num_epochs = 100
batch_size = 1
lr = 0.02

classes = ['__background__', 'dummy1', 'dummy2']


def collate_images(batch):
    image_tensors = []
    weak_labels_list = []
    info_dictionaries = []
    for image, weak_labels, info in batch:
        image_tensors += [image]
        weak_labels_list += [weak_labels]
        info_dictionaries += [info]
    return image_tensors, weak_labels_list, info_dictionaries


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = CellDataset("dataset/short_train.csv", "dataset/train", "dataset/short_mask/hpa_cell_mask")

model = get_model()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=7, collate_fn=collate_images)


def sum_prob(labels_list):
    #sum = 1 - torch.prod(1-labels_list, dim=0)
    sum = torch.sum(labels_list, dim=0)
    return sum


def custom_loss(result, weak_labels):
    logits = result[0]['logits']
    if len(logits) == 0:
        print('empty')
        return torch.tensor(0.0)
    print("wooo")
    #score_list = torch.sigmoid(result[0]['logits'])

    global_result = torch.sigmoid(sum_prob(logits))

    print(global_result[:19])

    print("global shape: ", global_result[:19].shape)
    print("weak lab shape: ", weak_labels[0].shape, weak_labels[0].dtype)
    print(weak_labels[0])
    loss = F.mse_loss(global_result[:19].squeeze(0), weak_labels[0].squeeze(0).float())
    print("paolos loss", loss)
    return loss


for epoch in range(1, num_epochs + 1):
    for (images, weak_labels, targets) in data_loader:
        optimizer.zero_grad()

        images = list(image.to(device) for image in images)
        for tgt in targets:
            tgt["boxes"] = tgt["boxes"].to(device)
            tgt["labels"] = tgt["labels"].to(device)

        #model.eval()
        #result = model(images, targets)
        #print("res", result)

        model.train()
        result, loss_dict = model(images, targets)

        # We want to use a custom loss for the classification
        losses = torch.tensor(0.0)
        for name in loss_dict.keys():
            if name != 'loss_classifier':
                losses += loss_dict[name]

        losses += custom_loss(result, weak_labels)
        losses.backward()
        optimizer.step()

        print("\rloss is ", losses)