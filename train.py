import torch
from loader import CellDataset
import torchvision
from model.model import get_model
import torch.nn.functional as F

num_epochs = 100
batch_size = 2
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


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)


def sum_prob(labels_list):
    #print('\n\n\n\n\n')
    #print("ll", labels_list)
    #sum = 1 - torch.prod(1-labels_list, dim=0) #metodo teorico
    sum = torch.sum(labels_list, dim=0) #metodo stabile
    #print("sum", sum)
    #print('\n\n\n\n\n')
    return sum


def custom_loss_single(logits, weak_labels):
    #print("ls", logits.shape)
    if len(logits) == 0:
        return torch.tensor(0.0)

    logits = logits[:, :18]

    #print("logits", logits)

    #score_list = torch.sigmoid(result[0]['logits'])

    #global_result = sum_prob(torch.sigmoid(logits)) #metodo teorico
    global_result = sum_prob(logits) # metodo stabile

    print("target", weak_labels)
    print("result", global_result)
    #loss = F.multilabel_margin_loss(global_result.unsqueeze(0), weak_labels.unsqueeze(0)) #tra 0 e 1
    loss = F.multilabel_soft_margin_loss(global_result.unsqueeze(0), weak_labels.unsqueeze(0)) #valore reale
    #loss = F.mse_loss(global_result[:19], weak_labels.float())
    return loss
    #return torch.clamp(loss, 0, 4)


def custom_loss_batch(result, batch_weak_labels):
    loss = torch.tensor(0.0)
    for img, weak_labels in zip(result, batch_weak_labels):
        loss += custom_loss_single(img['logits'], weak_labels)

    return loss

for epoch in range(1, num_epochs + 1):
    for (images, weak_labels, targets) in data_loader:
        optimizer.zero_grad()

        images = list(image.to(device) for image in images)
        for tgt in targets:
            tgt["boxes"] = tgt["boxes"].to(device)
            tgt["labels"] = tgt["labels"].to(device)

        model.train()
        result, loss_dict = model(images, targets)
        print("loss dict", loss_dict)

        # We want to use a custom loss for the classification
        losses = torch.tensor(0.0)
        for name in loss_dict.keys():
            if name != 'loss_classifier':
                losses += loss_dict[name]

        losses += custom_loss_batch(result, weak_labels)
        losses.backward()
        clip_gradient(model, 10.)
        optimizer.step()

        print("\rloss is ", losses)