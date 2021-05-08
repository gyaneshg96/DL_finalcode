import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image # PIL is a library to process images
import os
import torchvision
import torch.nn.functional as F
from tqdm import tqdm


# Dataset class before active learning

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, extra=None):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        self.num_images = len(os.listdir(self.image_dir))
        self.extra = extra

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)

    def __len__(self):
        x = 0
        if self.extra:
          x = self.extra[0].shape[0]
        return self.num_images + x 

    def __getitem__(self, idx):
        if (idx < self.num_images):
            with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
                img = Image.open(f).convert('RGB')

            return self.transform(img), self.labels[idx], idx
        else:
            return self.extra[0][idx - self.num_images], self.extra[1][idx - self.num_images], idx

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

train_dataset = CustomDataset(root="/dataset", split="train", transform=transforms.ToTensor())
test_dataset = CustomDataset(root="/dataset", split="val", transform=transforms.ToTensor())

device = torch.device('cuda:0')
print("Using device:", device)

model = torchvision.models.resnet50(pretrained=False, num_classes=800).to(device)

checkpoint = torch.load('checkpoint_moco.pth', map_location=device)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
  if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
  del state_dict[k]


log = model.load_state_dict(state_dict, strict=False)
print(log)
assert log.missing_keys == ['fc.weight', 'fc.bias']

labeled_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
unlabeled_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
    else:
      print(name)

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
print(parameters[1].shape)

optimizer = torch.optim.SGD(model.parameters(), lr=30, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=0)
criterion = torch.nn.CrossEntropyLoss().to(device)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

epochs = 50
epochs = tqdm(range(epochs))

print(device)
max_accuracy = 0
print(len(train_dataset), len(test_dataset))
for epoch in epochs:
  top1_train_accuracy = 0
  train_loss = 0
  test_loss = 0
  for counter, (x_batch, y_batch, _) in enumerate(labeled_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    train_loss += loss.item()
    
    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_train_accuracy += top1[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  top1_train_accuracy /= (counter + 1)
  top1_accuracy = 0
  top5_accuracy = 0
  for counter, (x_batch, y_batch, _) in enumerate(unlabeled_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    test_loss += loss.item()
  
    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]
  
  top1_accuracy /= (counter + 1)
  top5_accuracy /= (counter + 1)
  if (top1_accuracy.item() > max_accuracy):
    max_accuracy = top1_accuracy.item()
    torch.save(model.state_dict(), "model.pth")
  if epoch > 10:
    scheduler.step()
  print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTrain loss: {train_loss/25600}\t Test Loss: {test_loss/25600}")

