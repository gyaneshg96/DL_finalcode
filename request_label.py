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

def confusion_threshold(output, idx, threshold=0.5):
  with torch.no_grad():
    pred = output.max(1)
    val = pred.values
    ind = pred.indices
    correct = val < threshold
    idx = idx[correct]
    return idx, correct.sum()*100/output.shape[0]

def confusion_two_threshold(output, idx, threshold=0.5):
  with torch.no_grad():
    pred = output.topk(2,1)
    val = pred.values
    ind = pred.indices
    diff = val[:, 0] - val[:, 1]
    correct = diff < threshold
    idx = idx[correct]
    return idx, correct.sum()*100/output.shape[0]

def confusion_entropy(output, idx, threshold=0.5):
  with torch.no_grad():
    val = - output * torch.log2(output)
    val = val.sum(dim = 1)/torch.log2(torch.tensor(output.shape[1]))
    correct = val > threshold
    idx = idx[correct]
    return idx, correct.sum()*100/output.shape[0]

device = torch.device('cuda:0')
print("Using device:", device)

unlabeled_dataset = CustomDataset(root="/dataset", split="unlabeled", transform=transforms.ToTensor())

unsupervised_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=1024)

model1 = torchvision.models.resnet34(pretrained=False, num_classes=800).to(device)

model1.load_state_dict(torch.load('finalcheckpoint_34_199.pth.tar', map_location=device))

model1.eval()

active_dataset1 = []

for counter, (x_batch, y_batch, idx) in enumerate(unsupervised_loader):
  x_batch = x_batch.to(device)
  idx = idx.to(device)

  prob = model1(x_batch)
  
  idx1, fr1 = confusion_threshold(prob, idx, 0.08)
  idx2, fr2 = confusion_two_threshold(prob, idx, 0.01)
  idx3, fr3 = confusion_entropy(prob, idx, 0.75)
  combined = torch.cat((idx1, idx2, idx3))
  uniques, counts = combined.unique(return_counts=True)
  intersection = uniques[counts > 2]
  active_dataset1.append(intersection)

temp = torch.cat(active_dataset1, dim = 0)
final_list = np.random.choice(np.asarray(temp.cpu()), 12800)

with open('request_11.csv', 'w') as csv_file:
  for i in range(12800):
    csv_file.write('/dataset/unlabeled/'+str(final_list[i])+'.png,\n')