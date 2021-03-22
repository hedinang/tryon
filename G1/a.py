import os
from PIL import Image
import torch
from torchvision import transforms

transforms = transforms.Compose([transforms.ToTensor()])
img = Image.open(
    '/home/dung/Project/Data/ACGPN_traindata/train_label/019595_0.png')
img = transforms(img)
print('aaaa')