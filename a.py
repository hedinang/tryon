import os
import tqdm
import cv2
from unet import UNet2
import json
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
import torch
from torch import optim, nn
from PIL import Image
aaa
model = UNet2(3, 1)
summary(model, (3, 300, 200))
class_dict = {"short sleeve top": 1, "long sleeve top": 2, "short sleeve outwear": 3, "long sleeve outwear": 4, "vest": 5, "sling": 6,
              "shorts": 7, "trousers": 8,  "skirt": 9,  "short sleeve dress": 10, "long sleeve dress": 11,  "vest dress": 12, "sling dress": 13}


class DeepFashion(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
#

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        annos = os.path.join(self.root, "annos")
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        mask = np.zeros([height, width, 1],
                        dtype=np.int32)
        a = json.load(
            open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json")))
        for (item, value) in a.items():
            if(item[:4] != "item"):
                continue
            for poly in value['segmentation']:
                points = []
                for l in range(int(len(poly)/2)):
                    points.append([poly[l*2], poly[l*2+1]])
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
                break
        mask = torch.as_tensor(mask, dtype=torch.float32)
        mask = mask.permute(2, 0, 1)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, mask

    def __len__(self):
        return len(self.imgs)


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


dataset = DeepFashion(
    '/home/dung/Data/train', get_transform())
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0)
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=0.0005)

criterion = nn.BCELoss()
step = 0
for i in range(1000):
    print('epoch ', i)
    for (img, mask) in (data_loader):
        print()
        output = model(img).view(-1)
        mask = mask.view(-1)
        loss = criterion(output, mask)
        print('Loss step {} = {}'.format(step, loss.detach().numpy()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
    torch.save(model, '1.pth')
