import os
import tqdm
import cv2
import datetime
from unet import UNet2
import json
import time
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
import torch
from torch import optim, nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/log')
model = UNet2(3, 2)
device = torch.device('cuda')
model.to(device)
model = torch.load('/home/dung/Project/AI/tryon/segment/1.pth')

class DeepFashion(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.transforms = T.Compose([
            T.Resize((256, 192)),
            T.ToTensor()
        ])

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "image", self.imgs[idx])
        annos = os.path.join(self.root, "annos")
        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            json_data = json.load(
                open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json")))
            mask = np.zeros([2, height, width, 1],
                            dtype=np.float32)
            mask_ref = np.zeros([2, height, width, 3],
                                dtype=np.float32)
            for (item, value) in json_data.items():
                if(item[:4] != "item"):
                    continue
                idCloth = value["category_id"]
                label = 0
                if idCloth == 7 or idCloth == 8 or idCloth == 9:
                    # quan
                    label = 1
                else:
                    # ao
                    label = 0
                for poly in value['segmentation']:
                    points = []
                    for l in range(int(len(poly)/2)):
                        points.append([poly[l*2], poly[l*2+1]])
                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask[label], [points], 1)
                    cv2.fillPoly(mask_ref[label], [points], (0, 0, 150))
                    break
            m = []
            m.append(cv2.resize(mask[0], (192, 256),
                                interpolation=cv2.INTER_AREA))
            m.append(cv2.resize(mask[1], (192, 256),
                                interpolation=cv2.INTER_AREA))
            mr = []
            mr.append(cv2.resize(mask_ref[0], (192, 256),
                                 interpolation=cv2.INTER_AREA))
            mr.append(cv2.resize(mask_ref[1], (192, 256),
                                 interpolation=cv2.INTER_AREA))

            mask = torch.as_tensor(m, dtype=torch.float32)
            if self.transforms is not None:
                img = self.transforms(img)
        except Exception:
            mr = []
            img = torch.zeros((3, 256, 192), dtype=torch.float32)
            mask = torch.zeros((2, 256, 192), dtype=torch.float32)
            mr.append(np.zeros((256, 192, 3), dtype=np.float32))
            mr.append(np.zeros((256, 192, 3), dtype=np.float32))

        return img, mask, mr

    def __len__(self):
        return len(self.imgs)


dataset = DeepFashion('/home/dung/Project/Data/train')
batch_size = 12
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0)
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=0.0005)

criterion = nn.BCELoss().to(device)
step_per_batch = len(dataset) / batch_size
iter_path = '/home/dung/Project/AI/tryon/segment/iter.txt'
start_epoch, epoch_iter = np.loadtxt(
    iter_path, delimiter=',', dtype=int)
total_epoch = 1000
i = 0
for epoch in range(start_epoch, total_epoch):
    print('start epoch ', epoch)
    for (step, (img, mask, mask_ref)) in enumerate(data_loader, start=epoch_iter):
        try:
            h = 256
            w = 192
            mask_ref_upcloth = mask_ref[0].permute(0, 3, 1, 2).to(device)
            mask_ref_pant = mask_ref[1].permute(0, 3, 1, 2).to(device)
            iter_start_time = time.time()
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            output_fil = (output.detach().cpu().numpy() > 0.5).astype(np.int)
            output_ref = np.zeros([batch_size, 2, h, w, 3],
                                  dtype=np.float32)
            for j in range(batch_size):
                for k in range(2):
                    for wj in range(w):
                        for hj in range(h):
                            if output_fil[j][k][hj][wj] == 1:
                                output_ref[j][k][hj][wj] = [0, 0, 150]
            output_ref = torch.tensor(
                output_ref, dtype=torch.int).permute(0, 1, 4, 2, 3).to(device)
            combine = torch.cat(
                [img[0], mask_ref_upcloth[0], mask_ref_pant[0], output_ref[0][0], output_ref[0][1]], 2).squeeze()

            output = output.view(-1)
            mask = mask.view(-1)
            loss = criterion(output, mask)

            writer.add_scalar('loss', loss, i)
            writer.add_image('image', combine, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            np.savetxt(iter_path, (epoch, step), delimiter=',', fmt='%d')

            iter_end_time = time.time()
            iter_delta_time = iter_end_time - iter_start_time
            step_delta = (step_per_batch-step) + \
                step_per_batch*(total_epoch-epoch)
            eta = iter_delta_time*step_delta
            eta = str(datetime.timedelta(seconds=int(eta)))
            time_stamp = datetime.datetime.now()
            now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
            print('{}   step: {} -- loss: {} -- eta: {}'.format(
                now,
                step,
                loss.detach().cpu().numpy(),
                eta))
            i += 1
            if i % 100 == 0:
                torch.save(model, '1.pth')
        except Exception:
            continue

    print('end epoch ', epoch)
    np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
