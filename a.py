import torch
import numpy as np
import json
from PIL import Image
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import torchvision.transforms as T
import tqdm
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/log')
class_dict = {"short sleeve top": 1, "long sleeve top": 2, "short sleeve outwear": 3, "long sleeve outwear": 4, "vest": 5, "sling": 6,
              "shorts": 7, "trousers": 8,  "skirt": 9,  "short sleeve dress": 10, "long sleeve dress": 11,  "vest dress": 12, "sling dress": 13}


class DeepFashion(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
#

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, 'image', self.imgs[idx])
        annos = os.path.join(self.root, "annos")
        img = Image.open(img_path).convert("RGB")

        n_masks = 0
        for item in json.load(open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json"))):
            if(item[:4] != "item"):
                continue
            n_masks += 1

        width, height = img.size
        target = {}
        masks = np.zeros([n_masks, height, width, 1],
                         dtype=np.uint8)
        boxes = []
        labels = []
        i = 0
        a = json.load(
            open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json")))
        for item in json.load(open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json"))):
            cur = a[item]
            if(item[:4] != "item"):
                continue
            boxes.append(cur["bounding_box"])
            idCloth = cur["category_id"]
            labels.append(1)
            # if idCloth == 7 or idCloth == 8 or idCloth == 9:
            #     # quan
            #     labels.append(0)
            # else:
            #     # ao
            #     labels.append(1)
            for poly in cur['segmentation']:
                points = []
                for l in range(int(len(poly)/2)):
                    points.append([poly[l*2], poly[l*2+1]])
                points = np.array(points, dtype=np.int32)
                all_x = (poly[0::2])
                all_y = (poly[1::2])
                cv2.fillPoly(masks[i], [points], 1)
                break
            i += 1
        num_objs = i

        # convert everything into a torch.Tensor
        # scale 600 *600
        # scaleBoxes = []
        # for box in boxes:
        #     ratioW = 600/width
        #     ratioH = 600/height
        #     scaleX1 = box[0]*ratioW
        #     scaleY1 = box[1]*ratioH
        #     scaleX2 = box[2]*ratioW
        #     scaleY2 = box[3]*ratioH
        #     scaleBoxes.append([scaleX1, scaleY1, scaleX2, scaleY2])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        masks.permute(0, 2, 3, 1)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks.permute(0, 3, 1, 2)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            # target["masks"] = self.transforms(target["masks"])
        img = torch.unsqueeze(img, 0)
        return img, target, img_path, width, height

    def __len__(self):
        return len(self.imgs)


def get_transform(w, h):
    transforms = []
    transforms.append(T.ToTensor())
    # transforms.append(T.Resize((600, 600)))
    return T.Compose(transforms)


dataset = DeepFashion(
    '/home/dung/Project/Data/train', get_transform(600, 600))
# img = dataset[0][1]['masks'][0]


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   2)
device = torch.device('cuda')
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
step = 0
for i in range(1000):
    print('Epoch {}\n'.format(i))
    for (images, targets, path, w, h) in data_loader:
        print('{} - {} - {}'.format(path, w, h))
        images = images[0].to(device)
        a = {}

        a['boxes'] = targets['boxes'][0].to(device)
        a['labels'] = targets['labels'][0].to(device)
        a['masks'] = targets['masks'][0].to(device)
        a['image_id'] = targets['image_id'][0].to(device)
        a['area'] = targets['area'][0].to(device)
        a['iscrowd'] = targets['iscrowd'][0].to(device)
        try:
            output = model(images, [a])
        except Exception:
            continue

        losses = sum(loss for loss in output.values())
        writer.add_scalar('loss_classifier',
                          output['loss_classifier'].item(), step)
        writer.add_scalar(
            'loss_box_reg', output['loss_box_reg'].item(), step)
        writer.add_scalar('loss_objectness',
                          output['loss_objectness'].item(), step)
        writer.add_scalar('loss_rpn_box_reg',
                          output['loss_rpn_box_reg'].item(), step)
        writer.add_scalar('loss_mask',
                          output['loss_mask'].item(), step)
        # if step % 30 == 0:
        print('Step {} -- loss_classifier = {} -- loss_box_reg = {} -- loss_objectness = {} -- loss_rpn_box_reg = {} -- loss_mask = {}\n'
              .format(step,
                      output['loss_classifier'].item(),
                      output['loss_box_reg'].item(),
                      output['loss_objectness'].item(),
                      output['loss_rpn_box_reg'].item(),
                      output['loss_mask'].item()))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        step += 1
    if i % 10 == 0:
        print('save model')
        torch.save(model.state_dict(), '1.pth')
print('done')
