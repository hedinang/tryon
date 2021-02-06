
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from parsing.transforms import transform_logits
from collections import OrderedDict
from parsing.AugmentCE2P import resnet101


class Parsing:
    def __init__(self, model_restore='/home/dung/Project/AI/tryon/checkpoints/exp-schp-201908261155-lip.pth', gpu=0):
        super(Parsing, self).__init__()

        self.num_classes = 20
        self.input_size = [473, 473]
        self.label = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                      'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                      'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
        self.model_restore = model_restore
        self.gpu = gpu
        self.logits = False
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __call__(self, image):
        model = resnet101(num_classes=self.num_classes, pretrained=None)
        state_dict = torch.load(self.model_restore)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # model.cuda()
        model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                0.225, 0.224, 0.229])
        ])
        w, h = image.size
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        c = person_center
        # s = s
        # w = w
        # h = h

        output = model(image)
        upsample = torch.nn.Upsample(
            size=self.input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(
            1, 2, 0)  # CHW -> HWC
        logits_result = transform_logits(
            upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
        parsing_result = np.argmax(logits_result, axis=2)

        for i in range(h):
            for j in range(w):
                if parsing_result[i][j] == 1 or parsing_result[i][j] == 3 or parsing_result[i][j] == 4 or\
                        parsing_result[i][j] == 6 or parsing_result[i][j] == 7 or parsing_result[i][j] == 8 or\
                        parsing_result[i][j] == 10 or parsing_result[i][j] == 11 or parsing_result[i][j] == 12 or\
                        parsing_result[i][j] == 16 or parsing_result[i][j] == 17 or parsing_result[i][j] == 18 or parsing_result[i][j] == 19:
                    parsing_result[i][j] = 0

                if parsing_result[i][j] == 2:
                    parsing_result[i][j] = 1

                elif parsing_result[i][j] == 5:
                    parsing_result[i][j] = 4

                elif parsing_result[i][j] == 9:
                    parsing_result[i][j] = 8

                elif parsing_result[i][j] == 13:
                    parsing_result[i][j] = 12

                elif parsing_result[i][j] == 15:
                    parsing_result[i][j] = 13

                elif parsing_result[i][j] == 14:
                    parsing_result[i][j] = 11

        return Image.fromarray(np.uint8(parsing_result)).convert('RGB')
