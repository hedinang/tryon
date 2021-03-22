from data.image_folder import make_dataset, get_params, get_transform, normalize
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import cv2
import os


class FashionDataset:
    def __init__(self, opt):
        self.opt = opt
        self.diction = {}
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5
        params = get_params(self.opt, (192, 256))
        self.transformRGB = get_transform(self.opt, params)
        self.transformGray = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False)
        self.dir_human = sorted(make_dataset(
            '{}/{}_img'.format(self.opt.dataroot, self.opt.phase)))
        self.dir_human_mask = '{}/{}_mask'.format(self.opt.dataroot,
                                                  self.opt.phase)
        self.dir_cloth_mask = '{}/{}_colormask'.format(self.opt.dataroot,
                                                       self.opt.phase)

        self.dataset_size = len(self.dir_human)
        self.build_index(self.dir_human)

    def build_index(self, dirs):

        for k, dir in enumerate(dirs):
            name = dir.split('/')[-1]
            name = name.split('-')[0]
            for k, d in enumerate(dirs[max(k-20, 0):k+20]):
                if name in d:
                    if name not in self.diction.keys():
                        self.diction[name] = []
                        self.diction[name].append(d)
                    else:
                        self.diction[name].append(d)

    def __getitem__(self, index):

        human_file = self.dir_human[0]
        human = Image.open(human_file).convert('RGB')
        human = self.transformRGB(human)
        human_parse = Image.open(
            human_file.replace('img', 'label').replace('jpg', 'png')).convert('L')
        human_parse = self.transformGray(human_parse)

        with open(human_file.replace('.jpg', '_keypoints.json').replace('train_img', 'train_pose'), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        point_num = pose_data.shape[0]
        human_pose = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx +
                                r, pointy+r), 'white', 'white')
                pose_draw.rectangle(
                    (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transformRGB(one_map.convert('RGB'))
            human_pose[i] = one_map[0]
        human_mask = os.listdir(self.dir_human_mask)[np.random.randint(1200)]
        human_mask = Image.open(
            '{}/{}'.format(self.dir_human_mask, human_mask)).convert('L')
        human_mask = self.transformGray(human_mask)

        cloth = Image.open(
            human_file.replace('img', 'color').replace('_0', '_1')).convert('RGB')
        cloth = self.transformRGB(cloth)
        cloth_parse = Image.open(
            human_file.replace('img', 'edge').replace('_0', '_1')).convert('L')
        cloth_parse = self.transformGray(cloth_parse)
        cloth_mask = os.listdir(self.dir_cloth_mask)[np.random.randint(1200)]
        cloth_mask = Image.open(
            '{}/{}'.format(self.dir_cloth_mask, cloth_mask)).convert('L')
        cloth_mask = self.transformGray(cloth_mask)

        return {
            'human_parse': human_parse, 'human': human, 'human_file': human_file, 'human_pose': human_pose,
            'cloth_parse': cloth_parse, 'cloth': cloth, 'human_mask': human_mask, 'cloth_mask': cloth_mask
        }

    def __len__(self):
        return len(self.dir_human) // self.opt.batchSize
