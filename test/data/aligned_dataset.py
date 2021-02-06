
import os.path
from data.image_folder import make_dataset, get_params, get_transform, normalize
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import cv2
from estimator import BodyPoseEstimator
from utils import draw_body_connections, draw_keypoints


class AlignedDataset:
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.diction = {}

        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5

        # input B (person images try)
        dir_B = '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.B_paths)
        self.build_index(self.B_paths)

        # input E (edge maps)
        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))
        self.ER_paths = make_dataset(self.dir_E)

        # input M (masks)
        dir_M = '_mask'
        self.dir_M = os.path.join(opt.dataroot, opt.phase + dir_M)
        self.M_paths = sorted(make_dataset(self.dir_M))

        # input MC(color_masks)
        dir_MC = '_colormask'
        self.dir_MC = os.path.join(opt.dataroot, opt.phase + dir_MC)
        self.MC_paths = sorted(make_dataset(self.dir_MC))

        # input C(color)
        dir_C = '_color'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.CR_paths = make_dataset(self.dir_C)
        # self.build_index(self.C_paths)

        # input A test (label maps)
        if not (opt.isTrain):
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))

    def build_index(self, dirs):
        # ipdb.set_trace()
        for k, dir in enumerate(dirs):
            name = dir.split('/')[-1]
            name = name.split('-')[0]

            # print(name)
            for k, d in enumerate(dirs[max(k-20, 0):k+20]):
                if name in d:
                    if name not in self.diction.keys():
                        self.diction[name] = []
                        self.diction[name].append(d)
                    else:
                        self.diction[name].append(d)

    def __getitem__(self, index):
        # A_path = self.A_paths[index]
        A_path = osp.join(self.dir_A, h_name.replace(".jpg", ".png"))
        
        A = Image.open(A_path).convert('L')
        params = get_params(self.opt, A.size)
        transform_A = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False)


        A_tensor = transform_A(A) * 255.0
        # input B (person images)
        B_path = self.B_paths[0]
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)
        # BR_tensor = transform_B(BR)

        ### input_C (color)
        # print(self.C_paths)
        C_path = self.C_paths[1]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        # Edge
        E_path = self.E_paths[1]
        # print(E_path)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)

        # Pose

        estimator = BodyPoseEstimator(self.opt.pose_model)
        image_src = cv2.imread(B_path)
        pose_data = estimator(image_src)
        pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
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
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor = pose_map

        return {
            'label': A_tensor, 'image': B_tensor,
            'name': A_path.split("/")[-1].split("\\")[0], 'edge': E_tensor,
            'color': C_tensor, 'pose': P_tensor
        }

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize