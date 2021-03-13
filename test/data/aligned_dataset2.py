import os.path
from data.image_folder import make_dataset, get_params, get_transform, normalize
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import cv2
from data.estimator import BodyPoseEstimator
from utils import draw_body_connections, draw_keypoints
from parsing.human_parsing import Parsing


class AlignedDataset:
    def __init__(self, opt):
        self.parsing = Parsing()
        self.opt = opt
        self.root = opt.dataroot
        self.diction = {}
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5

    def transform(self, human, cloth):

        B = Image.open(human).convert('RGB')
        params = get_params(self.opt, B.size)
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)
# parsing
        A = cv2.imread(human, cv2.IMREAD_COLOR)
        A = self.parsing(A)

        transform_A = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False)
        A_tensor = transform_A(A) * 255.0
        # Pose
        estimator = BodyPoseEstimator(self.opt.pose_model)
        image_src = cv2.imread(human)
        pose_data = estimator(image_src)
        pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        
        P_tensor = pose_map

        ### input_C (color)
        C = Image.open(cloth).convert('RGB')
        C_tensor = transform_B(C)

        # Edge
        E = cv2.imread(cloth, 0)
        ret, E = cv2.threshold(E, 240, 255, cv2.THRESH_BINARY_INV)

        E = Image.fromarray(E)
        E_tensor = transform_A(E)

        return {

            'label': A_tensor, 'image': B_tensor, 'path': human, 'edge': E_tensor,
            'color': C_tensor, 'pose': P_tensor
        }
