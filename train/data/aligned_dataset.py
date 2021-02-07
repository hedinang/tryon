
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

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'name': img_name,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta

    def __init__(self, opt):
        self.parsing = Parsing()
        self.opt = opt
        self.root = opt.dataroot
        self.diction = {}
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5
        # input B (real images)
        dir_B = '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.BR_paths = sorted(make_dataset(self.dir_B))

        # input A (label maps)

        self.dataset_size = len(self.B_paths)
        self.build_index(self.B_paths)

        # input E (edge_maps)
        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))
        self.ER_paths = make_dataset(self.dir_E)

        # input M (masks)
        dir_M = '_mask'
        self.dir_M = os.path.join(opt.dataroot, opt.phase + dir_M)
        self.M_paths = sorted(make_dataset(self.dir_M))
        self.MR_paths = make_dataset(self.dir_M)

        # input MC(color_masks)
        dir_MC = '_colormask'
        self.dir_MC = os.path.join(opt.dataroot, opt.phase + dir_MC)
        self.MC_paths = sorted(make_dataset(self.dir_MC))
        self.MCR_paths = make_dataset(self.dir_MC)
        # input C(color)
        dir_C = '_color'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.CR_paths = make_dataset(self.dir_C)

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
        # input B (real images)
        B_path = self.B_paths[0]
        B = Image.open(B_path).convert('RGB')
        params = get_params(self.opt, B.size)
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)

        # A_path = self.A_paths[index]
        A = self.parsing(B)
        

        transform_A = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False)
        A_tensor = transform_A(A) * 255.0

        # input M (masks)
        MR_path = self.MR_paths[np.random.randint(12000)]
        MR = Image.open(MR_path).convert('L')
        M_tensor = transform_A(MR)

        ### input_MC (colorMasks)
        MCR_path = B_path  # self.MCR_paths[1]
        MCR = Image.open(MCR_path).convert('L')
        MC_tensor = transform_A(MCR)

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
            'label': A_tensor, 'image': B_tensor, 'path': B_path, 'edge': E_tensor,
            'color': C_tensor, 'mask': M_tensor, 'colormask': MC_tensor, 'pose': P_tensor
        }

    def __len__(self):
        return len(self.B_paths) // self.opt.batchSize * self.opt.batchSize
