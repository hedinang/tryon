import numpy as np
import torch
import os
from torch.autograd import Variable
import torch.nn as nn
import functools
import cv2
import torch.nn.functional as F
import torch
from util.image_pool import ImagePool
import sys
import itertools
from torch.nn import Module
from torchvision import models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    return functools.partial(nn.InstanceNorm2d, affine=False)


class CNN(nn.Module):
    def __init__(self, num_output, input_nc=5, ngf=8, n_layers=5, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(CNN, self).__init__()
        downconv = nn.Conv2d(5, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2 ** i * ngf if 2 ** i * ngf < 1024 else 1024
            out_ngf = 2 ** (i + 1) * ngf if 2 ** i * ngf < 1024 else 1024
            downconv = nn.Conv2d(
                in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, norm_layer(out_ngf), nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  norm_layer(64), nn.ReLU(True)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  norm_layer(64), nn.ReLU(True)]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_output)

    def forward(self, x):
        x = self.model(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = torch.tanh(self.cnn(x))
        # ipdb.set_trace()
        coor = points.view(batch_size, -1, 2)
        row = self.get_row(coor, 5)
        col = self.get_col(coor, 5)
        rg_loss = sum(self.grad_row(coor, 5))
        cg_loss = sum(self.grad_col(coor, 5))
        rg_loss = torch.max(rg_loss, torch.tensor(0.02).cuda())
        cg_loss = torch.max(cg_loss, torch.tensor(0.02).cuda())
        rx, ry, cx, cy = torch.tensor(0.08).cuda(), torch.tensor(
            0.08).cuda(), torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda()
        row_x, row_y = row[:, :, 0], row[:, :, 1]
        col_x, col_y = col[:, :, 0], col[:, :, 1]
        rx_loss = torch.max(rx, row_x).mean()
        ry_loss = torch.max(ry, row_y).mean()
        cx_loss = torch.max(cx, col_x).mean()
        cy_loss = torch.max(cy, col_y).mean()

        return coor, rx_loss, ry_loss, cx_loss, cy_loss, rg_loss, cg_loss

    def get_row(self, coor, num):
        sec_dic = []
        for j in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for i in range(num-1):
                differ = (coor[:, j*num+i+1, :]-coor[:, j*num+i, :])**2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)
                    sec_dic.append(second_dif)

                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic, dim=1)

    def get_col(self, coor, num):
        sec_dic = []
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for j in range(num - 1):
                differ = (coor[:, (j+1) * num + i, :] -
                          coor[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)
                    sec_dic.append(second_dif)
                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic, dim=1)

    def grad_row(self, coor, num):
        sec_term = []
        for j in range(num):
            for i in range(1, num - 1):
                x0, y0 = coor[:, j * num + i - 1, :][0]
                x1, y1 = coor[:, j * num + i + 0, :][0]
                x2, y2 = coor[:, j * num + i + 1, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term

    def grad_col(self, coor, num):
        sec_term = []
        for i in range(num):
            for j in range(1, num - 1):
                x0, y0 = coor[:, (j - 1) * num + i, :][0]
                x1, y1 = coor[:, j * num + i, :][0]
                x2, y2 = coor[:, (j + 1) * num + i, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term


class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)


class TPSGridGen(nn.Module):

    def compute_partial_repr(self, input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(
            N, 1, 2) - control_points.view(1, M, 2)
        # original implementation, very slow
        # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:,
                                             :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self.compute_partial_repr(
            target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(
            range(target_height), range(target_width)))
        # print(target_coordinate)
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        # convert from (y, x) to (x, y)
        target_coordinate = torch.cat([X, Y], dim=1)
        target_coordinate_partial_repr = self.compute_partial_repr(
            target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(
                HW, 1), target_coordinate
        ], dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(
            self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(
            Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate


class STNNet(nn.Module):

    def __init__(self):
        super(STNNet, self).__init__()
        range = 0.9
        r1 = range
        r2 = range
        grid_size_h = 5
        grid_size_w = 5

        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_size_h - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_size_w - 1)),
        )))
        # ipdb.set_trace()
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)
        # self.get_row(target_control_points,5)
        # GridLocNet = {
        #     'unbounded_stn': UnBoundedGridLocNet,
        #     'bounded_stn': BoundedGridLocNet,
        # }['bounded_stn']
        # self.loc_net = GridLocNet(
        #     grid_size_h, grid_size_w, target_control_points)
        self.loc_net = BoundedGridLocNet(
            grid_size_h, grid_size_w, target_control_points)

        self.tps = TPSGridGen(256, 192, target_control_points)

    def get_row(self, coor, num):
        for j in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for i in range(num - 1):
                differ = (coor[j * num + i + 1, :] - coor[j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ - buffer)

                buffer = differ
                sum += second_dif
            print(sum / num)

    def get_col(self, coor, num):
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for j in range(num - 1):
                differ = (coor[(j + 1) * num + i, :] -
                          coor[j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)

                buffer = differ
                sum += second_dif
            print(sum)

    def grid_sample(self, input, grid, canvas=None):
        output = F.grid_sample(input, grid)
        if canvas is None:
            return output
        else:
            input_mask = Variable(input.data.new(input.size()).fill_(1))
            output_mask = F.grid_sample(input_mask, grid)
            padded_output = output * output_mask + canvas * (1 - output_mask)
            return padded_output

    def forward(self, x, reference, mask, grid_pic=None):
        if grid_pic == None:
            batch_size = x.size(0)
            source_control_points, rx, ry, cx, cy, rg, cg = self.loc_net(
                reference)
            source_control_points = (source_control_points)
            # print('control points',source_control_points.shape)
            source_coordinate = self.tps(source_control_points)
            grid = source_coordinate.view(batch_size, 256, 192, 2)
            # print('grid size',grid.shape)
            transformed_x = self.grid_sample(x, grid, canvas=0)
            warped_mask = self.grid_sample(mask, grid, canvas=0)
            return transformed_x, warped_mask, rx, ry, cx, cy, rg, cg
        else:
            batch_size = x.size(0)
            source_control_points, rx, ry, cx, cy, rg, cg = self.loc_net(
                reference)
            source_control_points = (source_control_points)
            # print('control points',source_control_points.shape)
            source_coordinate = self.tps(source_control_points)
            grid = source_coordinate.view(batch_size, 256, 192, 2)
            # print('grid size',grid.shape)
            transformed_x = self.grid_sample(x, grid, canvas=0)
            warped_mask = self.grid_sample(mask, grid, canvas=0)
            warped_gpic = self.grid_sample(grid_pic, grid, canvas=0)
            return transformed_x, warped_mask, rx, ry, cx, cy, warped_gpic


class UnetMask(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(UnetMask, self).__init__()
        self.stn = STNNet()
        nl = nn.InstanceNorm2d
        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(
                                         64), nn.ReLU(),
                                     nn.Conv2d(
                                         64, output_nc, kernel_size=3, stride=1, padding=1)
                                     ])

    def forward(self, input, refer, mask, grid=None):
        if grid == None:
            input, warped_mask, rx, ry, cx, cy, rg, cg = self.stn(
                input, torch.cat([mask, refer, input], 1), mask)
            conv1 = self.conv1(torch.cat([refer.detach(), input.detach()], 1))
            pool1 = self.pool1(conv1)

            conv2 = self.conv2(pool1)
            pool2 = self.pool2(conv2)

            conv3 = self.conv3(pool2)
            pool3 = self.pool3(conv3)

            conv4 = self.conv4(pool3)
            drop4 = self.drop4(conv4)
            pool4 = self.pool4(drop4)

            conv5 = self.conv5(pool4)
            drop5 = self.drop5(conv5)

            up6 = self.up6(drop5)
            conv6 = self.conv6(torch.cat([drop4, up6], 1))

            up7 = self.up7(conv6)
            conv7 = self.conv7(torch.cat([conv3, up7], 1))

            up8 = self.up8(conv7)
            conv8 = self.conv8(torch.cat([conv2, up8], 1))

            up9 = self.up9(conv8)
            conv9 = self.conv9(torch.cat([conv1, up9], 1))
            return conv9, input, warped_mask, rx, ry, cx, cy, rg, cg
        else:
            input, warped_mask, rx, ry, cx, cy, grid = self.stn(
                input, torch.cat([mask, refer, input], 1), mask, grid)
            conv1 = self.conv1(torch.cat([refer.detach(), input.detach()], 1))
            pool1 = self.pool1(conv1)

            conv2 = self.conv2(pool1)
            pool2 = self.pool2(conv2)

            conv3 = self.conv3(pool2)
            pool3 = self.pool3(conv3)

            conv4 = self.conv4(pool3)
            drop4 = self.drop4(conv4)
            pool4 = self.pool4(drop4)

            conv5 = self.conv5(pool4)
            drop5 = self.drop5(conv5)

            up6 = self.up6(drop5)
            conv6 = self.conv6(torch.cat([drop4, up6], 1))

            up7 = self.up7(conv6)
            conv7 = self.conv7(torch.cat([conv3, up7], 1))

            up8 = self.up8(conv7)
            conv8 = self.conv8(torch.cat([conv2, up8], 1))

            up9 = self.up9(conv8)
            conv9 = self.conv9(torch.cat([conv1, up9], 1))
            return conv9, input, warped_mask, grid


class Refine(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(Refine, self).__init__()
        nl = nn.InstanceNorm2d
        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(
                                         64), nn.ReLU(),
                                     nn.Conv2d(
                                         64, output_nc, kernel_size=3, stride=1, padding=1)
                                     ])

    def refine(self, input):
        conv1 = self.conv1(input)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([drop4, up6], 1))

        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], 1))

        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], 1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], 1))
        return conv9


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' +
                            str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]

            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))

            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLossWarp(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLossWarp, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        loss += self.weights[4] * self.criterion(x_vgg[4], y_vgg[4].detach())
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=True)
        # vgg.load_state_dict(torch.load('/home/dung/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = vgg.features
        self.vgg = vgg
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def extract(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        return x


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def warp(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        loss += self.weights[4] * self.criterion(x_vgg[4], y_vgg[4].detach())
        return loss


class StyleLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(StyleLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            N, C, H, W = x_vgg[i].shape
            for n in range(N):
                phi_x = x_vgg[i][n]
                phi_y = y_vgg[i][n]
                phi_x = phi_x.reshape(C, H * W)
                phi_y = phi_y.reshape(C, H * W)
                G_x = torch.matmul(phi_x, phi_x.t()) / (C * H * W)
                G_y = torch.matmul(phi_y, phi_y.t()) / (C * H * W)
                loss += torch.sqrt(torch.mean((G_x - G_y) ** 2)
                                   ) * self.weights[i]
        return loss


class Fashion(torch.nn.Module):
    def __init__(self, opt):
        super(Fashion, self).__init__()
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.opt = opt
        self.loss_names = ['G_GAN', 'G_GAN_Feat',
                           'G_VGG', 'D_real', 'D_fake']
        self.count = 0
        # Generator network

        self.Unet = UnetMask(4, output_nc=4)
        self.Unet.cuda(0)
        self.Unet.apply(weights_init)

        self.G1 = Refine(37, 14)
        self.G1.cuda(0)
        self.G1.apply(weights_init)

        self.G2 = Refine(19+18, 1)
        self.G2.cuda(0)
        self.G2.apply(weights_init)

        self.G = Refine(24, 3)
        self.G.cuda(0)
        self.G.apply(weights_init)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.BCE = torch.nn.BCEWithLogitsLoss()

        # Discriminator network
        if self.isTrain:
            norm_layer = get_norm_layer(norm_type=opt.norm)
            self.D1 = MultiscaleDiscriminator(
                34+14+3, opt.ndf, opt.n_layers_D, norm_layer, opt.no_lsgan, opt.num_D, not opt.no_ganFeat_loss)
            self.D1.cuda(self.gpu_ids[0])
            self.D1.apply(weights_init)

            self.D2 = MultiscaleDiscriminator(
                20+18, opt.ndf, opt.n_layers_D, norm_layer, opt.no_lsgan, opt.num_D, not opt.no_ganFeat_loss)
            self.D2.cuda(self.gpu_ids[0])
            self.D2.apply(weights_init)

            self.D = MultiscaleDiscriminator(
                27, opt.ndf, opt.n_layers_D, norm_layer, opt.no_lsgan, opt.num_D, not opt.no_ganFeat_loss)
            self.D.cuda(self.gpu_ids[0])
            self.D.apply(weights_init)

            self.D3 = MultiscaleDiscriminator(
                7, opt.ndf, opt.n_layers_D, norm_layer, opt.no_lsgan, opt.num_D, not opt.no_ganFeat_loss)
            self.D3.cuda(self.gpu_ids[0])
            self.D3.apply(weights_init)

        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError(
                    "Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()  # MAE dung ko
            if not opt.no_vgg_loss:
                self.criterionVGG = VGGLoss(self.gpu_ids)
            # khong dung
            # self.criterionStyle = StyleLoss(self.gpu_ids)
            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat',
                               'G_VGG', 'D_real', 'D_fake']
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:

                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer ork (for %d epochs) ------------' %
                      opt.niter_fix_global)
                print('The layers that are finetuned are ',
                      sorted(finetune_list))
            else:
                params = list(self.Unet.parameters())+list(self.G.parameters()) + \
                    list(self.G1.parameters())+list(self.G2.parameters())
            self.optimizer_G = torch.optim.Adam(
                params, lr=0.0002, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.D3.parameters())+list(self.D.parameters()) + \
                list(self.D2.parameters())+list(self.D1.parameters())
            self.optimizer_D = torch.optim.Adam(
                params, lr=0.0002, betas=(opt.beta1, 0.999))

            # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = opt.load_pretrain
            self.load_network(
                self.Unet, 'U', opt.which_epoch, pretrained_path)
            self.load_network(
                self.G1, 'G1', opt.which_epoch, pretrained_path)
            self.load_network(
                self.G2, 'G2', opt.which_epoch, pretrained_path)
            self.load_network(
                self.G, 'G', opt.which_epoch, pretrained_path)
            # self.load_network(
            #     self.D, 'D', opt.which_epoch, pretrained_path)
            # self.load_network(
            #     self.D1, 'D1', opt.which_epoch, pretrained_path)
            # self.load_network(
            #     self.D2, 'D2', opt.which_epoch, pretrained_path)
            # self.load_network(
            #     self.D3, 'D3', opt.which_epoch, pretrained_path)

    def generate_discrete_label(self, inputs, label_nc, onehot=True, encode=True):
        pred_batch = []
        size = inputs.size()
        for input in inputs:
            input = input.view(1, label_nc, size[2], size[3])
            pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
            pred_batch.append(pred)
        pred_batch = np.array(pred_batch)
        pred_batch = torch.from_numpy(pred_batch)
        label_map = []
        for p in pred_batch:
            p = p.view(1, 256, 192)
            label_map.append(p)
        label_map = torch.stack(label_map, 0)
        if not onehot:
            return label_map.float().cuda()
        size = label_map.size()
        oneHot_size = (size[0], label_nc, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(
            1, label_map.data.long().cuda(), 1.0)

        return input_label

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:

            network.load_state_dict(torch.load(save_path))

    def encode_input(self, label_map, clothes_mask, all_clothes_label):
        size = label_map.size()
        oneHot_size = (size[0], 14, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        a = label_map.long()

        input_label = input_label.scatter_(
            1, a, 1.0)
        masked_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        masked_label = masked_label.scatter_(
            1, (label_map*(1-clothes_mask)).data.long().cuda(), 1.0)

        c_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_label = c_label.scatter_(
            1, all_clothes_label.data.long().cuda(), 1.0)

        input_label = Variable(input_label)

        return input_label, masked_label, c_label

    def discriminate(self, netD, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return netD.forward(fake_query)
        else:
            return netD.forward(input_concat)

    def gen_noise(self, shape):
        noise = np.zeros(shape, dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()

    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(
                ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )

        return loss
# skin color

    def ger_average_color(self, mask, arms):
        color = torch.zeros(arms.shape).cuda()
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 10:
                color[i, 0, :, :] = 0
                color[i, 1, :, :] = 0
                color[i, 2, :, :] = 0

            else:
                color[i, 0, :, :] = arms[i, 0, :, :].sum()/count
                color[i, 1, :, :] = arms[i, 1, :, :].sum()/count
                color[i, 2, :, :] = arms[i, 2, :, :].sum()/count
        return color

    def morpho(self, mask, iter, bigger=True):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        new = []
        for i in range(len(mask)):
            tem = mask[i].cpu().detach().numpy(
            ).squeeze().reshape(256, 192, 1)*255
            tem = tem.astype(np.uint8)
            if bigger:
                tem = cv2.dilate(tem, kernel, iterations=iter)
            else:
                tem = cv2.erode(tem, kernel, iterations=iter)
            tem = tem.astype(np.float64)
            tem = tem.reshape(1, 256, 192)
            new.append(tem.astype(np.float64)/255.0)
        new = np.stack(new)
        new = torch.FloatTensor(new).cuda()
        return new

    def morpho_smaller(self, mask, iter, bigger=True):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        new = []
        for i in range(len(mask)):
            tem = mask[i].cpu().detach().numpy(
            ).squeeze().reshape(256, 192, 1)*255
            tem = tem.astype(np.uint8)
            if bigger:
                tem = cv2.dilate(tem, kernel, iterations=iter)
            else:
                tem = cv2.erode(tem, kernel, iterations=iter)
            tem = tem.astype(np.float64)
            tem = tem.reshape(1, 256, 192)
            new.append(tem.astype(np.float64)/255.0)
        new = np.stack(new)
        new = torch.FloatTensor(new).cuda()
        return new

    def encode(self, label_map, size):
        label_nc = 14
        oneHot_size = (size[0], label_nc, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(
            1, label_map.data.long().cuda(), 1.0)
        return input_label

    def forward(self, label, pre_clothes_mask, img_fore, clothes_mask, clothes, all_clothes_label, real_image, pose, mask):

        input_label, masked_label, all_clothes_label = self.encode_input(
            label, clothes_mask, all_clothes_label)

        arm1_mask = torch.FloatTensor(
            (label.cpu().numpy() == 11).astype(np.float)).cuda()
        arm2_mask = torch.FloatTensor(
            (label.cpu().numpy() == 13).astype(np.float)).cuda()
        pre_clothes_mask = torch.FloatTensor(
            (pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()

        clothes = clothes*pre_clothes_mask
        shape = pre_clothes_mask.shape

        G1_in = torch.cat([pre_clothes_mask, clothes,
                           all_clothes_label, pose, self.gen_noise(shape)], dim=1)
        arm_label = self.G1.refine(G1_in)

        arm_label = self.sigmoid(arm_label)

        CE_loss = self.cross_entropy2d(
            arm_label, (label * (1 - clothes_mask)).transpose(0, 1)[0].long())*10

        armlabel_map = self.generate_discrete_label(
            arm_label.detach(), 14, False)
        dis_label = self.generate_discrete_label(arm_label.detach(), 14)

        G2_in = torch.cat([pre_clothes_mask, clothes,
                           masked_label, pose, self.gen_noise(shape)], 1)
        fake_cl = self.G2.refine(G2_in)
        fake_cl = self.sigmoid(fake_cl)
        CE_loss += self.BCE(fake_cl, clothes_mask)*10

        fake_cl_dis = torch.FloatTensor(
            (fake_cl.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        new_arm1_mask = torch.FloatTensor(
            (armlabel_map.cpu().numpy() == 11).astype(np.float)).cuda()
        new_arm2_mask = torch.FloatTensor(
            (armlabel_map.cpu().numpy() == 13).astype(np.float)).cuda()
        arm1_occ = clothes_mask*new_arm1_mask
        arm2_occ = clothes_mask*new_arm2_mask
        arm1_full = arm1_occ+(1-clothes_mask)*arm1_mask
        arm2_full = arm2_occ+(1-clothes_mask)*arm2_mask
        armlabel_map *= (1-new_arm1_mask)
        armlabel_map *= (1-new_arm2_mask)
        armlabel_map = armlabel_map*(1-arm1_full)+arm1_full*11
        armlabel_map = armlabel_map*(1-arm2_full)+arm2_full*13

        # construct full label map
        armlabel_map = armlabel_map*(1-fake_cl_dis)+fake_cl_dis*4

        fake_c, warped, warped_mask, rx, ry, cx, cy, rg, cg = self.Unet(
            clothes, clothes_mask, pre_clothes_mask)
        composition_mask = fake_c[:, 3, :, :]
        fake_c = fake_c[:, 0:3, :, :]
        fake_c = self.tanh(fake_c)
        composition_mask = self.sigmoid(composition_mask)

        skin_color = self.ger_average_color(
            (arm1_mask+arm2_mask-arm2_mask*arm1_mask), (arm1_mask+arm2_mask-arm2_mask*arm1_mask)*real_image)

        img_hole_hand = img_fore*(1-clothes_mask)*(1-arm1_mask)*(
            1-arm2_mask)+img_fore*arm1_mask*(1-mask)+img_fore*arm2_mask*(1-mask)

        G_in = torch.cat([img_hole_hand, masked_label, real_image *
                          clothes_mask, skin_color, self.gen_noise(shape)], 1)
        fake_image = self.G.refine(G_in.detach())
        fake_image = self.tanh(fake_image)
        # THE POOL TO SAVE IMAGES

        input_pool = [G1_in, G2_in, G_in, torch.cat(
            [clothes_mask, clothes], 1)]  # fake_cl_dis to replace

        real_pool = [masked_label, clothes_mask,
                     real_image, real_image*clothes_mask]
        fake_pool = [arm_label, fake_cl, fake_image, fake_c]
        D_pool = [self.D1, self.D2, self.D, self.D3]
        pool_lenth = len(fake_pool)
        loss_D_fake = 0
        loss_D_real = 0
        loss_G_GAN = 0
        loss_G_GAN_Feat = 0

        for iter_p in range(pool_lenth):

            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(
                D_pool[iter_p], input_pool[iter_p].detach(), fake_pool[iter_p], use_pool=True)

            loss_D_fake += self.criterionGAN(pred_fake_pool, False)
            # Real Detection and Loss
            pred_real = self.discriminate(
                D_pool[iter_p], input_pool[iter_p].detach(), real_pool[iter_p])
            loss_D_real += self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = D_pool[iter_p].forward(
                torch.cat((input_pool[iter_p].detach(), fake_pool[iter_p]), dim=1))
            loss_G_GAN += self.criterionGAN(pred_fake, True)
            if iter_p < 2:
                continue
            # # GAN feature matching loss
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        comp_fake_c = fake_c.detach()*(1-composition_mask).unsqueeze(1) + \
            (composition_mask.unsqueeze(1))*warped.detach()

        # VGG feature matching loss
        loss_G_VGG = 0
        loss_G_VGG += self.criterionVGG.warp(warped, real_image*clothes_mask) + \
            self.criterionVGG.warp(comp_fake_c, real_image*clothes_mask) * 10
        loss_G_VGG += self.criterionVGG.warp(fake_c,
                                             real_image*clothes_mask) * 20
        loss_G_VGG += self.criterionVGG(fake_image, real_image) * 10

        L1_loss = self.criterionFeat(fake_image, real_image)

        L1_loss += self.criterionFeat(warped_mask, clothes_mask) + \
            self.criterionFeat(warped, real_image*clothes_mask)
        L1_loss += self.criterionFeat(fake_c, real_image*clothes_mask)*0.2
        L1_loss += self.criterionFeat(comp_fake_c, real_image*clothes_mask)*10
        L1_loss += self.criterionFeat(composition_mask, clothes_mask)

        style_loss = L1_loss
        # Only return the fake_B image if necessary to save BW
        return [[loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake], fake_c, comp_fake_c, dis_label,
                L1_loss, style_loss, fake_cl, warped, clothes, CE_loss, rx*0.1, ry*0.1, cx*0.1, cy*0.1, rg*0.1, cg*0.1]

    def inference(self, label, pre_clothes_mask, img_fore, clothes_mask, clothes, all_clothes_label, real_image, pose, grid, mask_fore):
        # Encode Inputs
        input_label, masked_label, all_clothes_label = self.encode_input(
            label, clothes_mask, all_clothes_label)
        arm1_mask = torch.FloatTensor(
            (label.cpu().numpy() == 11).astype(np.float)).cuda()
        arm2_mask = torch.FloatTensor(
            (label.cpu().numpy() == 13).astype(np.float)).cuda()
        pre_clothes_mask = torch.FloatTensor(
            (pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        clothes = clothes * pre_clothes_mask

        shape = pre_clothes_mask.shape

        G1_in = torch.cat([pre_clothes_mask, clothes,
                           all_clothes_label, pose, self.gen_noise(shape)], dim=1)
        arm_label = self.G1.refine(G1_in)

        arm_label = self.sigmoid(arm_label)

        armlabel_map = self.generate_discrete_label(
            arm_label.detach(), 14, False)
        dis_label = self.generate_discrete_label(arm_label.detach(), 14)
        G2_in = torch.cat([pre_clothes_mask, clothes,
                           dis_label, pose, self.gen_noise(shape)], 1)
        fake_cl = self.G2.refine(G2_in)
        fake_cl = self.sigmoid(fake_cl)

        fake_cl_dis = torch.FloatTensor(
            (fake_cl.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        fake_cl_dis = self.morpho(fake_cl_dis, 1, True)

        new_arm1_mask = torch.FloatTensor(
            (armlabel_map.cpu().numpy() == 11).astype(np.float)).cuda()
        new_arm2_mask = torch.FloatTensor(
            (armlabel_map.cpu().numpy() == 13).astype(np.float)).cuda()
        fake_cl_dis = fake_cl_dis*(1 - new_arm1_mask)*(1-new_arm2_mask)
        fake_cl_dis *= mask_fore

        arm1_occ = clothes_mask * new_arm1_mask
        arm2_occ = clothes_mask * new_arm2_mask
        bigger_arm1_occ = self.morpho(arm1_occ, 20)
        bigger_arm2_occ = self.morpho(arm2_occ, 20)
        arm1_full = arm1_occ + (1 - clothes_mask) * arm1_mask
        arm2_full = arm2_occ + (1 - clothes_mask) * arm2_mask
        armlabel_map *= (1 - new_arm1_mask)
        armlabel_map *= (1 - new_arm2_mask)
        armlabel_map = armlabel_map * (1 - arm1_full) + arm1_full * 11
        armlabel_map = armlabel_map * (1 - arm2_full) + arm2_full * 13
        armlabel_map *= (1-fake_cl_dis)
        dis_label = self.encode(armlabel_map, armlabel_map.shape)

        fake_c, warped, warped_mask, warped_grid = self.Unet(
            clothes, fake_cl_dis, pre_clothes_mask, grid)
        mask = fake_c[:, 3, :, :]
        mask = self.sigmoid(mask)*fake_cl_dis
        fake_c = self.tanh(fake_c[:, 0:3, :, :])
        fake_c = fake_c*(1-mask)+mask*warped
        skin_color = self.ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                                            (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * real_image)
        occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask+clothes_mask)) * \
            (1 - bigger_arm2_occ * (arm2_mask + arm1_mask+clothes_mask))
        img_hole_hand = img_fore * \
            (1 - clothes_mask) * occlude * (1 - fake_cl_dis)

        G_in = torch.cat([img_hole_hand, dis_label, fake_c,
                          skin_color, self.gen_noise(shape)], 1)
        fake_image = self.G.refine(G_in.detach())
        fake_image = self.tanh(fake_image)

        return [fake_image, warped, fake_c]


class Inference(Fashion):
    # why real image  =  grid
    def forward(self, label, pre_clothes_mask, img_fore, clothes_mask, clothes, all_clothes_label, real_image, pose, grid, mask_fore):
        label = torch.unsqueeze(label, 0)
        pre_clothes_mask = torch.unsqueeze(pre_clothes_mask, 0)
        img_fore = torch.unsqueeze(img_fore, 0)
        clothes_mask = torch.unsqueeze(clothes_mask, 0)
        clothes = torch.unsqueeze(clothes, 0)
        all_clothes_label = torch.unsqueeze(all_clothes_label, 0)
        real_image = torch.unsqueeze(real_image, 0)
        pose = torch.unsqueeze(pose, 0)
        grid = torch.unsqueeze(grid, 0)
        mask_fore = torch.unsqueeze(mask_fore, 0)

        return self.inference(label, pre_clothes_mask, img_fore, clothes_mask, clothes, all_clothes_label, real_image, pose, grid, mask_fore)
