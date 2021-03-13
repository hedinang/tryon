
import time
from collections import OrderedDict
from options import Options
from model import Fashion, Inference
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
import datetime
from data.aligned_dataset2 import AlignedDataset
from torch.utils.data import DataLoader
SIZE = 320
NC = 14


def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], NC))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img*(1-mask)
    M_c = (1-mask.cuda())*M_f
    M_c = M_c+torch.zeros(img.shape).cuda()
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    masked_label = label*(1-mask)
    masked_edge = mask*edge
    masked_color_strokes = mask*(1-color_mask)*color
    masked_noise = mask*noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((old_label.cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((old_label.cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((old_label.cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label


def main():
    opt = Options().parse()
    dataset = AlignedDataset(opt)
    data = dataset.transform(
        '/home/dung/Project/AI/DeepFashion_Try_On/Data_preprocessing/ACGPN_traindata/train_img/000024_0.jpg',
        '/home/dung/Project/AI/DeepFashion_Try_On/Data_preprocessing/ACGPN_traindata/train_color/000052_1.jpg')
    model = Inference(opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    mask_clothes = torch.FloatTensor(
        (data['label'].cpu().numpy() == 4).astype(np.int))
    mask_fore = torch.FloatTensor(
        (data['label'].cpu().numpy() > 0).astype(np.int))
    img_fore = data['image'] * mask_fore
    all_clothes_label = changearm(data['label'])

    fake_image, warped_cloth, refined_cloth = model(data['label'].cuda(), data['edge'].cuda(),
                                                    img_fore.cuda(), mask_clothes.cuda(), data['color'].cuda(
    ), all_clothes_label.cuda(), data['image'].cuda(),
        data['pose'].cuda(), data['image'].cuda(), mask_fore.cuda())

    # make output folders
    output_dir = os.path.join(opt.results_dir, opt.phase)
    fake_image_dir = os.path.join(output_dir, 'try-on')
    os.makedirs(fake_image_dir, exist_ok=True)
    warped_cloth_dir = os.path.join(output_dir, 'warped_cloth')
    os.makedirs(warped_cloth_dir, exist_ok=True)
    refined_cloth_dir = os.path.join(output_dir, 'refined_cloth')
    os.makedirs(refined_cloth_dir, exist_ok=True)

    # save output
    for j in range(opt.batchSize):
        name = data['path'].split('/')[-1]
        print("Saving", name)
        util.save_tensor_as_image(fake_image[j],
                                  os.path.join(fake_image_dir, name))
        util.save_tensor_as_image(warped_cloth[j],
                                  os.path.join(warped_cloth_dir, name))
        util.save_tensor_as_image(refined_cloth[j],
                                  os.path.join(refined_cloth_dir, name))


if __name__ == '__main__':
    main()
