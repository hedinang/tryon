
import time
from collections import OrderedDict
from options import Options
from model import Fashion, Refine, weights_init, MultiscaleDiscriminator, GANLoss
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
import datetime
from data.aligned_dataset import FashionDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
writer = SummaryWriter('runs/uniform_all')
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
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
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
    M_c = M_c+torch.zeros(img.shape).cuda()  # broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    # check=check>0
    # print(check)
    masked_label = label*(1-mask)
    masked_edge = mask*edge
    masked_color_strokes = mask*(1-color_mask)*color
    masked_noise = mask*noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise

# hop nhat 2 ban tay va cai ao


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor(
        (label.cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor(
        (label.cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor(
        (label.cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label


def cross_entropy2d(input, target, weight=None, size_average=True):
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


os.makedirs('sample', exist_ok=True)
opt = Options().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(
            iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
# load dataset
dataset = FashionDataset(opt)
dataloader = DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=int(opt.nThreads))
dataset_size = len(dataset)
print(f'training images = {dataset_size}')
device = torch.device('cuda')
generator = Refine(22, 14).to(device)
generator.apply(weights_init)

discriminator = MultiscaleDiscriminator(
    34+14+3, opt.ndf, opt.n_layers_D, norm_layer, opt.no_lsgan, opt.num_D, not opt.no_ganFeat_loss)

criterionGAN = GANLoss(use_lsgan=False, tensor=torch.FloatTensor)
lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(discriminator.parameters(),
                        lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(0, 200):  # 200 epoch
    for i, data in enumerate(dataloader, start=0):  # load batch size data

        human_parse = data['human_parse']
        human = data['human']
        human_file = data['human_file']
        human_pose = data['human_pose']
        cloth_parse = data['cloth_parse']
        cloth = data['cloth']
        human_mask = data['human_mask']
        cloth_mask = data['cloth_mask']
        human_changearm = changearm(human_parse)
        input = torch.cat(
            [human_changearm, human_pose, cloth], dim=1).to(device)
        output = generator.refine(input)
        output = torch.sigmoid(output)
        target = (human_parse * (1 - cloth_mask)
                  ).transpose(0, 1)[0].long().view(-1)
        output = output.transpose(1, 2).transpose(
            2, 3).contiguous().view(-1, 14)
        loss = torch.nn.CrossEntropyLoss()(output, target.to(device))

        loss_D_fake = criterionGAN(discriminator(output), False)
        loss_D_real = criterionGAN(target, True)

        generator.zero_grad()
        loss.backward()
        optimizerG.step()

        loss_G = criterionGAN(target, True)
        generator.zero_grad()
        loss_G.backward()
        optimizerG.step()

        discriminator.zero_grad()
        loss_D.backward()
        optimizerD.step()
