import numpy as np
import shutil
import os
import subprocess
import time
import multiprocessing

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)
from models.twod_models.temporal_modeling import TAM

def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size, kernel_size).view(kernel_size, kernel_size, kernel_size)
    y_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).t()
    z_grid = y_grid.repeat(1, kernel_size).view(kernel_size, kernel_size, kernel_size)
    y_grid = y_grid.repeat(kernel_size, 1).view(kernel_size, kernel_size, kernel_size)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xyz_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
    # print(gaussian_kernel.shape)
    if kernel_size == 3:
        padding = 1
    else:
        padding = 0
    gaussian_filter = nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)
    gaussian_kernel = gaussian_kernel.cuda()
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

def getTAMloss(model):
    loss = 0
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, TAM):
            weights = torch.stack(
                [m._modules['prev_se'].fc1.weight.data.view(-1), m._modules['curr_se'].fc1.weight.data.view(-1),
                 m._modules['next_se'].fc1.weight.data.view(-1)], dim=0)
            L2 = torch.norm(weights, p=1, dim=0)
            loss += torch.sum(L2)
            cnt += L2.shape[0]
    return loss / cnt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def map_charades(y_pred, y_true):
    """ Returns mAP """
    y_true = y_true.cpu().numpy().astype(np.int32)
    y_pred = y_pred.cpu().detach().numpy()
    m_aps = []
    n_classes = y_pred.shape[1]
    for oc_i in range(n_classes):
        pred_row = y_pred[:, oc_i]
        sorted_idxs = np.argsort(-pred_row)
        true_row = y_true[:, oc_i]
        tp = true_row[sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(np.nan)
            continue
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(y_pred.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    m_ap = m_ap if not np.isnan(m_ap) else 0.0
    return [m_ap], [0]


def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_total_flops_params(summary):
    for line in summary.split("\n"):
        line = line.strip()
        if line == "":
            continue
        if "Total flops" in line:
            total_flops = line.split(":")[-1].strip()
        elif "Total params" in line:
            total_params = line.split(":")[-1].strip()

    return total_flops, total_params


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    result = torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    return result


def get_TSM_regularization(model, blending_frames):
    sparsity_loss = 0
    for m in model.modules():
        if isinstance(m, LearnableTSM):
            if blending_frames == 3:
                weights = torch.stack(
                        [m._modules['prev_se'].fc1.weight.view(-1), m._modules['curr_se'].fc1.weight.view(-1),
                        m._modules['next_se'].fc1.weight.view(-1)], dim=0)
            else:
                weights = torch.stack(
                    [m._modules['blending_layers']._modules['{}'.format(i)].fc1.weight.view(-1) for i in
                    range(blending_frames)], dim=0)
            l1_term = 1.0 - torch.sum(weights, dim=0)
            l2_term = torch.sum(weights ** 2, dim=0)
            sparsity_loss += (torch.sum(l1_term * l1_term) + torch.sum(torch.ones_like(l2_term) - l2_term)) / (l1_term.numel() * 2)
            #                print (loss, sparsity_loss)
    return sparsity_loss


def train(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, label_smoothing=0.0, num_classes=None,
          clip_gradient=None, gpu_id=None, rank=0, eval_criterion=accuracy, use_sparsity=False, sparsity_w=1.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    tam_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            output = model(images)
            #TODO check label_smoothing
            if label_smoothing > 0.0:
                smoothed_target = torch.zeros([images.size(0), num_classes]).scatter_(
                    1, target.unsqueeze(1), 1.0) * (1.0 - label_smoothing) + 1 / float(num_classes) * label_smoothing
                smoothed_target = smoothed_target.type(torch.float)
                smoothed_target = smoothed_target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                loss = cross_entropy(output, smoothed_target)
            else:
                target = target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                loss = criterion(output, target)

            if use_sparsity:
                TAM_loss = getTAMloss(model)
                loss += TAM_loss * sparsity_w

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate(data_loader, model, criterion, gpu_id=None, eval_criterion=accuracy):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    return top1.avg, top5.avg, losses.avg, batch_time.avg
