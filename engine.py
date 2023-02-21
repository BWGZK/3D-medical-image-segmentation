import math
import sys
import random
import time
import datetime
from typing import Iterable
import numpy as np
import torch
import torchvision
import torch.nn.functional as Func
import PIL
import util.misc as utils


def convert_targets(target_masks, out_ch):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], out_ch, shp_y[2], shp_y[3], shp_y[4]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = len(dataloader)
    iterats = iter(dataloader)
    total_steps = numbers
    start_time = time.time()

    model.train()
    for step in range(total_steps):
        start = time.time()
        samples, targets = next(iterats)
        datatime = time.time() - start
        samples = samples.tensors.to(device)
        targets = [t["masks"].to(device) for t in targets]
        targets = torch.stack(targets)
        # train model
        model.train()
        outputs = model(samples)
        out_ch = outputs['pred_masks'].shape[1]
        targets_onehot = convert_targets(targets, out_ch)

        # PCE loss
        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.return_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in ['CrossEntropy_loss'])

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
                                    k in ['CrossEntropy_loss']}

        if step == 0:
            print("cross entropy loss:", losses.item())

        final_losses = losses

        optimizer.zero_grad()
        final_losses.backward()
        optimizer.step()

        metric_logger.update(loss=loss_dict_reduced_scaled['CrossEntropy_loss'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(model, criterion, dataloader, device, visualizer, epoch, writer, class_names):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    numbers = len(dataloader)
    iterats = iter(dataloader)
    total_steps = numbers
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    for step in range(total_steps):
        start = time.time()
        samples, targets = next(iterats)
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [t["masks"].to(device) for t in targets]
        targets = torch.stack(targets)
        outputs = model(samples.tensors)
        out_ch = outputs['pred_masks'].shape[1]
        targets_onehot = convert_targets(targets, out_ch)

        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.return_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
                                    k in weight_dict.keys()}
        for i in range(len(class_names)):
            class_name = class_names[i]
            loss_dict_reduced_scaled[class_name] = loss_dict_reduced_scaled["multiDice"][i]
        loss_dict_reduced_scaled["multiDice"] = loss_dict_reduced_scaled["multiDice"][-1]

        metric_logger.update(loss=loss_dict_reduced_scaled['CrossEntropy_loss'].float(), **loss_dict_reduced_scaled)
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)

        # save segmentation results to tensorboard
        if step % round(total_steps / 16.) == 0:
            sample_list.append(samples.tensors[0, :, :, :, 0])
            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks[:, :, :, 0])
            target_list.append(targets[0, :, :, :, 0])

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for i in range(len(class_names)):
        class_name = class_names[i]
        writer.add_scalar(class_name, stats[class_name], epoch)
    writer.add_scalar('avg_DSC', stats['multiDice'], epoch)
    writer.add_scalar('avg_loss', stats['CrossEntropy_loss'], epoch)

    # visualize images with tensorboard
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)

    return stats
