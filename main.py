import os
import argparse
import datetime
import random
import json
import time
from pathlib import Path
from tensorboardX import SummaryWriter
from copy import deepcopy
import clr
from inference import infer
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader, DistributedSampler
import data
import util.misc as utils
from data import build
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set UNet', add_help=False)
    # experiment settings
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--patch_size', default=[48, 48, 48], type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--lr_drop', default=1000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # input and output channels
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=3, type=int)
    parser.add_argument('--class_names', default=["central gland", "peripheral zone"], type=str)

    # if resume from previous checkpoint
    parser.add_argument('--resume', default='/media/zhangke/SH-018/to_zhangke/output2/best_checkpoint.pth',
                        help='resume from checkpoint')

    # dataset path
    parser.add_argument('--dataset_path',
                        default='/media/zhangke/SH-018/to_zhangke/Decathlon/Prostate_data_sample/nnUNet_raw_data/Task001_ProstateX/',
                        type=str,
                        help='dataset path')

    # Output path
    parser.add_argument('--output_dir', default='/media/zhangke/SH-018/to_zhangke/output2/',
                        help='path where to save, empty for no saving')

    # log path
    parser.add_argument('--log_dir', default='/media/zhangke/SH-018/to_zhangke/logdir/',
                        help='path where to save, empty for no saving')

    # test set Dice scores
    parser.add_argument('--csv_path', default='/media/zhangke/SH-018/to_zhangke/csv_path/',
                        help='path where to save, empty for no saving')

    # prediction save path
    parser.add_argument('--pred_dir', default='/media/zhangke/SH-018/to_zhangke/pred_save_folder/',
                        help='path where to save, empty for no saving')
    ##

    # Loss type
    parser.add_argument('--CrossEntropy_loss', default=1, type=float)
    parser.add_argument('--multiDice', default=1, type=float)

    # Device setting
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, action='store_true', help="evaluate the performance on test set")
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, visualizer = build_model(args)
    model.to(device)
    print(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr,
                                 weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print('Building validation dataset...')
    dataset_val = build(image_set='val', args=args)
    num_val = len(dataset_val)
    print('Number of validation images: {}'.format(num_val))

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    dataloader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                collate_fn=utils.collate_fn)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.whst.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    test_folder = os.path.join(args.dataset_path, "test")
    test_image_folder = os.path.join(test_folder, "images")
    test_label_folder = os.path.join(test_folder, "labels")
    test_files_paths = [os.path.join(test_image_folder, file) for file in os.listdir(test_image_folder)]
    label_files_paths = [os.path.join(test_label_folder, file) for file in os.listdir(test_label_folder)]

    if args.eval:
        _ = infer(model, criterion, device, test_files_paths, label_files_paths, args.pred_dir, args.patch_size,
                  args.class_names, args.csv_path)

    else:
        print("Start training")
        best_dice = None
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):

            dataset_train = build(image_set='train', args=args)
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
            dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn,
                                          num_workers=args.num_workers)
            optimizer.param_groups[0]['lr'] = 1e-4
            train_stats = train_one_epoch(model, criterion, dataloader_train, optimizer, device, epoch)
            test_stats = evaluate(model, criterion, dataloader_val, device, visualizer, epoch, writer, args.class_names)
            dice_score = test_stats["multiDice"]
            print("dice score:", dice_score)

            if args.output_dir:
                # save latest checkpoint
                checkpoint_paths = [output_dir / 'checkpoint.pth']

                # save best checkpoint
                if best_dice == None or dice_score > best_dice:
                    best_dice = dice_score
                    print("Update best model!")
                    checkpoint_paths.append(output_dir / 'best_checkpoint.pth')

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path, _use_new_zipfile_serialization=False)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # evaluation on test set
        _ = infer(model, criterion, device, test_files_paths, label_files_paths, args.pred_dir, args.patch_size,
                  args.class_names, args.csv_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    # inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    print(torch.cuda.is_available())
    main(args)
