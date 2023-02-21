import torchvision.transforms.functional as F
import torch.nn.functional as Func
import torchvision.transforms as T
import math
import sys
import random
import time
import datetime
from typing import Iterable
import numpy as np
import PIL
from PIL import Image
from skimage import transform
import nibabel as nib
import torch
import os
from medpy.metric.binary import dc
import pandas as pd
import glob
import re
import shutil
import SimpleITK as sitk
import copy
from skimage import measure

import util.misc as utils


def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    # resample images to 2mm spacing with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def CenterCrop(image, mask, patch_size):
    dx, dy, dz = image.shape
    target_x, target_y, target_z = patch_size
    if target_x < dx:
        crop_x = int(round((dx - target_x) / 2.))
    else:
        crop_x = 0
    if target_y < dy:
        crop_y = int(round((dy - target_y) / 2.))
    else:
        crop_y = 0
    if target_z < dz:
        crop_z = int(round((dz - target_z) / 2.))
    else:
        crop_z = 0
    image_cropped = image[crop_x:crop_x+target_x, crop_y:crop_y+target_y, crop_z:crop_z+target_z]
    mask_cropped = mask[crop_x:crop_x+target_x, crop_y:crop_y+target_y, crop_z:crop_z+target_z]
    return image_cropped, mask_cropped


def insert_back(output, image_shape):
    out_ch = output.shape[0]
    dx, dy, dz = image_shape
    target_x, target_y, target_z = output.shape[1], output.shape[2], output.shape[3]
    if target_x < dx:
        crop_x = int(round((dx - target_x) / 2.))
    else:
        crop_x = 0
    if target_y < dy:
        crop_y = int(round((dy - target_y) / 2.))
    else:
        crop_y = 0
    if target_z < dz:
        crop_z = int(round((dz - target_z) / 2.))
    else:
        crop_z = 0
    predictions = np.zeros((out_ch, dx, dy, dz))
    predictions[:, crop_x:crop_x + target_x, crop_y:crop_y + target_y, crop_z:crop_z + target_z] = output
    return predictions


def load_nii(img_path, resampling=True):
    itk_img = sitk.ReadImage(img_path)
    if resampling == True:
        itk_img = resample_img(itk_img)
    img = sitk.GetArrayFromImage(itk_img)
    return img


def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


@torch.no_grad()
def infer(model, criterion, device, test_files_paths, label_files_paths, output_folder, patch_size, class_names,
          csv_path):
    model.eval()
    criterion.eval()

    label_files_paths = sorted(label_files_paths, key=natural_order)
    test_files_paths = sorted(test_files_paths, key=natural_order)

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    fixed_resolution = (1.0, 1.0, 1.0)
    assert len(test_files_paths) == len(label_files_paths)

    # save predictions
    pred_files_paths = []
    for file_index in range(len(test_files_paths)):
        label_path = label_files_paths[file_index]
        label_name = label_path.split("/")[-1]
        out_file_name = os.path.join(output_folder, label_name)

        itk_label = sitk.ReadImage(label_path)
        label_resampled = load_nii(label_path)

        img_path = test_files_paths[file_index]
        img_resampled = load_nii(img_path)
        img_nomalized = np.divide((img_resampled - np.mean(img_resampled)), np.std(img_resampled))

        img_cropped, label_cropped = CenterCrop(img_nomalized, label_resampled,patch_size)
        img_cropped = np.divide((img_cropped - np.mean(img_cropped)), np.std(img_cropped))

        img_input = np.expand_dims(np.expand_dims(img_cropped, 0), 0)
        img_input = torch.from_numpy(img_input)
        img_input = img_input.to(device)
        img_input = img_input.float()
        outputs = model(img_input)
        softmax_out = outputs["pred_masks"]
        softmax_out = softmax_out.detach().cpu().numpy()
        prediction_cropped = np.squeeze(softmax_out[0, ...])
        prediction = insert_back(prediction_cropped, img_resampled.shape)
        prediction = np.uint8(np.argmax(prediction, axis=0))
        # to check
        prediction_arr = np.asarray(prediction, dtype=np.uint8)
        # prediction_arr = np.transpose(np.asarray(prediction, dtype=np.uint8), (1, 2, 0))
        # resampling
        out = sitk.GetImageFromArray(prediction_arr)
        out = resample_img(out, out_spacing=itk_label.GetSpacing(), is_label=True)
        out.SetOrigin(itk_label.GetOrigin())
        sitk.WriteImage(out, out_file_name)
        pred_files_paths.append(out_file_name)

    # compute the final Dice
    filenames_gt = label_files_paths
    filenames_pred = sorted(pred_files_paths, key=natural_order)
    file_names = []
    structure_names = []

    # measures per structure:
    dices_list = []
    structures_dict = {}
    for i in range(len(class_names)):
        structures_dict.update({int(i+1): class_names[i]})
    count = 0
    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))

        # load ground truth and prediction
        gt = load_nii(p_gt, resampling=False)
        pred = load_nii(p_pred, resampling=False)

        # calculate measures for each structure
        for struc in structures_dict.keys():
            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(
                    gt_binary) > 0:
                dices_list.append(0)
                count += 1
            else:
                dices_list.append(dc(gt_binary, pred_binary))
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])

    df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
    csv_path = os.path.join(csv_path, "dices.csv")
    df.to_csv(csv_path)
    return df
