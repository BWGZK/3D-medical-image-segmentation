import SimpleITK as sitk
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils import data
import nibabel as nib
import os
import data.transforms as T

class seg_train(data.Dataset):
    def __init__(self, img_paths, lab_paths, transforms):
        self._transforms = transforms
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}
        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            img = self.read_image(str(img_path))
            img_name = img_path.stem
            self.img_dict.update({img_name: img})
            lab = self.read_label(str(lab_path))
            lab_name = lab_path.stem
            self.lab_dict.update({lab_name: lab})
            self.examples += [(img_name, lab_name)]
            
    def __getitem__(self, idx):
        img_name, lab_name = self.examples[idx]
        img = self.img_dict[img_name]
        lab = self.lab_dict[lab_name]
        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)
        target = {'name': lab_name,'masks': lab, 'orig_size': lab.shape}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def read_image(self, img_path):
        img = self.load_nii(img_path, is_label=False)
        img = img.astype(np.float32)
        return (img-img.mean())/img.std()

    def read_label(self, lab_path):
        lab = self.load_nii(lab_path, is_label=True)
        return lab

    def load_nii(self, img_path, is_label):
        itk_img = sitk.ReadImage(img_path)
        resample_img = self.resample_img(itk_img, is_label= is_label)
        img = sitk.GetArrayFromImage(resample_img)
        return img

    def resample_img(self, itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
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

    def __len__(self):
        return len(self.examples)

def make_transforms(image_set, patch_size):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.CenterCrop(patch_size),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.CenterCrop(patch_size),
            normalize
        ])


def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'

    PATHS = {
        "train": (root / "train"/"images", root / "train"/"labels"),
        "val": (root / "val"/"images", root / "val"/"labels"),
    }
    data_transforms = make_transforms(image_set, args.patch_size)
    if image_set == "train":
        img_folder, lab_folder = PATHS[image_set]
        # get data paths
        img_paths = sorted(list(img_folder.iterdir()))
        lab_paths = sorted(list(lab_folder.iterdir()))
        order_indexs = [i for i in range(len(img_paths))]
        random.shuffle(order_indexs)
        img_paths = [img_paths[i] for i in order_indexs]
        lab_paths = [lab_paths[i] for i in order_indexs]
        dataset_train = seg_train(img_paths, lab_paths, transforms=data_transforms)
        return dataset_train

    elif image_set == "val":
        img_folder, lab_folder = PATHS[image_set]
        img_paths = sorted(list(img_folder.iterdir()))
        lab_paths = sorted(list(lab_folder.iterdir()))
        dataset_val = seg_train(img_paths, lab_paths, transforms=data_transforms)
        return dataset_val
