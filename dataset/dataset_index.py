# coding=utf8
from __future__ import division
import pickle
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageStat
class dataset(data.Dataset):
    def __init__(self, cfg, imgroot, anno_pd, aug_1=None, aug_2=None, totensor=None, train=False):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.aug_1 = aug_1
        self.aug_2 = aug_2
        self.anno_pd = anno_pd
        self.totensor = totensor
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        img_aug_1 = self.aug_1(img)
        label = self.labels[item]

        if self.train:
            img_aug_2 = self.aug_2(img)
            img_aug_1 = self.totensor(img_aug_1)
            img_aug_2 = self.totensor(img_aug_2)
            return img_aug_1, img_aug_2, label
        else:
            img_aug_1 = self.totensor(img_aug_1)
            return img_aug_1, label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

def collate_fn4train(batch):
    imgs = []
    imgs_aug_2 = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        imgs_aug_2.append(sample[1])
        label.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(imgs_aug_2, 0), label

def collate_fn4val(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
    return torch.stack(imgs, 0), label


def collate_fn1(batch):
    imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])
    return torch.stack(imgs, 0), labels

def collate_fn2(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
    return torch.stack(imgs, 0), label
