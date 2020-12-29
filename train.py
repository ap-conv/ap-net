#coding=utf-8
import os
import os.path as osp
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.dataset_index import collate_fn1, collate_fn2, collate_fn4train, collate_fn4val, dataset
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from torchvision import datasets, models#, transforms
from transforms import transforms
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util_destructionL import train, trainlog
from  torch.nn import CrossEntropyLoss, KLDivLoss
import logging
from models.sep_ir50_imagenet_destructionL import APIResNet50 as Extractor
from models.classifier import Classifier
from PIL import Image
import argparse
from RandAugment.augmentations import RandAugment

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser('Augmentation Pathway - IResNet50')
parser.add_argument('-M', type=int, required=True)
parser.add_argument('-N', type=int, required=True)
parser.add_argument('--desc', default='_')
args = parser.parse_args()

print("RandAug with M={} N={}".format(args.M, args.N))
print("Without AutoResume")
cfg = {}
time = datetime.datetime.now()
# set dataset, include{CUB_200_2011: CUB, Stanford car: STCAR, JDfood: FOOD}
print("USE DATASET           <<< {} >>>".format('IMAGE'))
cfg['dataset'] = 'IMAGE'

# prepare dataset
if cfg['dataset'] == 'IMAGE':
    rawdata_root = './datasets/imagenet/all'
    train_pd = pd.read_csv("./datasets/imagenet/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
    test_pd = pd.read_csv("./datasets/imagenet/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
    cfg['numcls'] = 1000
    numimage = train_pd.shape[0]


print('Dataset:',cfg['dataset'])
print('train images:', train_pd.shape)
print('test images:', test_pd.shape)
print('num classes:', cfg['numcls'])
print("********************************************************")



print('Set transform')
data_transforms = {  
    'aug_1': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]),
    'aug_2': transforms.Compose([
        RandAugment(args.N, args.M),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]),
    'totensor': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valtotensor': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'None': transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop((224)),
    ]),

}
data_set = {}
data_set['train'] = dataset(cfg,imgroot=rawdata_root,anno_pd=train_pd,
                       aug_1=data_transforms["aug_1"],data_transforms=aug_2["aug_2"],totensor=data_transforms["totensor"],train=True
                       )
data_set['val'] = dataset(cfg,imgroot=rawdata_root,anno_pd=test_pd,
                       aug_1=data_transforms["None"],data_transforms=aug_2["None"],totensor=data_transforms["valtotensor"],train=False
                       )
dataloader = {}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=256,
                                           shuffle=True, num_workers=32, collate_fn=collate_fn4train)
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=256,
                                           shuffle=False, num_workers=32, collate_fn=collate_fn4val)

print('done')
print('**********************************************')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = [Extractor(sep_num=2), Classifier(2048, cfg['numcls']), Classifier(1024, cfg['numcls'])]
print('swap + 2 loss')
model = [e.cuda() for e in model]
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = [nn.DataParallel(e) for e in model]

base_lr = 0.1 # rand_aug: batch_size 512/1024, lr 0.1
resume = None

params = []
for idx, m in enumerate(model):
    for key, value in m.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr 
        momentum = 0.9
        weight_decay = 1e-4
        params += [{"params": [value], "lr": lr, "momentum": momentum, "weight_decay": weight_decay}]
optimizer = optim.SGD(params)

criterion = CrossEntropyLoss()
constain_criterion = KLDivLoss()
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones = [30, 60, 80],
                                     gamma=0.1)

train(cfg,
      model,
      epoch_num=90,
      optimizer=optimizer,
      criterion=criterion,
      constain_criterion=constain_criterion,
      scheduler=scheduler,
      data_set=data_set,
      data_loader=dataloader,
      save_dir=save_dir,)

