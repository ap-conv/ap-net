#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil
from torch.nn import L1Loss, MultiLabelSoftMarginLoss, BCELoss
from torch import nn
import torch.nn.functional as F
from .reg import regularize

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def get_corrects(probs, labels):
    _, preds1 = torch.max(probs, 1)
    _, preds5 = torch.topk(probs, 5, 1)
    batch_corrects1 = torch.sum((preds1 == labels)).data.item()
    batch_corrects5 = torch.sum((preds5 ==  labels.unsqueeze(1).expand_as(preds5))).data.item()
    return batch_corrects1, batch_corrects5

def train(cfg,
          model,
          epoch_num,
          optimizer,
          criterion,
          scheduler,
          data_set,
          data_loader,
          save_dir,
          ):

    logfile = os.path.join(save_dir, 'train.log')
    for epoch in range(1, epoch_num + 1):
        # train phase
        scheduler.step()
        model = [e.train(True) for e in model]
        extractor, classifier_base, classifier_swap = model
        L = len(data_loader['train'])
        logging.info('current backbone lr: %f' % optimizer.param_groups[0]['lr'])
        if epoch < 30:
            reg_weight = 1e-5
        elif 30 <= epoch < 60:
            reg_weight = 1e-6
        else:
            reg_weight = 1e-7
        for batch_cnt, data in enumerate(data_loader['train']):
            step+=1
            batch_start = time.time()
            model = [e.train(True) for e in model]
            imgs, imgs_swap, labels = data

            imgs = Variable(imgs.cuda())
            imgs_swap = Variable(imgs_swap.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            N = imgs.size(0)
            optimizer.zero_grad()

            f_base = extractor(imgs, 0)
            probs = classifier_base(f_base)
            cls_loss_unswap = criterion(probs, labels)

            f_swap = extractor(imgs_swap, 1)
            swap_probs = classifier_swap(f_swap)
            cls_loss_swap = criterion(swap_probs, labels)

            f_base_1, f_base_2 = torch.split(f_base, [1024, 1024], dim=1)

            reg_loss = regularize(f_base_1, f_base_2) 
            loss = cls_loss_unswap + cls_loss_swap + reg_loss * reg_weight
            
            torch.cuda.synchronize()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            batch_end = time.time()
            stt = '[TRAIN]: Epoch {:03d} / {:03d}    |    Batch {:03d} / {:03d}    |    Loss {:6.4f} + {:6.4f} + {:6.4f} * {:.0e} = {:6.4f}   |   cost {:.2f}   |   lr {:f}'.format(epoch, epoch_num, batch_cnt+1, L, cls_loss_unswap.data.item(), cls_loss_swap.data.item(), reg_loss.data.item(), reg_weight, loss.data.item(), batch_end - batch_start, optimizer.param_groups[0]['lr'])
            logging.info(stt)
            with open(logfile, 'a') as f:
                f.write(stt + '\n')
        logging.info('current backbone lr: %f' % optimizer.param_groups[0]['lr'])
        # val phase
        model = [e.train(False) for e in model]

        val_loss = 0
        val_corrects1 = 0
        val_corrects5 = 0
        val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

        t0 = time.time()
        LL = len(data_loader['val'])
        for batch_cnt_val, data_val in enumerate(data_loader['val']):
            # print data
            inputs, labels = data_val
            #print(labels)

            with torch.no_grad():
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
                f_base = extractor(inputs, 0)

                probs_base = classifier_base(f_base)

                probs = probs_base

                loss = criterion(probs, labels)
                t1, t5 = get_corrects(probs, labels)
                val_corrects1 += t1
                val_corrects5 += t5
            val_loss += loss.data.item()
            logging.info('[TEST]: Batch {:03d} / {:03d}'.format(batch_cnt_val+1, LL))
        val_loss = val_loss / val_size
        val_acc1 = 1.0 * val_corrects1 / len(data_set['val'])
        val_acc5 = 1.0 * val_corrects5 / len(data_set['val'])

        t1 = time.time()
        since = t1-t0
        logging.info('--'*30)
        logging.info('current backbone lr:%f' % optimizer.param_groups[0]['lr'])

        logging.info('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||val-acc@5: %.4f ||time: %d'
                        % (dt(), epoch, val_loss, val_acc1, val_acc5, since))

        # save model
        save_path = os.path.join(save_dir,
                'weights-%d-%d-[%.4f].pth'%(epoch,batch_cnt,val_acc1))
        time.sleep(10)
        torch.save([e.state_dict() for e in model], save_path)
        logging.info('saved model to %s' % (save_path))
        logging.info('--' * 30)

