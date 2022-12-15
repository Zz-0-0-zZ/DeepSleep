# -*- coding: utf-8 -*-
"""
@author: zhaozhengyue
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks
import copy
import random

class FGSMTrain(object):
    def __init__(self, model, advtrain_ratio=0.5):
        self.model = model
        self.advtrain_ratio = advtrain_ratio

    def advtrain(self, train_loader, eps=8/255, train_epoch=10):
        adv_model = copy.deepcopy(self.model)
        adv_model.cuda()
        adv_model.train()
        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.SGD(adv_model.parameters(), lr=0.1)
        for epoch in range(train_epoch):
            for step, [imgs, targets] in enumerate(train_loader):
                imgs = imgs.cuda()
                targets = targets.cuda()
                attacked_imgs = torchattacks.PGD(adv_model, eps = eps)(imgs, targets)
                if random.random() < self.advtrain_ratio:
                    outputs = adv_model(attacked_imgs)
                else:
                    outputs = adv_model(imgs)
                loss = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return adv_model

class PGDTrain(object):
    def __init__(self, model, advtrain_ratio=0.5):
        self.model = model
        self.advtrain_ratio = advtrain_ratio

    def advtrain(self, train_loader, eps=8/255, train_epoch=10):
        adv_model = copy.deepcopy(self.model)
        adv_model.cuda()
        adv_model.train()
        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.SGD(adv_model.parameters(), lr=0.1)
        for epoch in range(train_epoch):
            for step, [imgs, targets] in enumerate(train_loader):
                imgs = imgs.cuda()
                targets = targets.cuda()
                attacked_imgs = torchattacks.FGSM(adv_model, eps = eps)(imgs, targets)
                if random.random() < self.advtrain_ratio:
                    outputs = adv_model(attacked_imgs)
                else:
                    outputs = adv_model(imgs)
                outputs = adv_model(attacked_imgs)
                loss = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return adv_model

        
