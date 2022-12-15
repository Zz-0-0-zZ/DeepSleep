# -*- coding: utf-8 -*-
"""
@author: zhaozhengyue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
import os

DIST_MODEL_DIR = './defense_models/distillation_model'
DIST_MODEL_FILE_NAME = 'DistLeNet'

class LeNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=10, temperature=10):
        super(LeNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 10, kernel_size=5, bias=False),
            nn.BatchNorm2d(10), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(10, 20, kernel_size=5, bias=False),
            nn.BatchNorm2d(20), nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, num_classes, bias=False)
        )
        self.T = temperature
        
    def forward(self, x):
        x = self.feature(x)
        x= self.fc(x)
        x = self.softmaxT(x)
        return x
    
    def softmaxT(self, x):
        x_exp = (x/self.T).exp()
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp/partition
    
    def change_temperature(self, temperature):
        self.T = temperature  

class Distillation(object):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
    
    def distillation_train(self, temperature=10, train_epoch=10):
        distmodel = LeNet(temperature=temperature)
        distmodel.cuda()
        adv_model = LeNet(temperature=temperature)
        adv_model.change_temperature(temperature)
        adv_model.cuda()
        
        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.SGD(adv_model.parameters(), lr=0.1)
        if DIST_MODEL_FILE_NAME+'_{}'.format(temperature)+'.pth' in os.listdir(DIST_MODEL_DIR):
            distmodel.load_state_dict(torch.load(DIST_MODEL_DIR+'/'+DIST_MODEL_FILE_NAME+'_{}'.format(temperature)+'.pth'))
            distmodel.cuda()
        else:
            # TRAIN DISTILLATION MODEL
            distmodel.train()
            adv_model.eval()
            for epoch in range(10):
                for step, [imgs, targets] in enumerate(self.train_loader):
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                    outputs = distmodel(imgs)
                    loss = loss_fn(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            torch.save(distmodel.state_dict(), DIST_MODEL_DIR+'/'+DIST_MODEL_FILE_NAME+'_{}'.format(temperature)+'.pth')

        # START DISTILLATION TRAIN
        distmodel.eval()
        adv_model.train()
        for epoch in range(train_epoch):
            for step, [imgs, targets] in enumerate(self.train_loader):
                imgs = imgs.cuda()
                targets = targets.cuda()
                soft_outputs = distmodel(imgs)
                outputs = adv_model(imgs)
                loss = loss_fn(outputs, soft_outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        adv_model.change_temperature(1)
        return adv_model
