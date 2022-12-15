# -*- coding: utf-8 -*-
"""
@author: zhaozhengyue
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
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
        
    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x


class CNN5(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, temperature=1):
        super(CNN5, self).__init__()
        self.num_classes = num_classes
        self.T = temperature
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2*2, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
        )
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

class CNN7(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, temperature=1):
        super(CNN7, self).__init__()
        self.num_classes = num_classes
        self.T = temperature
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
        )
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

# class VGGNet(nn.Module):
#     def __init__(self, in_channel=3, num_classes=10, temperature=1):
#         super(VGGNet, self).__init__()
#         self.T = temperature
#         self.num_classes = num_classes
#         self.feature = nn.Sequential(
#             nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d((2,2)),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d((2,2)),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d((2,2)),
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d((2,2)),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
#             nn.MaxPool2d((2,2)),
#         )
#         self.GAP = nn.AdaptiveMaxPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Linear(512, 4096), nn.Dropout(p=0.2),
#             nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
#             nn.Linear(4096,4096), nn.Dropout(p=0.2),
#             nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
#             nn.Linear(4096,64),  nn.Dropout(p=0.2), nn.ReLU(inplace=True),
#             nn.Linear(64, num_classes), nn.ReLU(inplace=True),
#             nn.BatchNorm1d(self.num_classes),
#         )        

#     def forward(self, x):
#         x = self.feature(x)
#         x = self.GAP(x).squeeze(dim=3).squeeze(dim=2)
#         x = self.fc(x)
#         x = self.softmaxT(x)
#         return x
    
#     def softmaxT(self, x):
#         x_exp = (x/self.T).exp()
#         partition = x_exp.sum(dim=1, keepdim=True)
#         return x_exp/partition
    
#     def change_temperature(self, temperature):
#         self.T = temperature 

class VGGNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, temperature=1):
        super(VGGNet, self).__init__()
        self.T = temperature
        self.num_classes = num_classes
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.GAP = nn.AdaptiveMaxPool2d((1, 1))
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,64)
        self.fc4 = nn.Linear(64, num_classes)
        self.max_pool=nn.MaxPool2d((2,2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        x = self.relu(self.conv1_1(X))
        x = self.relu(self.conv1_2(x))
        x = self.max_pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.max_pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.max_pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.max_pool(x)

        x = self.GAP(x).squeeze(dim=3).squeeze(dim=2)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        feat = self.relu(self.fc3(x))
        x = self.bn3(self.fc4(feat))

        x = self.softmaxT(x)
        return x
    
    def softmaxT(self, x):
        x_exp = (x/self.T).exp()
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp/partition
    
    def change_temperature(self, temperature):
        self.T = temperature  

class AlexNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, temperature=1):
        super(AlexNet, self).__init__()
        self.T = temperature
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 4096), nn.Dropout(p=0.2),                           
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), nn.Dropout(p=0.2),                                  
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),     
            nn.BatchNorm1d(self.num_classes),                      
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = self.softmaxT(x)
        return x
    
    def softmaxT(self, x):
        x_exp = (x/self.T).exp()
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp/partition
    
    def change_temperature(self, temperature):
        self.T = temperature  

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception,self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_planes,n1x1,kernel_size=1),
            nn.BatchNorm2d(n1x1),nn.ReLU(True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes,n3x3red,kernel_size=1),
            nn.BatchNorm2d(n3x3red),nn.ReLU(True),
            nn.Conv2d(n3x3red,n3x3,kernel_size=3,padding=1),
            nn.BatchNorm2d(n3x3),nn.ReLU(True)
        )

        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes,n5x5red,kernel_size=1),
            nn.BatchNorm2d(n5x5red),nn.ReLU(True),
            nn.Conv2d(n5x5red,n5x5,kernel_size=3,padding=1),
            nn.BatchNorm2d(n5x5),nn.ReLU(True),
            nn.Conv2d(n5x5,n5x5,kernel_size=3,padding=1),
            nn.BatchNorm2d(n5x5),nn.ReLU(True)
        )

        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            nn.Conv2d(in_planes,pool_planes,kernel_size=1),
            nn.BatchNorm2d(pool_planes),nn.ReLU(True)
        )

    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1,out2,out3,out4],dim=1)

class GoogleNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, temperature=1):
        super(GoogleNet,self).__init__()
        self.T = temperature
        self.num_classes = num_classes
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channel,192,kernel_size=3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.feature = nn.Sequential(
            Inception(192,64,96,128,16,32,32),
            Inception(256,128,128,192,32,96,64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Inception(480,192,96,208,16,48,64),
            Inception(512,160,112,224,24,64,64),
            Inception(512,128,128,256,24,64,64),
            Inception(512,112,144,288,32,64,64),
            Inception(528,256,160,320,32,128,128),
            Inception(832,256,160,320,32,128,128),
            Inception(832,384,192,384,48,128,128),
            nn.AvgPool2d(kernel_size=16,stride=1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,num_classes),
            nn.BatchNorm1d(self.num_classes),
        )

    def forward(self,x):
        x = self.pre_layer(x)
        x = self.feature(x)
        x = self.fc(x)
        x = self.softmaxT(x)
        return x
    
    def softmaxT(self, x):
        x_exp = (x/self.T).exp()
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp/partition
    
    def change_temperature(self, temperature):
        self.T = temperature  

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(Bottleneck, self).__init__()
        self.feature=nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes*4, kernel_size=1, bias=False),
        )
        self.downsample=downsample
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.feature(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, temperature=1):
        super(ResNet, self).__init__()
        self.T = temperature
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=(7,7), padding=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            Bottleneck(64, 32, 
                downsample=nn.Conv2d(64,128,kernel_size=1)),
            Bottleneck(128,64,
                downsample=nn.Conv2d(128,256,kernel_size=1)),
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.BatchNorm1d(self.num_classes),
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.gap(x).squeeze(dim=3).squeeze(dim=2)
        x = self.fc(x)
        x = self.softmaxT(x)
        return x
    
    def softmaxT(self, x):
        x_exp = (x/self.T).exp()
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp/partition
    
    def change_temperature(self, temperature):
        self.T = temperature  
