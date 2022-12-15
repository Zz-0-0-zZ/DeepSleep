# -*- coding: utf-8 -*-
"""
@author: zhaozhengyue
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.cnn_models import LeNet, CNN5, CNN7, VGGNet, AlexNet, GoogleNet, ResNet
import argparse
import time
from torchvision import datasets, transforms
import os
import torch.nn.functional as F
import torchattacks
from defense_models.advtrain import FGSMTrain, PGDTrain
from defense_models.distillation import Distillation
from defense_models.sleep import SleepDefense
from defense_models.deepsleep import DeepSleepDefense, SlowWaveOnly, FastWaveOnly

MODEL_DIR = "./model_save"
DATA_DIR = "./data"



def train_epoch(epoch_idx):
    model.train()
    for step, [imgs, targets] in enumerate(train_loader):
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step = len(train_loader)*epoch_idx+step+1
        if train_step % 100 == 0:
            print("train time：{}, Loss: {}".format(train_step, loss.item()))
    print("train Loss: {}".format(loss.item()))
    return float(loss.item())

def test(model):
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("test set Loss: {}".format(total_test_loss))
    print("test set accuracy: {}".format(total_accuracy/len(test_datasets)))
    return float(total_test_loss), float(total_accuracy/len(test_datasets))

def test_attacked(adv_model, adv_test_set):
    adv_model.eval()
    total_accuracy = 0
    for adv_imgs, targets in adv_test_set:
        adv_imgs = adv_imgs.cuda()
        targets = targets.cuda()
        outputs = adv_model(adv_imgs)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
    print("test set accuracy: {}".format(total_accuracy/len(test_datasets)))
    return float(total_accuracy/len(test_datasets))

def generate_adv_samples(target_model, attack_form='FGSM', eps=128/255, steps=100):
    attack = {'FGSM': torchattacks.PGD(target_model, eps = eps), 
              'PGD': torchattacks.FGSM(target_model, eps = eps), 
              'CW': torchattacks.CW(target_model, steps=100),
              'BIM': torchattacks.BIM(target_model, eps = eps),
              'DeepFool': torchattacks.DeepFool(target_model),
              'AutoAttack': torchattacks.AutoAttack(target_model, eps = eps),
              'OnePixel': torchattacks.OnePixel(target_model)}
    adv_test_set = []
    for imgs, targets in test_loader:
        imgs = imgs.cuda()
        targets = targets.cuda()
        adv_imgs = attack[attack_form](imgs, targets)
        adv_test_set.append([adv_imgs, targets])
    return adv_test_set



def build_dataset(is_train, dataset, path):
    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            path, train=is_train, transform=transforms.Compose([
            transforms.ToTensor(),
            ]), download=True)
        num_classes = 10
    elif dataset == 'CIFAR100':
        dataset = datasets.CIFAR100(
            path, train=is_train, transform=transforms.Compose([
            transforms.ToTensor(),
            ]), download=True)
        num_classes = 100
    elif dataset == 'MNIST':
        dataset = datasets.MNIST(
            path, train=is_train, transform=transforms.Compose([
            transforms.ToTensor(),
            ]), download=True)
        num_classes = 10
    else:
        raise NotImplementedError
    return dataset, num_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN Training and Evaluating')
    parser.add_argument('--model', default='CNN5', type=str)
    parser.add_argument('--lr', default=0.1, type=float, help="Learning Rate")
    parser.add_argument('--batchsize', default=128, type=int, help="Batch Size")
    parser.add_argument('--epoch', default=10, type=int, help="Epoch Num")
    parser.add_argument('--dataset', default='CIFAR10', type=str, help="DataSet")
    parser.add_argument('--defense', default='NoDefense', type=str)
    parser.add_argument('--noisescale', default=128, type=int)
    # paraser.add_argument
    args = parser.parse_args()
    cnn_model=['LeNet', 'CNN5', 'CNN7', 'VGGNet', 'ResNet', 'AlexNet', 'GoogleNet']
    learning_rate = args.lr
    epoch = args.epoch
    batch_size = args.batchsize
    model_name = args.model
    dataset_name = args.dataset
    defense  =args.defense
    noisescale = args.noisescale
    task_name = model_name+"_{}".format(dataset_name)+"_Defense={}".format(defense)+"_NoiseScale={}".format(noisescale)+"_lr={}".format(learning_rate)+"_epoch={}".format(epoch)+"_bs={}".format(batch_size)
    model_file_name = model_name+"_{}".format(dataset_name)+"_lr={}".format(learning_rate)+"_epoch={}".format(epoch)+"_bs={}".format(batch_size)
    print("###############################################################")
    print('Ablation Experiment')
    print("###############################################################")
    print(task_name)
    print("###############################################################")

    train_datasets, num_classes = build_dataset(True, dataset_name, DATA_DIR)
    test_datasets, _ = build_dataset(False, dataset_name, DATA_DIR)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, pin_memory=True, drop_last=False, num_workers=8)
    
    if dataset_name != "MNIST" or model_name != "LeNet":
        print("Support MNIST and LeNet only!")
        exit(0)

    if dataset_name == "MNIST":
        in_channel = 1
    else:
        in_channel = 3

    if model_name == "LeNet":
        model = LeNet(in_channel=in_channel, num_classes=num_classes)
    elif model_name == "CNN5":
        model = CNN5(in_channel=in_channel, num_classes=num_classes)
    elif model_name == "CNN7":
        model = CNN7(in_channel=in_channel, num_classes=num_classes)
    elif model_name == "VGGNet":
        model = VGGNet(in_channel=in_channel, num_classes=num_classes)
    elif model_name == "AlexNet":
        model = AlexNet(in_channel=in_channel, num_classes=num_classes)
    elif model_name == "GoogleNet":
        model = GoogleNet(in_channel=in_channel, num_classes=num_classes)
    elif model_name == "ResNet":
        model = ResNet(in_channel=in_channel, num_classes=num_classes)
    
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    
    model_files = os.listdir(MODEL_DIR)
    start = time.time()
    if model_file_name+'.pth' in model_files:
        print("Model has been trained. Loading Model ...")
        model.load_state_dict(torch.load(MODEL_DIR+'/'+model_file_name+'.pth'))
        model = model.cuda()
    else: 
        model = model.cuda()
        loop = tqdm(range(epoch), total=epoch, desc='tarin')
        epoch_idx=0
        train_loss = []
        test_loss = []
        test_acc = []
        
        for epoch_item in loop:
            print("----------epoch {} ------------".format(epoch_idx))
            train_loss_epoch = train_epoch(epoch_idx=epoch_idx)
            test_loss_epoch, test_acc_epoch = test(model)
            train_loss.append(train_loss_epoch)
            test_loss.append(test_loss_epoch)
            test_acc.append(test_acc_epoch)
            epoch_idx+=1
        torch.save(model.state_dict(), "model_save/"+model_file_name+".pth")
    end = time.time()
    #Generate Adversarial Samples
    FGSM_adv_samples = generate_adv_samples(model, 'FGSM', eps=noisescale/255)
    PDG_adv_samples = generate_adv_samples(model, 'PGD', eps=noisescale/255)
    CW_adv_samples = generate_adv_samples(model, 'CW', steps=int(200*noisescale/128))
    BIM_adv_samples = generate_adv_samples(model, 'BIM', eps=noisescale/255)
    AutoAttack_adv_samples = generate_adv_samples(model, 'AutoAttack', eps=noisescale/255)

    print("===============================================================")
    print("START TESTING")
    print("===============================================================")
    print("WITHOUT ATTACK:")
    test_loss, test_acc = test(model)
    print("WITH ATTACK:FGSM")
    test_acc = test_attacked(model, FGSM_adv_samples)
    print("WITH ATTACK:PGD")
    test_acc = test_attacked(model, PDG_adv_samples)
    print("WITH ATTACK:CW")
    test_acc = test_attacked(model, CW_adv_samples)
    print("WITH ATTACK:BIM")
    test_acc = test_attacked(model, BIM_adv_samples)
    print("WITH ATTACK:AutoAttack")
    test_acc = test_attacked(model, AutoAttack_adv_samples)

 
    if defense == 'SlowWaveOnly':
        defense_model = SlowWaveOnly(model, train_loader=train_loader, cycle_num = 1)
        robust_model = defense_model.sleep_defence(sleep_depth=2, init_spike_rate=1/100, 
                                sub_sleeplength=1, rate_decrease_ratio=2)
    elif defense == 'FastWaveOnly':
        defense_model = FastWaveOnly(model, train_loader=train_loader, cycle_num = 1)
        robust_model = defense_model.sleep_defence(add_noise=True)
    elif defense == 'FastWaveWithoutNoise':
        defense_model = FastWaveOnly(model, train_loader=train_loader, cycle_num = 1)
        robust_model = defense_model.sleep_defence(add_noise=False)
    elif defense == 'DeepSleep':
        defense_model = DeepSleepDefense(model, train_loader=train_loader, cycle_num = 1)
        robust_model = defense_model.sleep_defence(sleep_depth=2, init_spike_rate=1/100, 
                                sub_sleeplength=1, rate_decrease_ratio=2, add_noise=True)
    else:
        robust_model = model
    
    print('============ {} ============'.format(defense))
    print("WITHOUT ATTACK:")
    test_loss, test_acc = test(robust_model)
    print("WITH ATTACK:FGSM")
    test_acc = test_attacked(robust_model, FGSM_adv_samples)
    print("WITH ATTACK:PGD")
    test_acc = test_attacked(robust_model, PDG_adv_samples)
    print("WITH ATTACK:CW")
    test_acc = test_attacked(robust_model, CW_adv_samples)
    print("WITH ATTACK:BIM")
    test_acc = test_attacked(robust_model, BIM_adv_samples)
    print("WITH ATTACK:AutoAttack")
    test_acc = test_attacked(robust_model, AutoAttack_adv_samples)
