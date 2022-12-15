from functools import partial
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule
from braincog.datasets import is_dvs_data
from braincog.base.connection.layer import *
from braincog.base.connection .layer import *
from braincog.base.strategy.LateralInhibition import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import torchattacks
import copy
import torch.optim as optim

DATA_DIR = "../data"
device = torch.device('cuda')

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
    


@register_model
class LeSNN(BaseModule):
    def __init__(self,
                init_channel = 1,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False
        self.num_classes = num_classes
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)
        

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, 10, kernel_size=(5, 5), padding = (0,0), node=self.node, n_preact=self.n_preact),
            nn.MaxPool2d(2),
            BaseConvModule(10, 20, kernel_size=(5, 5), padding = (0,0), node=self.node, n_preact=self.n_preact),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, self.num_classes, bias=False),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)

class STDPConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,groups,
                 tau_decay=torch.exp(-1.0 / torch.tensor(100.0)), offset=0.3, static=True, inh=6.5, avgscale=5):
        super().__init__()
        self.tau_decay = tau_decay
        self.offset = offset
        self.static = static
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups,
                              bias=False)
        self.mem = self.spike = self.refrac_count = None
        self.normweight()
        self.inh = inh
        self.avgscale = avgscale
        self.onespike=True
        self.node=LIFSTDPNode(act_fun=STDPGrad,tau=tau_decay,mem_detach=True)
        self.WTA=WTALayer( )
        self.lateralinh=LateralInhibition(self.node,self.inh,mode="threshold")
        
    def mem_update(self, x, onespike=True):  # b,c,h,w
        x=self.node( x)
        if x.max() > 0:
            x=self.WTA(x)
            self.lateralinh(x)
        self.spike= x 
        return self.spike

    def forward(self, x, T=None, onespike=True):
        if not self.static:
            batch, T, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)
        x = self.conv(  x)
        n = self.getthresh(x)
        self.node.threshold.data = n
        x=x.clamp(min=0)
        x = n / (1 + torch.exp(-(x - 4 * n / 10) * (8 / n)))
        if not self.static:
            x = x.reshape(batch, T, c, h, w)
            xsum = None
            for i in range(T):
                tmp = self.mem_update(x[:, i], onespike).unsqueeze(1)
                if xsum is not None:
                    xsum = torch.cat([xsum, tmp], 1)
                else:
                    xsum = tmp
        else:
            xsum = 0
            for i in range(T):
                xsum += self.mem_update(x, onespike)
        return xsum

    def reset(self):
        self.node.n_reset()
    def normgrad(self, force=False):
        if force:
            min = self.conv.weight.grad.data.min(1, True)[0].min(2, True)[0].min(3, True)[0]
            max = self.conv.weight.grad.data.min(1, True)[0].max(2, True)[0].max(3, True)[0]
            self.conv.weight.grad.data -= min
            tmp = self.offset * max
        else:
            tmp = self.offset * self.spike.mean(0, True).mean(2, True).mean(3, True).permute(1, 0, 2, 3)
        self.conv.weight.grad.data -= tmp
        self.conv.weight.grad.data = -self.conv.weight.grad.data

    def normweight(self, clip=False):
        if clip:
            self.conv.weight.data = torch. \
                clamp(self.conv.weight.data, min=-3, max=1.0)
        else:
            c, i, w, h = self.conv.weight.data.shape
            avg=self.conv.weight.data.mean(1, True).mean(2, True).mean(3, True)
            self.conv.weight.data -=avg
            tmp = self.conv.weight.data.reshape(c, 1, -1, 1)
            self.conv.weight.data /= tmp.std(2, unbiased=False, keepdim=True)


    def getthresh(self, scale):
        tmp2= scale.max(0, True)[0].max(2, True)[0].max(3, True)[0]+0.0001
        return tmp2

class STDPLinear(nn.Module):
    def __init__(self, in_planes, out_planes,
                 tau_decay=0.99, offset=0.05, static=True,inh=10):
        super().__init__()
        self.tau_decay = tau_decay
        self.offset = offset
        self.static = static
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.mem = self.spike = self.refrac_count = None
        self.normweight(False)
        self.threshold = torch.ones(out_planes, device=device) *20
        
        self.inh=inh
        self.node=LIFSTDPNode(act_fun=STDPGrad,tau=tau_decay  ,mem_detach=True)
        self.WTA=WTALayer( )
        self.lateralinh=LateralInhibition(self.node,self.inh,mode="max")
        self.init=False 

    def mem_update(self, x, onespike=True):  # b,c,h,w
        if not self.init: 
            self.node.threshold.data= (x.max(0)[0].detach()*3).to(device) 
            self.init=True
        xori=x
        x=self.node(x)
        if x.max() > 0:
            x=self.WTA(x)
            self.lateralinh(x,xori)
        self.spike=x
        return self.spike

    def forward(self, x, T, onespike=True):
        x = x.detach()
        x = self.linear(x)
        self.x=x.detach()
        xsum = 0
        for i in range(T):
            xsum += self.mem_update(x, onespike)
        return xsum

    def reset(self):
        self.node.n_reset()

    def normgrad(self, force=False):
        if force:
            pass
        else:
            tmp = self.offset * self.spike.mean(0, True).permute(1, 0)
        self.linear.weight.grad.data = -self.linear.weight.grad.data

    def normweight(self, clip=False):
        if clip:
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
        else:
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
            sumweight = self.linear.weight.data.sum(1, True)
            sumweight += (~(sumweight.bool())).float()
            self.linear.weight.data /= self.linear.weight.data.max(1, True)[0] / 0.1

    def getthresh(self, scale):
        tmp = self.linear.weight.clamp(min=0) * scale
        tmp2 = tmp.sum(1, True).reshape(1, -1)
        return tmp2

    def updatethresh(self, plus=0.05):
        self.node.threshold += (plus*self.x * self.spike.detach()).sum(0)
        tmp=self.node.threshold.max()-350
        if tmp>0:
            self.node.threshold-=tmp

class STDPFlatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x, T):  # [batch,T,c,w,h]
        return self.flatten(x)

class STDPMaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, static=True):
        super().__init__()
        self.static = static
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x, T):  # [batch,T,c,w,h]
        x = self.pool(x)
        return x

class Normliaze(nn.Module):
    def __init__(self, num_features, static=True):
        super().__init__()
        self.num_features = num_features
        self.static = static
        self.bn = nn.BatchNorm2d(self.num_features)

    def forward(self, x, T):  # [batch,T,c,w,h]
        # print(x.shape)
        # print(x)
        x /= x.max(1, True)[0].max(2, True)[0].max(3, True)[0]
        # print(x)
        # x = self.bn(x)
        return x

class LeSNN_STDP(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(LeSNN_STDP, self).__init__()
        self.conv = nn.ModuleList([
            STDPConv(in_channel, num_classes, 5, 1, 0,1, static=True, inh=1.625, avgscale=5 ),
            Normliaze(10),
            STDPMaxPool(2, 2, 0, static=True),
            STDPConv(10, 20, 5, 1, 0,1, static=True, inh=1.625, avgscale=5 ),
            Normliaze(20),
            STDPMaxPool(2, 2, 0, static=True),
            STDPFlatten(start_dim=1),
            STDPLinear(320, 10, static=True,inh=25),
        ])

    def forward(self, x, inlayer, outlayer, T, onespike=True):  # [b,t,w,h]
        for i in range(inlayer, outlayer + 1):
            x = self.conv[i](x, T)
        return x

    def normgrad(self, layer, force=False):
        self.conv[layer].normgrad(force)

    def normweight(self, layer, clip=False):
        self.conv[layer].normweight(clip)

    def updatethresh(self, layer, plus=0.05):
        self.conv[layer].updatethresh(plus)

    def reset(self, layer):
        if isinstance(layer, list):
            for i in layer:
                self.conv[i].reset()
        else:
            self.conv[layer].reset()

# model_param[0:5] = conv1, bn1, conv2, bn2, fc

def load_stdp_snn(model_param):
    model = LeSNN_STDP().cuda()
    param_idx = 0
    for layer_idx, param in enumerate(model.named_parameters()):
        if layer_idx in [0, 3, 4, 5, 8, 9, 10]:
            param[1].data = model_param[param_idx]
            param_idx +=1
    return model

def load_bptt_snn(model_param):
    model = LeSNN().cuda()
    param_idx = 0
    for layer_idx, param in enumerate(model.named_parameters()):
        if layer_idx in [0, 1, 2, 5, 6, 7, 10]:
            param[1].data = model_param[param_idx]
            param_idx +=1
    return model

def load_cnn(model_param):
    model = LeNet().cuda()
    param_idx = 0
    for layer_idx, param in enumerate(model.named_parameters()):
        if layer_idx in [0, 1, 2, 3, 4, 5, 6]:
            param[1].data = model_param[param_idx]
            param_idx +=1
    return model

def get_bptt_snn_para(model):
    model_param=[]
    for layer_idx, param in enumerate(model.named_parameters()):
        if layer_idx in [0, 1, 2, 5, 6, 7, 10]:
            model_param.append(param[1].data)
    return model_param

def get_stdp_snn_para(model):
    model_param=[]
    for layer_idx, param in enumerate(model.named_parameters()):
        if layer_idx in [0, 3, 4, 5, 8, 9, 10]:
            model_param.append(param[1].data)
    return model_param


def get_cnn_para(model):
    model_param=[]
    for layer_idx, param in enumerate(model.named_parameters()):
        if layer_idx in [0, 1, 2, 3, 4, 5, 6]:
            model_param.append(param[1].data)
    return model_param
    

class SlowSleepBlock(object):
    def __init__(self, train_loader, sleep_depth=2, init_spike_rate=1/100, sub_sleeplength=1, rate_decrease_ratio=2):
        self.model = None
        self.sleep_depth = sleep_depth
        self.rate_decrease_ratio = rate_decrease_ratio
        self.init_spike_rate = init_spike_rate
        self.sub_sleeplength = sub_sleeplength
        self.train_loader = train_loader

    def slowsleep(self, param):
        print('####### Slow-wave Sleep #######')
        self.model = load_stdp_snn(param)
        #Decrease spike rate
        for depth_idx in range(self.sleep_depth):
            print("Deep Sleep Layer{}".format(depth_idx))
            spike_rate = self.init_spike_rate/(self.rate_decrease_ratio*(2**depth_idx))
            self.subslowsleep(self.sub_sleeplength, spike_rate)
        #Increase spike rate
        for depth_idx in range(self.sleep_depth-2, -1, -1):
            print("Deep Sleep Layer{}".format(depth_idx))
            spike_rate = self.init_spike_rate/(self.rate_decrease_ratio*(2**depth_idx))
            self.subslowsleep(self.sub_sleeplength, spike_rate)
        return get_stdp_snn_para(self.model) 
    
    def subslowsleep(self, sleep_length, spike_rate):
        for epoch_idx in range(sleep_length):
            print("Epoch: {}".format(epoch_idx))
            self.train_epoch_stdp(T=int(1/spike_rate))
        return 
    
    def train_epoch_stdp(self, T):
        self.model.train()
        convlist = [index for index, i in enumerate(self.model.conv) if isinstance(i, (STDPConv, STDPLinear))]
        lr = 0.001
        for layer in range(len(convlist)-1):
            optimizer = torch.optim.SGD(list(self.model.parameters())[layer:layer + 1], lr=lr)
            for step, [imgs, targets] in enumerate(self.train_loader):
                imgs = imgs.cuda()
                targets = targets.cuda()
                spikes = self.model(imgs, 0, convlist[layer], T)
                optimizer.zero_grad()
                spikes.sum().backward(torch.tensor(1/ (spikes.shape[0] * spikes.shape[2] * spikes.shape[3])))
                # spikes.sum().backward()
                self.model.conv[convlist[layer]].spike = spikes.detach()
                self.model.normgrad(convlist[layer], force=True)
                optimizer.step()
                self.model.normweight(convlist[layer], clip=False)
                self.model.reset(convlist)
        layer = len(convlist) - 1
        plus = 0.002
        lr = 0.0001
        optimizer = torch.optim.SGD(list(self.model.parameters())[layer:], lr=lr)
        for step, (imgs, targets) in enumerate(self.train_loader):
            imgs = imgs.cuda()
            targets = targets.cuda()
            spikes = self.model(imgs, 0, convlist[layer], T)
            optimizer.zero_grad()
            spikes.sum().backward()
            self.model.conv[convlist[layer]].spike = spikes.detach()
            self.model.normgrad(convlist[layer], force=False)
            optimizer.step()
            self.model.updatethresh(convlist[layer], plus=plus)
            self.model.normweight(convlist[layer], clip=False)
            spikes = spikes.reshape(spikes.shape[0], 1, -1).detach()
        return

class FastSleepBlock(object):
    def __init__(self, train_loader, attacked_model, add_noise=True):
        self.model = None
        self.train_loader = train_loader
        self.attacked_model = attacked_model
        self.add_noise = add_noise

    def fastsleep(self, param):
        print('####### Fast-wave Sleep #######')
        self.model = load_bptt_snn(param)
        if self.add_noise:
            self.random_advtrain()
        else:
            self.bptt_train()
        return get_bptt_snn_para(self.model)

    def random_advtrain(self, advtrain_ratio=0.5, eps=128/255, train_epoch=5):
        self.model.cuda()
        self.model.train()
        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.SGD(self.model.parameters(), lr=0.1)
        for epoch in range(train_epoch):
            print("Epoch: {}".format(epoch))
            for step, [imgs, targets] in enumerate(self.train_loader):
                imgs = imgs.cuda()
                targets = targets.cuda()
                attacked_imgs = torchattacks.FGSM(self.attacked_model, eps = eps)(imgs, targets)
                if random.random() < advtrain_ratio:
                    outputs = self.model(attacked_imgs)
                else:
                    outputs = self.model(imgs)
                loss = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return
    
    def bptt_train(self, train_epoch=5):
        self.model.cuda()
        self.model.train()
        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.SGD(self.model.parameters(), lr=0.1)
        for epoch in range(train_epoch):
            for step, [imgs, targets] in enumerate(self.train_loader):
                imgs = imgs.cuda()
                targets = targets.cuda()
                outputs = self.model(imgs)
                loss = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return

class DeepSleepDefense(object):
    def __init__(self, model, train_loader, cycle_num = 1):
        init_model = copy.deepcopy(model)
        self.model = init_model
        self.train_loader = train_loader
        self.cycle_num = cycle_num

    def sleep_defence(self, sleep_depth=2, init_spike_rate=1/100, sub_sleeplength=1, rate_decrease_ratio=2, add_noise=True):
        dataflow = get_cnn_para(self.model)
        fast_sleep = []
        slow_sleep = []
        for idx in range(self.cycle_num):
            fast_sleep.append(FastSleepBlock(self.train_loader, self.model,  add_noise=add_noise))
            slow_sleep.append(SlowSleepBlock(self.train_loader, sleep_depth=sleep_depth, init_spike_rate=init_spike_rate, 
                                             sub_sleeplength=sub_sleeplength, rate_decrease_ratio=rate_decrease_ratio))
            dataflow = slow_sleep[idx].slowsleep(dataflow)
            dataflow = fast_sleep[idx].fastsleep(dataflow)
        advmodel = load_cnn(dataflow)
        return advmodel

class SlowWaveOnly(object):
    def __init__(self, model, train_loader, cycle_num = 1):
        init_model = copy.deepcopy(model)
        self.model = init_model
        self.train_loader = train_loader
        self.cycle_num = cycle_num

    def sleep_defence(self, sleep_depth=2, init_spike_rate=1/100, sub_sleeplength=1, rate_decrease_ratio=2):
        dataflow = get_cnn_para(self.model)
        slow_sleep = SlowSleepBlock(self.train_loader, sleep_depth=sleep_depth, init_spike_rate=init_spike_rate, 
                                             sub_sleeplength=sub_sleeplength, rate_decrease_ratio=rate_decrease_ratio)
        dataflow = slow_sleep.slowsleep(dataflow)
            
        advmodel = load_cnn(dataflow)
        return advmodel

class FastWaveOnly(object):
    def __init__(self, model, train_loader, cycle_num = 1):
        init_model = copy.deepcopy(model)
        self.model = init_model
        self.train_loader = train_loader
        self.cycle_num = cycle_num

    def sleep_defence(self, add_noise=True):
        dataflow = get_cnn_para(self.model)
        fast_sleep = FastSleepBlock(self.train_loader, self.model,  add_noise=add_noise)
        dataflow = fast_sleep.fastsleep(dataflow)
        advmodel = load_cnn(dataflow)
        return advmodel


def train_epoch_bp(model, epoch_idx):
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

def train_epoch_stdp(model, epoch_idx, T):
    model.train()
    convlist = [index for index, i in enumerate(model.conv) if isinstance(i, (STDPConv, STDPLinear))]
    lr = 0.1
    for layer in range(len(convlist)-1):
        optimizer = torch.optim.SGD(list(model.parameters())[layer:layer + 1], lr=lr)
        for step, [imgs, targets] in enumerate(train_loader):
            imgs = imgs.cuda()
            targets = targets.cuda()
            spikes = model(imgs, 0, convlist[layer], T)
            optimizer.zero_grad()
            spikes.sum().backward(torch.tensor(1/ (spikes.shape[0] * spikes.shape[2] * spikes.shape[3])))
            # spikes.sum().backward()
            model.conv[convlist[layer]].spike = spikes.detach()
            model.normgrad(convlist[layer], force=True)
            optimizer.step()
            model.normweight(convlist[layer], clip=False)
            model.reset(convlist)
    layer = len(convlist) - 1
    plus = 0.001
    lr = 0.001
    optimizer = torch.optim.SGD(list(model.parameters())[layer:], lr=lr)
    for step, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.cuda()
        targets = targets.cuda()
        spikes = model(imgs, 0, convlist[layer], T)
        optimizer.zero_grad()
        spikes.sum().backward()
        model.conv[convlist[layer]].spike = spikes.detach()
        model.normgrad(convlist[layer], force=False)
        optimizer.step()
        model.updatethresh(convlist[layer], plus=plus)
        model.normweight(convlist[layer], clip=False)
        spikes = spikes.reshape(spikes.shape[0], 1, -1).detach()
    return


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

def generate_adv_samples(target_model, attack_form='FGSM', eps=128/255):
    attack = {'FGSM': torchattacks.FGSM(target_model, eps = eps), 
              'PGD': torchattacks.PGD(target_model, eps = eps), 
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



if __name__ == '__main__':
    dataset_name = 'MNIST'
    batch_size = 256
    learning_rate = 0.1
    epoch = 3


    train_datasets, num_classes = build_dataset(True, dataset_name, DATA_DIR)
    test_datasets, _ = build_dataset(False, dataset_name, DATA_DIR)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, pin_memory=True, drop_last=False, num_workers=8)
    

    CNN_model = LeNet().cuda()
    SNN_model = LeSNN().cuda()
    STDP_model = LeSNN_STDP().cuda()

    # print(CNN_model)
    # print(SNN_model)
    # print(STDP_model)

    #Train CNN
    print('############### Train CNN #################')
    optimizer = optim.SGD(CNN_model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    # loop = tqdm(range(epoch), total=epoch, desc='tarin')
    epoch_idx=0
    
    for epoch_item in range(epoch):
        print("----------epoch {} ------------".format(epoch_idx))
        train_loss_epoch = train_epoch_bp(CNN_model, epoch_idx=epoch_idx)
        test_loss_epoch, test_acc_epoch = test(CNN_model)
        epoch_idx+=1
    
    deepsleep = DeepSleepDefense(model=CNN_model,train_loader=train_loader)
    advmodel = deepsleep.sleep_defence()

    #Train SNN BP
    print('############### Train SNN BP #################')
    optimizer = optim.SGD(SNN_model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    # loop = tqdm(range(epoch), total=epoch, desc='tarin')
    epoch_idx=0
    
    for epoch_item in range(epoch):
        print("----------epoch {} ------------".format(epoch_idx))
        train_loss_epoch = train_epoch_bp(SNN_model, epoch_idx=epoch_idx)
        test_loss_epoch, test_acc_epoch = test(SNN_model)
        epoch_idx+=1

    

    FGSM_adv_samples = generate_adv_samples(CNN_model, 'FGSM')
    PDG_adv_samples = generate_adv_samples(CNN_model, 'PGD')
    CW_adv_samples = generate_adv_samples(CNN_model, 'CW')
    BIM_adv_samples = generate_adv_samples(CNN_model, 'BIM')
    AutoAttack_adv_samples = generate_adv_samples(CNN_model, 'AutoAttack')
    print("===============================================================")
    print("START TESTING")
    print("===============================================================")
    print('===== CNN ==========')
    print("WITHOUT ATTACK:")
    test_loss, test_acc = test(CNN_model)
    print("WITH ATTACK:FGSM")
    test_acc = test_attacked(CNN_model, FGSM_adv_samples)
    print("WITH ATTACK:PGD")
    test_acc = test_attacked(CNN_model, PDG_adv_samples)
    print("WITH ATTACK:CW")
    test_acc = test_attacked(CNN_model, CW_adv_samples)
    print("WITH ATTACK:BIM")
    test_acc = test_attacked(CNN_model, BIM_adv_samples)
    print("WITH ATTACK:AutoAttack")
    test_acc = test_attacked(CNN_model, AutoAttack_adv_samples)
    print('===== SNN ==========')
    print("WITHOUT ATTACK:")
    test_loss, test_acc = test(SNN_model)
    print("WITH ATTACK:FGSM")
    test_acc = test_attacked(SNN_model, FGSM_adv_samples)
    print("WITH ATTACK:PGD")
    test_acc = test_attacked(SNN_model, PDG_adv_samples)
    print("WITH ATTACK:CW")
    test_acc = test_attacked(SNN_model, CW_adv_samples)
    print("WITH ATTACK:BIM")
    test_acc = test_attacked(SNN_model, BIM_adv_samples)
    print("WITH ATTACK:AutoAttack")
    test_acc = test_attacked(SNN_model, AutoAttack_adv_samples)
    print('===== DeepSleep ==========')
    
    print("WITHOUT ATTACK:")
    test_loss, test_acc = test(advmodel)
    print("WITH ATTACK:FGSM")
    test_acc = test_attacked(advmodel, FGSM_adv_samples)
    print("WITH ATTACK:PGD")
    test_acc = test_attacked(advmodel, PDG_adv_samples)
    print("WITH ATTACK:CW")
    test_acc = test_attacked(advmodel, CW_adv_samples)
    print("WITH ATTACK:BIM")
    test_acc = test_attacked(advmodel, BIM_adv_samples)
    print("WITH ATTACK:AutoAttack")
    test_acc = test_attacked(advmodel, AutoAttack_adv_samples)
    


    

    # #Train SNN STDP
    # print('############### Train SNN STDP #################')
    # epoch_idx=0
    # for epoch_item in range(epoch):
    #     print("----------epoch {} ------------".format(epoch_idx))
    #     train_loss_epoch = train_epoch_stdp(STDP_model, T=100, epoch_idx=epoch_idx)
    #     # test_loss_epoch, test_acc_epoch = test(STDP_model)
    #     epoch_idx+=1



    
    # print('###############################################################')
    # print('After Train')
    # print("=============================CNN===============================")
    # for layer_idx, param in enumerate(CNN_model.named_parameters()):
    #     print(layer_idx)
    #     print(param[1].size())
    #     print(param[0])
    #     print(param[1].data)
    #     print('-------------------------------------------------')
    # print("=============================SNN BP===============================")
    # for layer_idx, param in enumerate(SNN_model.named_parameters()):
    #     print(layer_idx)
    #     print(param[1].size())
    #     print(param[0])
    #     print(param[1].data)
    #     print('-------------------------------------------------')
    # print("=============================SNN STDP===============================")
    # for layer_idx, param in enumerate(STDP_model.named_parameters()):
    #     print(layer_idx)
    #     print(param[1].size())
    #     print(param[0])
    #     print(param[1].data)
    #     print('-------------------------------------------------')
