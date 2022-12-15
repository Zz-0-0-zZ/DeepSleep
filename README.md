# DeepSleep

DeepSleep is a biologically-inspired SNN-based adversarial defense method. [**PAPER**](https://github.com/Zz-0-0-zZ/DeepSleep/blob/main/DeepSleep.pdf)
Experiments run with dataset [**MNIST**](http://yann.lecun.com/exdb/mnist/)
SNN models are built on [**BrainCog**](http://www.brain-cog.network/docs/index.html) framework.
 
## 
 
`main.py`: run general experiments
`ablation_main.py`: run ablation experiments
`./models/cnn_models.py`: CNN models  
`./defense_models/advtrain.py`: Adversarial Train method (Madry A. et al. 2018) [**Paper**](https://arxiv.org/abs/1706.06083)
`./defense_models/distillation.py`: Defense Distillation method (Papemot N. et al. 2016) [**Paper**](https://arxiv.org/pdf/1511.04508)
`./defense_models/sleep.py`: Sleep Defense method (Tadros T. et al. 2019) [**Paper**](https://openreview.net/pdf?id=r1xGnA4Kvr)
`./defense_models/deepsleep.py`: the proposed DeepSleep method

## Running main.py
 
 
### parameters
 
```
--model: LeNet
```

```
--dataset: MNIST
```

```
--defense: FGSNTrain | PGDTrain | DefenseDist | Sleep | DeepSleep
```

```
--noisescale: (int: 0-255) adversarial perturbation noise scale
```

```
--lr: (float) learning rate
```

```
--batchsize: (int) batch size
```

```
--epoch: (int) the number of epochs
```


### Example
 
Run DeepSleep method: `python main.py --dataset MNIST --model LeNet --defense DeepSleep --noisescale 64 --lr 0.1 --batchsize 128 --epoch 10`

  

## Running ablation_main.py 
 ### parameters
 
```
--model: LeNet
```

```
--dataset: MNIST
```

```
--defense: SlowWaveOnly | FastWaveOnly | FastWaveWithoutNoise
```

```
--noisescale: (int: 0-255) adversarial perturbation noise scale
```

```
--lr: (float) learning rate
```

```
--batchsize: (int) batch size
```

```
--epoch: (int) the number of epochs
```

### Example
 
Run DeepSleep method: `python main.py --dataset MNIST --model LeNet --defense SlowWaveOnly --noisescale 64 --lr 0.1 --batchsize 128 --epoch 10`


 
