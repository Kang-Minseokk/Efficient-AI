# Training Binary Neural Networks in a Binary Weight Space
---
The attached codes were used for the experiments conducted in the submitted paper "Training Binary Neural Networks in a Binary Vector Space".
## How to use
The model training process, which includes the proposed method, is defined in main.py, and can be executed as follows with the command like:
```
python main.py --affine=False --batch_norm_out=True --batch_size=16384 --bias=False --dataset=MNIST --depth=4 --epochs=2000 --hid_dim=128 --lr=10 --model=MLP --training_space=binary --b_act=true --b_weight=true --batch_norm=true --test
```
The above command is for the binary-space training of the large FCN on MNIST, which was used in Experiment 5.1. Please note that due to differences in operating environments, the performance obtained may vary from the values reported in the literature.

### Training Algorithm
You can use each method used in the paper by setting the arguments `--training_space`, `--mask_type` and `--scheduler_type` as below:
| Weight Space        | `--training_space` |
| ------------------- | ------------------ |
| Binary-space training| `binary`           |
| Real-space training   | `real`             |

| Hypermask            | `--mask_type`  |
| --------------------- | -------------  |
| EMP mask  | `EMP`  |
| MMP mask | `MMP` |
| Random mask        | `RAND`       |

| LR Scheduler  | `--scheduler_type`  |
| --------------------- | -------------  |
| Cosine Decay          | `cosine`  |
| Constant              | `const`        |

### Architecture
You can change the architecture by setting the argumnts `--hid_dim` and `--depth`.

### Dataset
You can use all datasets used in the experiments (Digit, MNIST, CIFAR-10 and Tiny-ImageNet) by setting the argument `--dataset` as bellow:
| Dataset       | `--dataset`  |
| ------------- | ------------ |
| Digit  Dataset        | `Digit`        |
| MNIST         | `MNIST`        |
| CIFAR-10      | `CIFAR10`      |
| CIFAR-100     | `CIFAR100`     |
| Tiny-ImageNet     | `TinyIN`     |