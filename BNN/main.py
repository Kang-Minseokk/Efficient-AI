from utils import *
from datasets import get_dataset
from get_args import get_args
from models.VGG import *
from models.MLP import *
from models.base import *
from models.mask_generator import get_mask_generator

import os
import gc
import sys
import time
import wandb
import torch
import pickle
import numpy as np
from torch import nn

import pdb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DATA_INFO = {
    "Digit": {"input": (8, 1), "output": 10},
    "MNIST": {"input": (28, 1), "output": 10},
    "CIFAR10": {"input": (32, 3), "output": 10},
    "CIFAR100": {"input": (32, 3), "output": 100},
    "TinyIN": {"input": (64, 3), "output": 200},
}


def main(**kwargs):
    fix_seed(kwargs["seed"])
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = set_device()
    print(f"DEVICE: {device}")

    if kwargs["log"]:
        wandb.init(config=kwargs)

    # set dataloader
    train_set, test_set, loss_function = get_dataset(kwargs["dataset"], kwargs["test"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=kwargs["batch_size"],
                                               shuffle=True, num_workers=8, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=kwargs["batch_size"],
                                               shuffle=False, num_workers=8, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = valid_loader

    if kwargs["model"] == "VGG":
        model = get_vgg(mode=kwargs["vgg_mode"], b_act=kwargs["b_act"], b_weight=kwargs["b_weight"],
                        in_shape=DATA_INFO[kwargs["dataset"]]["input"], out_dim=DATA_INFO[kwargs["dataset"]]["output"]).to(device)
    elif kwargs["model"] == "MLP":
        model = get_mlp(bias=kwargs["bias"], batch_norm=kwargs["batch_norm"], batch_norm_out=kwargs["batch_norm_out"], affine=kwargs["affine"],
                        b_act=kwargs["b_act"], b_weight=kwargs["b_weight"], training_space=kwargs["training_space"],
                        in_shape=DATA_INFO[kwargs["dataset"]]["input"], hid_dim=kwargs["hid_dim"], out_dim=DATA_INFO[kwargs["dataset"]]["output"], depth=kwargs["depth"]).to(device)
    train(model, train_loader, valid_loader, kwargs, device=device)


def train(model, train_loader, valid_loader, kwargs, device):
    print(f"Epoch: 0")
    loss_train, acc = test(model, train_loader, 1, device)
    loss_test, acc_topk = test(model, valid_loader, [1, 5], device)

    results = {}
    results["train loss"] = loss_train
    results["train accuracy"] = acc[0]
    results["valid loss"] = loss_test
    results["valid accuracy (top-1)"] = acc_topk[0]
    results["valid accuracy (top-5)"] = acc_topk[1]
    send_log(results, kwargs["log"])

    loss_function = nn.CrossEntropyLoss(reduction="mean")
    mask_generator = {}
    std_dict = {}

    for e in range(kwargs["epochs"]):
        print(f"Epoch: {e+1}")
        # train
        start_time = time.time()
        for i, (x, y) in enumerate(train_loader):
            torch.cuda.empty_cache()
            gc.collect()

            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    m, n = name.split(".")[:2]
                    real_flag = False
                    for module_type in [nn.BatchNorm2d, nn.BatchNorm1d]:
                        real_flag = (real_flag or isinstance(model._modules[m][int(n)], module_type))

                    if kwargs["training_space"] == "real" or real_flag:
                        param.data -= kwargs["lr"] * param.grad.data
                    elif kwargs["training_space"] == "binary":
                        if name not in mask_generator:
                            temp_init = kwargs["lr"] / \
                                (0.01 * 2**0.5) if kwargs["scheduler_type"] == "gaussian" else kwargs["lr"]
                            mask_generator[name] = get_mask_generator(mask_type=kwargs["mask_type"], scheduler_type=kwargs["scheduler_type"],
                                                                      temp_init=temp_init, temp_min=1e-5, T_max=kwargs["epochs"] * len(train_loader))
                        mask = mask_generator[name].get_mask(param.grad.data, sign_function(param.data))
                        param_target = sign_function(-param.grad.data)
                        param.data[mask] = param_target[mask]
        end_time = time.time()

        # predict
        loss_train, acc = test(model, train_loader, 1, device)
        loss_test, acc_topk = test(model, valid_loader, [1, 5], device)

        results = {}
        results["train loss"] = loss_train
        results["train accuracy"] = acc[0]
        results["valid loss"] = loss_test
        results["valid accuracy (top-1)"] = acc_topk[0]
        results["valid accuracy (top-5)"] = acc_topk[1]
        results["proccess time"] = end_time - start_time
        send_log(results, kwargs["log"])


def test(model, test_loader, topk, device):
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    pred, label = None, None
    for i, (x, y) in enumerate(test_loader):
        torch.cuda.empty_cache()
        gc.collect()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
            pred = y_pred if pred is None else torch.concat([pred, y_pred])
            label = y if label is None else torch.concat([label, y])

    if not isinstance(topk, list):
        topk = [topk]
    acc_list = [calc_accuracy(pred, label, k=k) for k in topk]

    return loss_function(pred, label), acc_list


if __name__ == '__main__':
    FLAGS = vars(get_args())
    main(**FLAGS)
