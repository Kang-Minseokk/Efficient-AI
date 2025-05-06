import os
import numpy as np

from sklearn.datasets import load_iris
from sklearn import datasets

import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label


def get_MNIST():
    channel_size, image_size = 1, 28
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    train_x, train_y = torch.empty([60000, channel_size, image_size, image_size]), torch.empty([60000], dtype=torch.long)
    for i, t in enumerate(list(mnist_train)):
        train_x[i], train_y[i] = t[0], t[1]

    mnist_test = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, channel_size, image_size, image_size]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(mnist_test)):
        test_x[i], test_y[i] = t[0], t[1]

    return (train_x, train_y), (test_x, test_y)


def get_FashionMNIST():
    channel_size, image_size = 1, 28
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    fashion_train = tv.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([60000, channel_size, image_size, image_size]), torch.empty([60000], dtype=torch.long)
    for i, t in enumerate(list(fashion_train)):
        train_x[i], train_y[i] = t[0], t[1]

    fashion_test = tv.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 784]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(fashion_test)):
        test_x[i], test_y[i] = t[0], t[1]

    return (train_x, train_y), (test_x, test_y)


def get_CIFAR10():
    channel_size, image_size = 3, 32
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    cifar_train = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([50000, channel_size, image_size, image_size]), torch.empty([50000], dtype=torch.long)
    for i, t in enumerate(list(cifar_train)):
        train_x[i], train_y[i] = t[0], t[1]

    cifar_test = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, channel_size, image_size, image_size]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(cifar_test)):
        test_x[i], test_y[i] = t[0], t[1]

    return (train_x, train_y), (test_x, test_y)


def get_CIFAR100():
    channel_size, image_size = 3, 32
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    cifar_train = tv.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([50000, channel_size, image_size, image_size]), torch.empty([50000], dtype=torch.long)
    for i, t in enumerate(list(cifar_train)):
        train_x[i], train_y[i] = t[0], t[1]

    cifar_test = tv.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, channel_size, image_size, image_size]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(cifar_test)):
        test_x[i], test_y[i] = t[0], t[1]

    return (train_x, train_y), (test_x, test_y)

def get_Digit():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    digits = datasets.load_digits()
    digits_data, digits_target = digits.data, digits.target
    
    data_x, data_y = torch.empty([1797, 64]), torch.empty([1797], dtype=torch.long)
    for i in range(len(digits_data)):
        data_x[i], data_y[i] = torch.from_numpy(digits_data[i].reshape((-1)).astype(np.float32)).clone(), digits_target[i].item()
    
    _, indices = data_y.sort()
    test = torch.zeros([10,180])
    test[:,int(180*0.8):] = 1
    test = (test.reshape(-1)[:1797]==1)
    train_indices = indices[~test].sort()[0]
    test_indices = indices[test].sort()[0]
    train_x, train_y = data_x[train_indices], data_y[train_indices]
    test_x, test_y = data_x[test_indices], data_y[test_indices]

    return (train_x, train_y), (test_x, test_y)


def make_dataset(dataset_name, test):
    # get dataset
    if dataset_name == "Digit":
        train_dataset, test_dataset = get_Digit()
    elif dataset_name == "MNIST":
        train_dataset, test_dataset = get_MNIST()
    elif dataset_name == "FashionMNIST":
        train_dataset, test_dataset = get_FashionMNIST()
    elif dataset_name == "CIFAR10":
        train_dataset, test_dataset = get_CIFAR10()
    elif dataset_name == "CIFAR100":
        train_dataset, test_dataset = get_CIFAR100()
    
    if test:
        train_set = Dataset(train_dataset[0], train_dataset[1])
        test_set = Dataset(test_dataset[0], test_dataset[1])
    else:
        size = len(train_dataset[0])
        train_set = Dataset(train_dataset[0][:int(size*0.9)], train_dataset[1][:int(size*0.9)])
        test_set = Dataset(train_dataset[0][int(size*0.9):], train_dataset[1][int(size*0.9):])
    return train_set, test_set


def get_dataset(dataset_name, test=True):
    if dataset_name=="TinyIN":
        path = "/gs/hs0/tga-RLA/22M30965/tiny-imagenet-200"
        train_dir = os.path.join(path, 'train')
        val_dir = os.path.join(path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_set = tv.datasets.ImageFolder(train_dir, transforms.Compose([transforms.ToTensor(),normalize,]))
        test_set = tv.datasets.ImageFolder(val_dir,transforms.Compose([transforms.ToTensor(),normalize,]))
    else:
        train_set, test_set = make_dataset(dataset_name, test)
    
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    
    return train_set, test_set, loss_function
