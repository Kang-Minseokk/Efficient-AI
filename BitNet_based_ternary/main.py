from models.bit1 import BitNetMLP  
from models.tricky_ternary import TerTrickMLP
from models.bit1_58 import TernaryMLP
from models.fp_32 import FP32MLP
from get_args import get_args

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pynvml
from models.mlp_mixer import MLPMixer # MLP Mixer


if __name__ == "__main__" :
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    pynvml.nvmlInit() # 전력 측정을 하기 위한 초기화
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"Found {device_count} GPU(s).") # GPU 개수 확인
    
    power_list = []
    
    args = get_args()        
    if args.model == 'binary' :
        print("Binary Mode")
        model = BitNetMLP()                    
    
    elif args.model == 'trick_ternary':
        print("Trick Ternary Mode")
        model = TerTrickMLP()
        
    elif args.model == 'qat_ternary':
        print("QAT Ternary Mode")
        model = TernaryMLP()                             
    
    elif args.model == 'fp32':
        print("FP32 Mode")
        model = FP32MLP()
    
    elif args.model == 'mlp_mixer':
        print("MLP Mixer Mode")
        model = MLPMixer(
            image_size = 32, # 이미지 한 변의 길ㅣ
            channels = 3,
            patch_size = 4,
            dim = 512,
            depth = 12,
            num_classes = 10
        )
        
    # This is for MNIST Dataset!    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    
    # This is for CIFAR-10 Dataset!
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2010]
        )
    ])

    # Datasets & loaders
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)
    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=8e-4) # BitNet 논문에서는 2e-4, 4e-4, 8e-4가 존재한다.
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=5e-5) # MLP Mixer 관련 글에서 AdamW를 사용하였다.
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min = 1e-6, last_epoch = -1, verbose = False)
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 30
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:>2}/{epochs} - Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        acc = correct / len(test_loader.dataset)
        print(f"          Test Acc:  {acc*100:>.2f}%")
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)      
    
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_watts = power_mw / 1000.0
        power_list.append(power_watts)
    
    max_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Max GPU Memory Usage: {max_mem :.2f} MB")                    
    
    avg_power = sum(power_list) / len(power_list)
    print(f"Average Power Draw: {avg_power:.2f} Watts")