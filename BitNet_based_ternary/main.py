from models.bit1 import BitNetMLP  
from models.tricky_ternary import TerTrickMLP
from models.bit1_58 import TernaryMLP
from models.fp_16 import FP16MLP
from get_args import get_args

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


if __name__ == "__main__" :
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    args = get_args()        
    if args.model == 'binary' :
        print("Binary Mode")
        model = BitNetMLP(
                in_features=32*32*3, # Dataset에 맞게 변환 필요
                hidden_features=1024,
                num_classes=10,
                depth=4,
                dropout=0.1                
            )                    
    
    elif args.model == 'trick_ternary':
        print("Trick Ternary Mode")
        model = TerTrickMLP(
            in_features = 32*32*32, 
            hidden_features = args.hidden_features,
            dropout = args.dropout
        )
        
    elif args.model == 'qat_ternary':
        print("QAT Ternary Mode")
        model = TernaryMLP(
            in_features=32*32*3, # Dataset에 맞게 변환 필요
            hidden_features=1024,
            num_classes=10,
            depth=4,
            dropout=0.1            
        )
    
    elif args.model == 'fp16':
        print("FP16 Mode")
        model = FP16MLP(
            in_features = 32*32*3,
            hidden_features = 256,
            num_classes = 10,
            depth = 4,
            dropout = 0.1
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
            mean = [0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # Datasets & loaders
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)
    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=2e-4) # BitNet 논문에서는 2e-4, 4e-4, 8e-4가 존재한다.
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