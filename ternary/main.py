import torch
import torch.nn as nn
import torch.optim as optim
from binary_layer import BinaryLinear
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ternary_layer import TernaryLinear

# 모델 정의
model = nn.Sequential(
    TernaryLinear(784, 256),
    nn.ReLU(),
    TernaryLinear(256, 10)
)

# Loss 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 예제 데이터
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
num_epochs = 10
# 학습 예시
for epoch in range(num_epochs) :
    for inputs, targets in train_loader:
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Loss: {loss.item()}')
