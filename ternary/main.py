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

# 아무리 생각해도 인공지능은 공부량이 중요한 것 같다. 
# 아는게 많을수록 무조건 유리함. 모르면 GPT도 의미가 없어 보인다.. 
# 따라서, 공부 많이 하고 논문 많이 읽자. 어떤 방법이 있는지 많이 알아두고 경험을 많이 해보자. 
# 그것만이 실력 향상을 위한 길인 것 같아.
# 아참, 깃허브에 정리도 잘 해두자. 
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
