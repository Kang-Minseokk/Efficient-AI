import torch
import torch.nn as nn
import torch.optim as optim
from ternary_layer import TernaryLinear
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 데이터셋 불러오기
train_dataset = datasets.MNIST(root='data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
model = nn.Sequential(    
    nn.Flatten(),
    TernaryLinear(784, 256),    
    nn.ReLU(),
    TernaryLinear(256, 10)
).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    correct = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        # print("Logits range:", output.min().item(), output.max().item())
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(y).sum().item()

    acc = correct / len(train_loader.dataset)    
    print(f"Epoch {epoch}: Train Loss = {total_loss:.4f}, Accuracy = {acc * 100:.2f}%")

model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += pred.eq(y).sum().item()
test_acc = correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

