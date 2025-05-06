import torch
import os
from models.reactnet_backbone import birealnet18
from models.MLP import get_mlp
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset
from utils import *
from datasets import get_dataset
from get_args import get_args
from models.VGG import *
from models.MLP import *
from models.base import *
from models.mask_generator import get_mask_generator

# 1. device 설정해두기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Binary MLP 준비
mlp = get_mlp(
    in_shape=(32, 3),     # backbone 출력 shape
    out_dim=100,           # 클래스 수
    hid_dim=512,           # 은닉층 차원 (논문: 512)
    depth=4,               # 2-layer MLP
    b_weight=True,         # binary weight 사용
    b_act=True,            # binary activation 사용
    training_space="binary",
    batch_norm=True
).to(device)

# 3. 데이터셋 로딩 
train_set, test_set, _ = get_dataset("CIFAR100", test=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32768, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32768) 

# 4. 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()

# 5. 학습
num_epochs = 10
mask_generator = {}

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)                
        flat_x = x.view(x.size(0), -1) # 이미지 평탄화 작업
        
        mlp.train()                   # MLP를 학습 모드로 설정해주기
        output = mlp(flat_x)          # mlp에 백본의 출력값을 넣어주기                
        # print(mlp.activations)      # activation을 저장했는지 확인을 위한 코드입니다.        
        loss = criterion(output, y)   # 크로스 엔트로피로 손실함수 구해주기
        # loss.backward() 제거 예정
        # ✅ 이진 그래디언트 업데이트
        with torch.no_grad():
            activation_idx = 0
            for name, param in mlp.named_parameters():
                if 'weight' not in name:
                    continue  # bias 제외

                # 현재 weight와 대응하는 activation 가져오기
                activation = mlp.activations[activation_idx]
                activation_idx += 1

                # 이진 그래디언트 계산
                binary_grad = activation.T @ torch.sign(output - one_hot(y, output.size(1)))  # simple approx
                binary_grad = binary_grad.to(param.device)

                # 업데이트 (이진 형태 유지 or 실수 업데이트 중 택 1)
                param.data = param.data - lr * binary_grad
        
        # with torch.no_grad():
        #     for name, param in mlp.named_parameters():                
        #         if param.grad is None:
        #             continue

        #         # 1. Gradient Scaling (XNOR-Net style)
        #         scaling_factor = torch.mean(torch.abs(param.data)) # 하나의 Layer의 가중치 절댓값 평균을 구합니다.
        #         scaled_grad = param.grad.data * scaling_factor

        #         # 2. Mask generator 생성 (처음 한 번만)
        #         if name not in mask_generator:
        #             mask_generator[name] = get_mask_generator(
        #                 mask_type="EMP", scheduler_type="cosine",
        #                 temp_init=1e-3, temp_min=1e-5, T_max=num_epochs * len(train_loader)
        #             )

        #         # 3. 마스크 생성 및 업데이트
        #         mask = mask_generator[name].get_mask(scaled_grad, sign_function(param.data))
        #         param_target = sign_function(-scaled_grad)
        #         param.data[mask] = param_target[mask]  # 마스크를 적용하여 param_target의 업데이트 여부를 정합니다.
        # # 이 코드는 마스크가 학습되는지를 확인하기 위해서 작성하였습니다.        
        # total_loss += loss.item()
        # _, predicted = output.max(1) # 텐서에서 가장 큰 값을 가져옵니다. (이를 통해서 예측을 수행하는 것입니다.)
        # correct += predicted.eq(y).sum().item()
        # total += y.size(0)

    acc = 100. * correct / total
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
