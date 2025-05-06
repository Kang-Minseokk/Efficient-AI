import torch
import os
from models.reactnet_backbone import birealnet18
from models.MLP import get_mlp
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset
from utils import *
from get_args import get_args
from models.VGG import *
from models.MLP import *
from models.base import *
from models.mask_generator import get_mask_generator
import torch.nn.functional as F


# from torchvision import datasets, transforms
# def test_activation_saving(model, device, train_loader):
#     model.eval()  # 평가 모드
#     dummy_input = torch.randn(2, 28, 28)  # 작은 배치, MNIST 사이즈
#     dummy_input = dummy_input.view(2, -1)  # Flatten: (2, 784)

#     with torch.no_grad():
#         output = model(dummy_input)

#     print(f"🔍 Activation 저장 개수: {len(model.activations)}")
#     for i, act in enumerate(model.activations):
#         print(f"[{i}] Activation shape: {act.shape}")


# 메인 코드
def main():        
    # MNIST 데이터셋 사용
    train_set, test_set, loss_function = get_dataset("MNIST", test=True)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)
    
    # 2. 모델 생성
    # 이미지 크기와 채널 수에 맞게 설정 (CIFAR-10은 32x32x3이지만 64x64로 변환)
    model = get_mlp(
        in_shape=(28, 1),     # 64x64 크기, 3채널 이미지
        hid_dim=512,          # 은닉층 뉴런 수
        out_dim=10,           # CIFAR-10은 10개 클래스
        depth=2,              # 1개의 은닉층
        batch_norm=True,      # 배치 정규화 사용
        b_weight=True,        # 이진 가중치 사용 안 함
        b_act=True           # 이진 활성화 함수 사용 안 함
    )
    print(f"👀 model 클래스: {model.__class__}")
    print(model)
    # 3. 손실 함수와 최적화 알고리즘 정의
    criterion = nn.CrossEntropyLoss()
    lr = 0.001 # 학습률 직접 사용
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 학습 루프    
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        # test_activation_saving(model, device, train_loader)
        # 학습 모드 설정
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)            

            # === 이진 업데이트는 그 다음에 ===
            probs = torch.softmax(outputs, dim=1)            
            one_hot = F.one_hot(labels, num_classes=10).float()
            error_sign = torch.sign(probs - one_hot)  # 오차 방향 계산            
            activation = model.activations[0]  # -1 또는 1     
          
            with torch.no_grad():                
                weight_layer = model.classifier[-1]
                weight = weight_layer.weight.data   # shape: (C, H), ex: (10, 512)
                
                # 1. ΔL 계산
                approx_grad = activation.T @ error_sign   # shape: (H, C)
                deltaL = approx_grad.T * (1 - 2 * weight) # shape: (C, H)
                
                # 2. ΔL ≤ 0 인 것만 후보로 사용
                mask = (deltaL <= 0)
                if mask.any():
                    deltaL_candidates = deltaL[mask]

                    # 3. ΔL 정규화 (LayerNorm 스타일)
                    mu = deltaL_candidates.mean()
                    sigma = deltaL_candidates.std(unbiased=False) + 1e-8
                    hat_deltaL = (deltaL_candidates - mu) / sigma

                    # 4. z 계산 (학습 가능한 gamma, beta 대신 상수)
                    gamma = 1.0
                    beta = -(0.0 - mu) / sigma  # ΔL_target = 0 기준
                    z = gamma * hat_deltaL + beta

                    # 5. Laplace 기반 확률 계산
                    b = 0.3  # Laplace scale (낮게 조정)
                    p_candidates = torch.exp(-torch.abs(z) / b)

                    # 6. 전체 확률 배열에 반영
                    p_i = torch.zeros_like(deltaL)
                    p_i[mask] = p_candidates

                    # 7. 기대 손실 감소량 계산 및 스케일 조정
                    expected = (p_i * (-deltaL)).sum()
                    target_loss_reduction = 0.3  # 실험값
                    if expected > target_loss_reduction:
                        scale = target_loss_reduction / expected
                        p_i = torch.clamp(p_i * scale, max=1.0)
                else:
                    p_i = torch.zeros_like(deltaL)

                # 8. flip 결정
                flip_mask = torch.bernoulli(p_i)

                # 9. flip 적용 및 이진화
                weight *= (1 - 2 * flip_mask)
                weight.copy_(weight.sign())

            # === 이진 그래디언트 업데이트 끝 ===

            running_loss += loss.item()

            # 정확도 계산
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # === 검증 단계 ===
        # print("🔍 error_sign unique:", torch.unique(error_sign))               
        # print("🔍 activation unique:", torch.unique(activation))
        # print("🔍 weight unique:", torch.unique(weight_layer.weight))
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch {epoch+1} Result - Loss: {test_loss / len(test_loader):.3f}, Accuracy: {100 * correct / total:.2f}%')
        # print(model.classifier[-1].weight)
        # print(torch.norm(binary_grad))
        # print(error_sign)        

    print('Done!')       

if __name__ == '__main__':
    main()

    
