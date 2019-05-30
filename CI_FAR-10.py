import torch
from torch import nn,optim
import torch.nn.functional as F

#cifar-10 을 불러오기 위해 필요한 코드
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

train_dataset = CIFAR10(root='./data', train=True, download=True, 
    transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]))

BATCHSIZE=128
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
#Made 신경망

#우리 신경망은 nn.Module을 상속받음
class Network(nn.Module):
  def __init__(self):
    super().__init__()
    
    #시각적인 자료 학습 이땐 CNN구조가 좋음
    #Made convolutional layer
    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
    #3개의 채널 rgb로 받아서 64개의 특징을 추출.커널 크기는 3)
    #padding은 [커널/2]로 씀 보통.
    
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
    #특징을 받아 더 세부적인 특징을 뽑은
    
    self.conv3 = nn.Conv2d(64, 64, 3, padding = 1)
    self.conv4 = nn.Conv2d(64, 64, 3, padding = 1)
    
    self.conv5 = nn.Conv2d(64, 1, 1, padding = 0)
    #단순히 차원이 줄이는 것이 목적
    
    #conv는 특징을 추출하는 부분
    
    #추출한 특징을 분류하는 부분
    self.fc = nn.Linear(1 * 32 ** 2, 10)
    #1개의 통합된 32*32 픽셀의 그림 특징을 10개의 종류로 분류
    
    #연산 방법 정의하는 부분
  def forward(self, x):
    x = F.relu(self.conv1(x))
    #선형을 비선형의 conv로 바꿈.
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
      
    x = x.view(-1, 32**2)
    # -1은 배치사이즈
    # 2차원conv를 1차원으로 연산해야 하므로 2차원을 1차원으로 바꾼다.
    x = self.fc(x)
      
    return F.log_softmax(x, dim = 1)
    #정규화 (x의 dim=0 dms -1, dim=1 은 32**2 dim=1을 기준으로 정규화하라)
    
    net = Network().cuda()
    
#신경망 학습
EPOCHS = 20

opt = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4)
#weight_decay 가중치 감쇠
ACE = nn.CrossEntropyLoss().cuda()
#정답에서 얼마만큼 떨어졌는지 확인하는 오차함수

for epoch in range(1, EPOCHS+1):
  #각 epoch마다 오차의 합을 초기화 
  running_loss = 0
  for inputs, labels in train_loader:
    inputs, labels = inputs.cuda(), labels.cuda()
    
    #학습하기전 데이터 초기화
    opt.zero_grad()
    #기울기 를 0으로 초기화
    preds = net(inputs)
    #신경망에 input 담자
    
    loss = ACE(preds, labels)
    loss.backward()
    #기울기를 계산
    running_loss += loss.item()
    
    #학습
    opt.step()
    
  #출력
  print('[Epoch %d] Loss : %.4f' %(epoch, running_loss/len(train_loader)))
