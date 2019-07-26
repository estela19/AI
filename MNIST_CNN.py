import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init
import torchvision.transforms as transforms
import torchvision.datasets as dsets

#parameters
learning_rate = 1e-3
epoch = 50
batch_size = 100

mnist_train = dsets.MNIST(root = "MNIST_data/", train = True, transform = transforms.ToTensor(), download = True)
mnist_test = dsets.MNIST(root = "MNIST_data/", train = False, transform = transforms.ToTensor(), download = True)

data_loader = torch.utils.data.DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True, drop_last = True)


#모델구성
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    
    self.layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2)   
    )
    
    self.fc = nn.Linear(7*7*64, 10, bias = True)
    torch.nn.init.xavier_uniform(self.fc.weight)
    
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out
  
#모델 선언
func = CNN().cuda()

#optimizer 정의
optimizer = torch.optim.Adam(func.parameters(), lr = learning_rate)

#loss 함수 정의
LossFunc = nn.CrossEntropyLoss()

for i in range(epoch + 1):
    total = 0
    correct = 0
    for x, y in data_loader:
        x = x.cuda()
        y = y.cuda()

       
        #prediction 계산
        pred = func(x)
        
        #loss 계산
        loss = LossFunc(pred, y)

        #model update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += batch_size
        correct += (pred.argmax(dim = 1) == y).sum().item()

    if i % 10 == 0:
      print('Epoch %d. accuracy: %0.5f' %(i, correct / total * 100))
