import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms
import torchvision.datasets as dsets

#parameters
learning_rate = 1e-3
epoch = 20
batch_size = 100

mnist_train = dsets.MNIST(root = "MNIST_data/", train = True, transform = transforms.ToTensor(), download = True)
mnist_test = dsets.MNIST(root = "MNIST_data/", train = False
                         , transform = transforms.ToTensor(), download = True)

data_loader = torch.utils.data.DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True, drop_last = True)


#모델구성
func = nn.Sequential(
    nn.Linear(784, 784),
    nn.ReLU(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
).cuda()

#optimizer 정의
optimizer = torch.optim.Adam(func.parameters(), lr = learning_rate)

#loss 함수 정의
LossFunc = nn.CrossEntropyLoss()

for i in range(epoch + 1):
    total = 0
    correct = 0
    for x, y in data_loader:
        x = x.view(-1, 28*28).cuda()
        y = y.cuda()

        #prediction 계산
        pred = func(x)
        
        #loss 계산
        loss = LossFunc(pred, y)

        #model update
        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        
        total += batch_size
        correct += (pred.argmax(dim = 1) == y).sum().item()

    print('Epoch %d. accuracy: %0.5f' %(i, correct / total * 100))
