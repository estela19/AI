import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms
import numpy as np

sample = "if you want you"

#make dictionary
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}

#hyper parameters
dic_size = len(char_dic)
input_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

#data setting
sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [sample_idx[1:]]

#transform as torch tensor
x = torch.FloatTensor(x_one_hot)
y = torch.LongTensor(y_data)

#모델선언
rnn = nn.RNN(input_size, hidden_size, batch_first = True)

#Loss func 와 optimizer 정의
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)

#start training
for i in range(100):
  outputs, _status = rnn(x)
  loss = loss_func(outputs.view(-1, input_size), y.view(-1))
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if(i % 10 == 0):
    result = outputs.data.numpy().argmax(axis = 2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)
