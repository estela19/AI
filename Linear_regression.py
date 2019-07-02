import torch
import torch.nn as nn
import torch.optim as optim

epoch = 1000 + 1

#data 정의
data_x = torch.tensor([[0.], [1.], [2.], [3.], [4.], [5.]] )
data_y = torch.tensor([[1.], [3.], [5.], [7.], [9.], [11.]] )

#모델정의
func = nn.Linear(1, 1)

#optimizer 정의
optimizer = optim.Adam(func.parameters(), lr = 1e-2)

for i in range(epoch) :
	#hypothesis 계산
	hypothesis = func(data_x)

	#Loss 계산
	loss_func = nn.L1Loss()
	loss = loss_func(hypothesis, data_y)

	#Loss로 func 개선(gradient descent)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if i % 100 == 0:
		parameter_list = list(func.parameters())
		print('epoch %d : Loss %.5f' % (i, loss.item()))
		print('w : %.5f, b : %.5f\n' %(parameter_list[0], parameter_list[1]))
   
