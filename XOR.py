import torch
import torch.nn as nn
import torch.optim as optim

epoch = 5000 + 1

data_x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
data_y = torch.tensor([[0.], [1.], [1.], [0.]])

func = nn.Sequential(
		nn.Linear(2, 2),
		nn.Sigmoid(),
		nn.Linear(2, 1),
		nn.Sigmoid()
		)

optimizer = optim.Adam(func.parameters(), lr = 1e-3)

for i in range (epoch):
	hypothesis = func(data_x)
	criteria = nn.MSELoss()
	loss = criteria(hypothesis, data_y)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if i % 100 == 0:
		parameter_list = list(func.parameters())
		print('Epoch %d : Loss %.5f' %(i, loss))
		#print('w : %.5f, b : %.5f\n' %(parameter_list[0], parameter_list[1]))
