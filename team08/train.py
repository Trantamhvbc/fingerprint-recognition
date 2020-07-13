
from model import *
from data import *
import torch 
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn.functional as F
#Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#paramaters

epochs = 100
num_class = 10
batch_size = 16
lr = 0.01
'''
create dataset
'''
train_loader = DataLoader( dataset = image_Dataset('/home/dell/Desktop/f/data/data_create_pretrainmodel.csv') , batch_size = batch_size, shuffle = True)
validation_loader = DataLoader( dataset = image_Dataset('/home/dell/Desktop/f/data/test_search.csv') , batch_size = batch_size, shuffle = False)
criterion = torch.nn.CrossEntropyLoss()
#model 
model = CNNnet()
#optipmin
otipmizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
# model 

F_measure = {}
def train(epoch):
	loss_item_sum = 0
	count_loss_item = 0
	model.train()	
	for batch_idx, (datas,targets) in enumerate(train_loader):
		otipmizer.zero_grad()
		datas = datas.to(device)
		targets = targets.to(device)
		outs = model(datas)
		loss = criterion(outs,targets)
		loss_item_sum += loss.item()
		count_loss_item += 1
		loss.backward()
		otipmizer.step()
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(datas), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss_item_sum/count_loss_item))
		loss_item_sum = 0
		count_loss_item = 0 

			
def test(datas_loader,index):
	model.eval()
	loss = 0
	correct = 0
	count_batch_idx = 0
	for batch_idx, (datas,targets) in enumerate(datas_loader):
		datas = datas.to(device)
		targets = targets.to(device)
		outs = model(datas)
		loss += criterion(outs,targets)
		count_batch_idx += 1
		outs = outs.data.max(1,keepdim  = True)[1]
		correct += outs.eq(targets.data.view_as(outs)).cpu().sum()
	print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(index,
        loss/count_batch_idx, correct, len(datas_loader.dataset),
        100. * correct / len(datas_loader.dataset)))

def main():
	for epoch in range(epochs):
		train(epoch)
		test(datas_loader = validation_loader,index = 'validationset')
		test(datas_loader = train_loader,index = 'trainset')

if __name__ == '__main__':
	main()


