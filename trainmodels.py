

from __future__ import print_function
import argparse
import torch
from torchvision import datasets,transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import gc
import model as AllModels
import os
from importlib import import_module
torch.manual_seed(1)

if not os.path.exists('models):
    os.makedirs('models')

def train(model, train_loader,optimizer,device,epoch):
	model.train()
	for batch_indx, (data,target) in enumerate(train_loader):
		data,target=data.to(device),target.to(device)
		target_onehot=torch.nn.functional.one_hot(target,num_classes=10)
		optimizer.zero_grad()
		output=model(data)
		
		loss = torch.sum(- target_onehot * F.log_softmax(output, -1), -1).mean()
		#loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		
	print('Train Epoch: {}  \tLoss: {:.6f}'.format(epoch,loss.item()))

def test(model, test_loader,device):
	model.eval()
	test_loss=0
	accuracy=0
	with torch.no_grad():
		for batch_indx, (data,target) in enumerate(test_loader):
			data,target=data.to(device),target.to(device)
			target_onehot=torch.nn.functional.one_hot(target,num_classes=10)
			output=model(data)
			#softmax_cross_entropy_with_logits 
			output=F.log_softmax(output, -1)
			test_loss+= torch.sum(- target_onehot *output, -1).mean()
			#test_loss+= F.nll_loss(output, target)
			pred=output.argmax(dim=1,keepdim=True)
			accuracy+=pred.eq(target.view_as(pred)).sum().item()
	accuracy/=len(test_loader.dataset)
	test_loss/=len(test_loader.dataset)
	print('Test accuracy {:.4f} and  \tLoss: {:.6f}'.format(accuracy,test_loss.item()))
	return test_loss, accuracy

		


def main():
	parser=argparse.ArgumentParser(description='PyTorch Learning MNIST')
	parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
	parser.add_argument('--epochs', type=int, default=30, help='inumber of epoches for learning')
	parser.add_argument('--gamma', type=float, default=0.9, help='scheduler gamma')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--dataset', type=str, default="mnist", help='dataset name')
	parser.add_argument('--save_metric', type=str, default="accuracy", help='save_metrics')
	parser.add_argument('--model_type', type=str, default="vgg19", help='CIFAR model type')
	args=parser.parse_args()

	
	kwargs={'batch_size':args.batch_size}
	transform=transforms.Compose([transforms.ToTensor()])
	if torch.cuda.is_available():
		device=torch.device("cuda")
		kwargs.update({'num_workers':1, 'pin_memory':1,'shuffle':True})
		print('Using CUDA, GPU is available')
	else:
		device=torch.device("cpu")
		      
	file_name='models/'+str(args.dataset)+'.pt'
	if args.dataset=='mnist':
		model=AllModels.mnist_model().to(device)
		train_data=datasets.MNIST('./data',train=True, download=True,transform=transform)
		test_data=datasets.MNIST('./data',train=False, download=True,transform=transform)
	elif args.dataset=='cifar10':
		m= ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
			'vgg19_bn', 'vgg19', 'ResNet', 'resnet20', 'resnet32', 'resnet44',
			 'resnet56', 'resnet110', 'resnet1202']
		if args.model_type in m:
			
			func = getattr(AllModels, args.model_type.lower())
			model=func(10).to(device)
		      	file_name=file_name[:-3]+'_'+args.model_type.lower()+'.pt'
		      
		else:
			print('Unknown Model Type. Select from '+ model_type.join())
			quit()
		train_data=datasets.CIFAR10('./data',train=True, download=True,transform=transform)
		test_data=datasets.CIFAR10('./data',train=False, download=True,transform=transform)
	else:
		print("Dataset Not Found")
		quit()

	

	train_loader= torch.utils.data.DataLoader(train_data,**kwargs)
	test_loader= torch.utils.data.DataLoader(test_data,**kwargs)
	optimizer=optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, nesterov=True)
	scheduler= StepLR(optimizer, step_size=1, gamma=args.gamma)

	test_loss,accuracy=10000000,0
	for i in range(args.epochs):
		train(model, train_loader,optimizer,device,i+1)
		l, acc=test(model, test_loader,device)
		scheduler.step()
		
		if l<test_loss and args.save_metric=='loss':
			
			tyorch.save(model.state_dict(),file_name)
			print('Improved loss from {:.6f} to {:.6f} and saved model'.format(test_loss,l))
			test_loss=l
		elif acc > accuracy and args.save_metric=='accuracy':
			torch.save(model.state_dict(),file_name)
			print('Improved acuracy from {:.6f} to {:.6f} and saved model'.format(accuracy,acc))
			accuracy=acc

		elif args.save_metric!='accuracy' and args.save_metric!='loss':
			Torch.save(model.state_dict(),file_name)
			print('saved model')
			

if __name__ == '__main__':
	main()

