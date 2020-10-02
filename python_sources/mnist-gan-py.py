import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pylab as plt
import numpy as numpy

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.model = nn.Sequential(
			nn.Linear(784,1024),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024,512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512,256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256,1),
			nn.Sigmoid()
			)
	def forward(self,x):
		out = self.model(x.view(x.size(0),784))
		out = out.view(out.size(0),-1)
		return out

class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.model = nn.Sequential(
				nn.Linear(100,256),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Linear(256,512),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Linear(512,1024),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Linear(1024,2048),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Linear(2048,784),
				nn.Tanh()
			)
	def forward(self,x):
		x = x.view(x.size(0),100)
		out = self.model(x)
		return out

def train_discriminator(d,real_images,real_labels,fake_images,fake_labels,criterion,optim):
	d.zero_grad()
	outputs = d(real_images)
	real_loss = criterion(outputs,real_labels)

	outputs = d(fake_images)
	fake_loss = criterion(outputs,fake_labels)

	d_loss = fake_loss + real_loss
	if d_loss>=0.001:
		d_loss.backward()
		optim.step()
	return d_loss,optim

def train_generator(g,d_outputs,real_labels,criterion,optim,flag=False):
	g.zero_grad()
	g_loss = criterion(d_outputs,real_labels)
	if flag==True:
		g_loss.backward(retain_graph=True)
	else:
		g_loss.backward(retain_graph=False)
	optim.step()
	return g_loss,optim

def train(g,d,criterion,g_optim,d_optim,epochs,batch_size,learning_rate,dataloader):
	for i in range(epochs):
		l1 = 0
		l2 = 0
		for idx, (images,_) in enumerate(dataloader):
			images = Variable(images)
			real_labels = Variable(torch.ones(images.size(0)))

			noise = Variable(torch.randn(images.size(0),100))
			fake_images = g(noise)
			fake_labels = Variable(torch.zeros(images.size(0)))

			d_loss,d_optim = train_discriminator(d,images,real_labels,fake_images,fake_labels,criterion,d_optim)

			l1 = d_loss + l1

			noise = Variable(torch.randn(images.size(0),100))
			fake_images = g(noise)
			outputs = d(fake_images)

			g_loss,g_optim = train_generator(g,outputs,real_labels,criterion,g_optim,True)

			#g_loss,g_optim = train_generator(g,outputs,real_labels,criterion,g_optim,False)

			l2 = l2 + g_loss
		l1 = l1/float(idx)
		l2 = l2/float(idx)
		print('Epoch : %d, D_loss: %.4f, G_loss: %.4f'%(i,l1,l2))
		state = {
			'epoch': i,
			'g_state_dict': g.state_dict(),
			'g_optim': g_optim.state_dict(),
			'd_state_dict': d.state_dict(),
			'd_optim': d_optim.state_dict()
		}
		torch.save(state,'state.ckpt')
		#fig_size = plt.rcParams["figure.figsize"]
		#plt.rcParams["figure.figsize"][0] = 2
		#plt.rcParams["figure.figsize"][1] = 2
		#torch.save(state,'state.ckpt')
		#num_test_images = 4
		#test_noise = Variable(torch.randn(num_test_images,100))
		#test_images = g(test_noise)
		#for k in range(num_test_images):
		#	plt.imshow(test_images[k,:].detach().numpy().reshape(28,28),cmap = 'Greys')
		#	plt.show()



def load_datasets(transformer):
	train_dataset = datasets.MNIST(root='./data/',train=True,download=True,transform=transformer)
	return train_dataset

def train_util():
	batch_size = 50
	dataset = load_datasets(transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean = (0.5,0.5,0.5),std=(0.5,0.5,0.5))
		]
		))
	train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
	discriminator = Discriminator()
	generator = Generator()
	criterion = nn.BCELoss()
	learning_rate = 0.001
	d_optim = optim.Adam(discriminator.parameters(),lr = learning_rate*0.001,weight_decay = 0.0001)
	g_optim = optim.Adam(generator.parameters(),lr = learning_rate,weight_decay = 0.0001)
	epochs = 100
	train(generator,discriminator,criterion,g_optim,d_optim,epochs,batch_size,learning_rate,train_loader)

def resume(resume_i,g,d,criterion,g_optim,d_optim,epochs,batch_size,learning_rate,dataloader):
	for i in range(resume_i+1,epochs):
		l1 = 0
		l2 = 0
		for idx, (images,_) in enumerate(dataloader):
			images = Variable(images)
			real_labels = Variable(torch.ones(images.size(0)))

			noise = Variable(torch.randn(images.size(0),100))
			fake_images = g(noise)
			fake_labels = Variable(torch.zeros(images.size(0)))

			d_loss,d_optim = train_discriminator(d,images,real_labels,fake_images,fake_labels,criterion,d_optim)

			l1 = d_loss + l1

			noise = Variable(torch.randn(images.size(0),100))
			fake_images = g(noise)
			outputs = d(fake_images)

			g_loss,g_optim = train_generator(g,outputs,real_labels,criterion,g_optim,True)

			#g_loss,g_optim = train_generator(g,outputs,real_labels,criterion,g_optim,False)

			l2 = l2 + g_loss
		l1 = l1/float(idx)
		l2 = l2/float(idx)
		print('Epoch : %d, D_loss: %.4f, G_loss: %.4f'%(i,l1,l2))
		state = {
			'epoch': i,
			'g_state_dict': g.state_dict(),
			'g_optim': g_optim.state_dict(),
			'd_state_dict': d.state_dict(),
			'd_optim': d_optim.state_dict()
		}
		#fig_size = plt.rcParams["figure.figsize"]
		#plt.rcParams["figure.figsize"][0] = 2
		#plt.rcParams["figure.figsize"][1] = 2
		torch.save(state,'state.ckpt')
		#num_test_images = 4
		#test_noise = Variable(torch.randn(num_test_images,100))
		#test_images = g(test_noise)
		#for k in range(num_test_images):
		#	plt.imshow(test_images[k,:].detach().numpy().reshape(28,28),cmap = 'Greys')
		#	plt.show()





def resume_util():
	batch_size = 50
	dataset = load_datasets(transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
		]))
	train_loader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle=True)
	state = torch.load('state.ckpt')
	discriminator = Discriminator()
	generator = Generator()
	criterion = nn.BCELoss()
	learning_rate = 0.001
	d_optim = optim.Adam(discriminator.parameters(),lr = learning_rate*0.001,weight_decay = 0.0001)
	g_optim = optim.Adam(generator.parameters(),lr = learning_rate,weight_decay = 0.0001)
	i = state['epoch']
	discriminator.load_state_dict(state['d_state_dict'])
	generator.load_state_dict(state['g_state_dict'])
	g_optim.load_state_dict(state['g_optim'])
	d_optim.load_state_dict(state['d_optim'])
	epochs = 100
	resume(i,generator,discriminator,criterion,g_optim,d_optim,epochs,batch_size,learning_rate,train_loader)

def generate_images():
	state = torch.load('.state.ckpt')
	generator = Generator()
	generator.load_state_dict(state['g_state_dict'])
	noise = Variable(torch.randn(1,100))
	fake_images = generator(noise)
	plt.imshow(fake_images.detach().numpy().reshape(28,28),cmap='Greys')
	plt.show()

	pass


if __name__=="__main__":
	#generate_images()
	train_util()


