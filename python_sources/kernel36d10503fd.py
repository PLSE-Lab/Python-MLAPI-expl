import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.main = nn.Sequential(
			nn.ConvTranspose2d(100,512,4,1,0,bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.ConvTranspose2d(512,256,4,2,1,bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256,128,4,2,1,bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128,64,4,2,1,bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64,3,4,2,1,bias=False),
			nn.Tanh()
			)
	def forward(self,input):
		output = self.main(input)
		return output

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(3,64,4,2,1,bias=False),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(64,128,4,2,1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(128,256,4,2,1,bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(256,512,4,2,1,bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(512,1,4,1,0,bias=True),
			nn.Sigmoid()
			)
	def forward(self,input):
		output = self.main(input)
		return output


def load_dataset(transformer,batch_size):
	dataset = datasets.CIFAR10(root='./CIFAR-10 Python',download=True,transform = transformer)
	dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
	return dataloader

def train(G,D,dataloader,criterion,learning_rate,betas,epochs):
	optimizerD = optim.Adam(D.parameters(),lr=learning_rate,betas = betas)
	optimizerG = optim.Adam(G.parameters(),lr=learning_rate,betas = betas)
	for i in range(epochs):
		for j, (img,target) in enumerate(dataloader,0):
			D.zero_grad()
			input = Variable(img)
			target = Variable(torch.ones(input.size()[0]))
			output = D(input)
			err_real = criterion(output,target)

			noise = Variable(torch.randn(input.size()[0],100,1,1))
			fake = G(noise)
			target = Variable(torch.zeros(input.size()[0]))
			output = D(fake.detach())
			err_fake = criterion(output,target)
			errD = err_fake+err_real
			errD.backward()
			optimizerD.step()

			G.zero_grad()
			target = Variable(torch.ones(input.size()[0]))
			output = D(fake)
			errG = criterion(output,target)
			errG.backward()
			optimizerG.step()
		print('Train Epoch : {} Discriminator Loss : {:.6f} Generator Loss : {:.6f}'.format(i,errD,errG))

if __name__=="__main__":
	batch_size = 10
	image_size = 64
	transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])
	dataloader = load_dataset(transform,batch_size)
	G = Generator()
	D = Discriminator()
	criterion = nn.BCELoss()
	learning_rate = 0.0002
	betas  = (0.5,0.999)
	epochs = 25
	train(G,D,dataloader,criterion,learning_rate,betas,epochs)
