
import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

print("Imported all libraries successfully!")


parser = argparse.ArgumentParser(description='PyTorch MNIST Example for CAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=19, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 5, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('data', train=True, download=True,
		transform=transforms.ToTensor()),
	batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


lam = 1e-4


class CAE(nn.Module):
	def __init__(self):
		super(CAE, self).__init__()

		self.fc1 = nn.Linear(784, 400, bias = False) # Encoder
		self.fc2 = nn.Linear(400, 784, bias = False) # Decoder

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def encoder(self, x):
		h1 = self.relu(self.fc1(x.view(-1, 784)))
		return h1

	def decoder(self,z):
		h2 = self.sigmoid(self.fc2(z))
		return h2

	def forward(self, x):
		h1 = self.encoder(x)
		h2 = self.decoder(h1)
		return h1, h2

		# Writing data in a grid to check the quality and progress
	def samples_write(self, x, epoch):
		_, samples = self.forward(x)
		#pdb.set_trace()
		samples = samples.data.cpu().numpy()[:16]
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)
		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
		if not os.path.exists('out/'):
			os.makedirs('out/')
		plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
		#self.c += 1
		plt.close(fig)


mse_loss = nn.BCELoss(size_average = False)

def loss_function(W, x, recons_x, h):
	#mse = (x-recons_x).pow(2).sum()
  mse = mse_loss(recons_x, x)
  """
  W is shape of N_hidden x N. So, we do not need to transpose it as opposed to http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
  """
  dh = h*(1-h) # N_batch x N_hidden
  contractive_loss = torch.mm(dh**2,torch.sum(Variable(W)**2, dim=1)).sum().mul_(lam)
  return mse + contractive_loss



model = CAE()



optimizer = optim.Adam(model.parameters(), lr = 0.0001)

if args.cuda:
	model.cuda()

def train(epoch):
	model.train()

	train_loss = 0

	for idx, (data, _) in enumerate(train_loader):
		data = Variable(data)
		if args.cuda:
			data = data.cuda()
		
		optimizer.zero_grad()

		hidden_representation, recons_x = model(data)

		# Get the weights	
		# model.state_dict().keys()
		W = model.state_dict()['fc1.weight'] # change the key by seeing the keys manually. (In future I will try to make it automatic)

		loss = loss_function(W, data.view(-1, 784), recons_x, hidden_representation)


		# pdb.set_trace()

		loss.backward()

		train_loss += loss.data[0]

		optimizer.step()

		if idx % args.log_interval == 0:
			print('Train epoch: {} [{}/{}({:.0f}%)]\t Loss: {:.6f}'.format(
				epoch, idx*len(data), len(train_loader.dataset), 
				100*idx/len(train_loader),
				loss.data[0]/len(data)))
			

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))
	model.samples_write(data,epoch)

for epoch in range(args.epochs):
	train(epoch)



