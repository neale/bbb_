# # Bayes by Backprop
# An implementation of the algorithm described in https://arxiv.org/abs/1505.05424.  
# This notebook accompanies the article at https://www.nitarshan.com/bayes-by-backprop.

import os
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from scipy.stats import entropy

def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices


def generate_regression_data(n_train, n_test):
    x_train1 = torch.linspace(-6, -2, n_train//2).view(-1, 1)
    x_train2 = torch.linspace(2, 6, n_train//2).view(-1, 1)
    x_train3 = torch.linspace(-2, 2, 4).view(-1, 1)
    x_train = torch.cat((x_train1, x_train2, x_train3), dim=0)
    y_train = -(1 + x_train) * torch.sin(1.2*x_train) 
    y_train = y_train + torch.ones_like(y_train).normal_(0, 0.04)

    x_test = torch.linspace(-6, 6, n_test).view(-1, 1)
    y_test = -(1 + x_test) * torch.sin(1.2*x_test) 
    y_test = y_test + torch.ones_like(y_test).normal_(0, 0.04)
    return (x_train, y_train), (x_test, y_test)

def plot_bnn_regression(model, data):
    sns.set_style('darkgrid')
    gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
    gt_y = -(1+gt_x) * torch.sin(1.2*gt_x) 
    #gt_y += torch.ones_like(gt_x).normal_(0, 0.04).cpu()
    x_train, y_train, x_test, y_test = train_data.cuda(), train_target.cuda(), test_data.cuda(), test_target.cuda()
    outputs = []
    for _ in range(100):
        outputs.append(model(x_test, sample=True).detach().cpu())
    outputs = torch.stack(outputs)
    print (outputs.shape)
    mean = outputs.mean(0).squeeze()
    std = outputs.std(0).squeeze()
    print (mean.norm(), std.norm())
    print (mean.shape, std.shape)
    x_test = x_test.cpu().numpy()
    x_train = x_train.cpu().numpy()

    plt.fill_between(x_test.squeeze(), mean.T+2*std.T, mean.T-2*std.T, alpha=0.5)
    plt.plot(gt_x, gt_y, color='red', label='ground truth')
    plt.plot(x_test, mean.T, label='posterior mean', alpha=0.9)
    plt.scatter(x_train, y_train.cpu().numpy(),color='r', marker='+',
        label='train pts', alpha=1.0, s=50)
    plt.legend(fontsize=14, loc='best')
    plt.ylim([-6, 8])
    plt.savefig('plots/bbb_regression.png')
    plt.close('all') 

(train_data, train_target), (test_data, test_target) = generate_regression_data(80, 200)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())

BATCH_SIZE = 84
TEST_BATCH_SIZE = 200

TRAIN_SIZE = 84 #len(train_loader.dataset)
TEST_SIZE = 200#len(test_loader.dataset)
NUM_BATCHES = 1#len(train_loader)
NUM_TEST_BATCHES = 1#len(test_loader)

CLASSES = 1
TRAIN_EPOCHS = 10000
SAMPLES = 100
TEST_SAMPLES = 100

#assert (TRAIN_SIZE % BATCH_SIZE) == 0
#assert (TEST_SIZE % TEST_BATCH_SIZE) == 0


# ## Modelling
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()


PI = 0.5
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(1, 50)
        self.l2 = BayesianLinear(50, 1)
    
    def forward(self, x, sample=False):
        #x = x.view(-1, 28)
        x = F.relu(self.l1(x, sample))
        x = self.l2(x, sample)
        return x
    
    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior+ self.l2.log_variational_posterior+ self.l2.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            out = self(input, sample=True)
            outputs[i] = out
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.mse_loss(outputs.mean(0), target, reduction='sum')
        loss = (log_variational_posterior - log_prior)/NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

net = BayesianNetwork().to(DEVICE)

def train(net, optimizer, epoch):
    net.train()
    data = train_data
    target = train_target
    data, target = data.to(DEVICE), target.to(DEVICE)
    net.zero_grad()
    loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
    loss.backward()
    optimizer.step()

def test_ensemble():
    net.eval()
    data = test_data.cuda()
    target = test_target.cuda()
    with torch.no_grad():
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES+1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
        for i in range(TEST_SAMPLES):
            outputs[i] = net(data, sample=True)
        outputs[TEST_SAMPLES] = net(data, sample=False)
        output = outputs.mean(0)
        mse = (output - target).pow(2).mean()
    print('posterior_mean MSE: {}'.format(mse))


optimizer = optim.Adam(net.parameters())
for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer, epoch)
test_ensemble()
plot_bnn_regression(net, data=None)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


fmnist_sample = iter(test_loader).next()
fmnist_sample[0] = fmnist_sample[0].to(DEVICE)
print(fmnist_sample[1])
sns.set_style("dark")
show(make_grid(fmnist_sample[0].cpu()))


# #### Out-of-Domain Uncertainty

mnist_loader = val_loader

mnist_sample = iter(mnist_loader).next()
mnist_sample[0] = mnist_sample[0].to(DEVICE)
sns.set_style("dark")
show(make_grid(mnist_sample[0].cpu()))


net.eval()
mnist_outputs = net(mnist_sample[0], True).max(1, keepdim=True)[1].detach().cpu().numpy()
for _ in range(99):
    mnist_outputs = np.append(mnist_outputs, net(mnist_sample[0], True).max(1, keepdim=True)[1].detach().cpu().numpy(), axis=1)

sns.set_style("darkgrid")
plt.subplots(5,1,figsize=(10,4))
for i in range(5):
    plt.subplot(5,1,i+1)
    plt.ylim(0,100)
    plt.xlabel("Categories")
    plt.xticks(range(10), ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
    plt.ylabel("Count")
    plt.yticks(range(50,101,50))
    plt.hist(mnist_outputs[i], np.arange(-0.5, 10, 1))

