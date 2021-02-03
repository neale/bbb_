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

sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

def generate_class_data(n_samples=100,
    means=[(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]):
    #means=[(2., 2.), (-2., -2.)]):

    data = torch.zeros(n_samples, 2)
    labels = torch.zeros(n_samples)
    size = n_samples//len(means)
    for i, (x, y) in enumerate(means):
        dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
        samples = dist.sample([size])
        data[size*i:size*(i+1)] = samples
        labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
    
    return data, labels.long()


def plot_classification(model, epoch):
    basedir = 'plots'
    os.makedirs(basedir, exist_ok=True)
    x = torch.linspace(-12, 12, 100)
    y = torch.linspace(-12, 12, 100)
    gridx, gridy = torch.meshgrid(x, y)
    grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1).cuda()
    outputs = []
    for _ in range(100):  
        outputs.append(F.softmax(model(grid, sample=True), -1).cpu())
    outputs = torch.stack(outputs).transpose(0, 1)
    outputs = outputs.detach()  # [B, D]
    torch.save(outputs,'plots/outputs_epoch_{}.pt'.format(epoch))
    mean_outputs = outputs.mean(1).cpu()  # [B, D]
    std_outputs = outputs.std(1).cpu()
    conf_outputs = entropy(mean_outputs.T.numpy())
    conf_mean = mean_outputs.mean(-1)
    conf_std = std_outputs.max(-1)[0] * 1.94
    labels = mean_outputs.argmax(-1)
    data, _ = generate_class_data(n_samples=400) 
    
    p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std,
        cmap='rainbow')
    p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.3)
    cbar = plt.colorbar(p1)
    cbar.set_label("confidance (std)")
    plt.savefig('plots/std_{}.png'.format(epoch))
    print ('saved')
    plt.close('all')
    
train_data, train_target = generate_class_data(100)
test_data, test_target = generate_class_data(100)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())

BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

TRAIN_SIZE = 100 #len(train_loader.dataset)
TEST_SIZE = 100#len(test_loader.dataset)
NUM_BATCHES = 1#len(train_loader)
NUM_TEST_BATCHES = 1#len(test_loader)

CLASSES = 4
TRAIN_EPOCHS = 20000
SAMPLES = 10
TEST_SAMPLES = 10

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
        self.l1 = BayesianLinear(2, 10)
        self.l2 = BayesianLinear(10, 10)
        self.l3 = BayesianLinear(10, 4)
    
    def forward(self, x, sample=False):
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
        return x
    
    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior + self.l2.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            out = F.log_softmax(self(input, sample=True), -1)
            outputs[i] = out
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction='sum')
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
    correct = 0
    corrects = np.zeros(TEST_SAMPLES+1, dtype=int)
    data = test_data
    target = test_target
    outs = []
    with torch.no_grad():
        data, target = data.to(DEVICE), target.to(DEVICE)
        outputs = torch.zeros(TEST_SAMPLES+1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
        for i in range(TEST_SAMPLES):
            outputs[i] = F.softmax(net(data, sample=True), -1)
        outputs[TEST_SAMPLES] = F.softmax(net(data, sample=False), -1)
        outs.append(outputs)
        output = outputs.mean(0)
        preds = preds = outputs.max(2, keepdim=True)[1]
        pred = output.max(1, keepdim=True)[1] # index of max log-probability
        corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
    return torch.cat(outs, 1)


optimizer = optim.Adam(net.parameters())
for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer, epoch)
    if epoch % 2000 == 0:
        outputs = test_ensemble()
        plot_classification(net, epoch)
