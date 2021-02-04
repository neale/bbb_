# # Bayes by Backprop
# An implementation of the algorithm described in https://arxiv.org/abs/1505.05424.  
# This notebook accompanies the article at https://www.nitarshan.com/bayes-by-backprop.

import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from scipy.stats import entropy

def cov(data, rowvar=False):
    """Estimate a covariance matrix given data.

    Args:
        data (tensor): A 1-D or 2-D tensor containing multiple observations 
            of multiple dimentions. Each row of ``mat`` represents a
            dimension of the observation, and each column a single
            observation.
        rowvar (bool): If True, then each row represents a dimension, with
            observations in the columns. Othewise, each column represents
            a dimension while the rows contains observations.

    Returns:
        The covariance matrix
    """
    x = data.detach().clone()
    if x.dim() > 2:
        raise ValueError('data has more than 2 dimensions')
    if x.dim() < 2:
        x = x.view(1, -1)
    if not rowvar and x.size(0) != 1:
        x = x.t()
    fact = 1.0 / (x.size(1) - 1)
    x -= torch.mean(x, dim=1, keepdim=True)
    return fact * x.matmul(x.t()).squeeze()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
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
        epsilon = self.normal.sample(self.rho.size())
        epsilon = epsilon.cuda(0)
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


#PI = 0.5
SIGMA_1 = torch.cuda.FloatTensor([0])
SIGMA_2 = torch.cuda.FloatTensor([0])

PI = 0.1
#SIGMA_1 = torch.cuda.FloatTensor([math.exp(1)])
#SIGMA_2 = torch.cuda.FloatTensor([math.exp(1)])
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.zeros(out_features, in_features))
            #torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(
            torch.zeros(out_features, in_features))
            #torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Prior distributions
        #self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.weight_prior = Gaussian(0, SIGMA_1)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
        else:
            weight = self.weight.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight)
            self.log_variational_posterior = self.weight.log_prob(weight)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return torch.matmul(input.cuda(0), weight.t().cuda(0))

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(3, 1)
    
    def forward(self, x, sample=False):
        x = self.l1(x, sample)
        return x
    
    def log_prior(self):
        return self.l1.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=10):
        outputs = torch.zeros(samples, train_batch_size, 1).cuda(0)
        log_priors = torch.zeros(samples).cuda(0)
        log_variational_posteriors = torch.zeros(samples).cuda(0)
        for i in range(samples):
            out = self(input, sample=True)
            outputs[i] = out
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.mse_loss(outputs.mean(0), target, reduction='sum')
        loss = (log_variational_posterior - log_prior) + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

net = BayesianNetwork().cuda(0)

input_size = 3
train_batch_size = 10
output_dim = 1
batch_size = 100
inputs = torch.randn(batch_size, input_size)
beta = torch.rand(input_size, output_dim) + 5.
noise = torch.randn(batch_size, output_dim)
targets = inputs @ beta + noise
true_cov = torch.inverse(
        inputs.t() @ inputs)  # + torch.eye(input_size))
true_mean = (true_cov @ inputs.t() @ targets).cuda(0)

def train(net, optimizer, epoch):
    net.train()
    net.zero_grad()
    perm = torch.randperm(batch_size)
    idx = perm[:train_batch_size]
    train_inputs = inputs[idx].cuda(0)
    train_targets = targets[idx].cuda(0)

    loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(train_inputs, train_targets)
    loss.backward()
    optimizer.step()

def test_ensemble(i):
    weight_samples = []
    preds = []
    for _ in range(10):
        sample_w1 = net.l1.weight.sample().view(-1)
        weight_samples.append(sample_w1)
        preds.append(net(inputs.cuda(0), sample=True))
    preds = torch.stack(preds)
    weight_samples = torch.stack(weight_samples)
    computed_mean = weight_samples.mean(0).cuda(0)
    computed_cov = cov(weight_samples)
    computed_preds = inputs.cuda(0) @ computed_mean.cuda(0)
    preds = net(inputs.cuda(0), sample=True)
    pred_err = torch.norm((preds - targets.cuda(0)).mean(0))
    mean_err = torch.norm(computed_mean - true_mean.squeeze())
    mean_err = mean_err / torch.norm(true_mean)

    cov_err = torch.norm(computed_cov - true_cov)
    cov_err = cov_err / torch.norm(true_cov)

    print("train_iter {}: pred err {}".format(i, pred_err))
    print("train_iter {}: mean err {}".format(i, mean_err))
    print("train_iter {}: cov err {}".format(i, cov_err))
    print("computed_cov norm: {}".format(computed_cov.norm()))

optimizer = optim.Adam(net.parameters(), lr=1e-4)
for epoch in range(50000):
    train(net, optimizer, epoch)
    if epoch % 1000 == 0:
        test_ensemble(epoch)
test_ensemble(epoch)
