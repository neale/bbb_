#!/usr/bin/env python
# coding: utf-8

# # Bayes by Backprop
# An implementation of the algorithm described in https://arxiv.org/abs/1505.05424.  
# This notebook accompanies the article at https://www.nitarshan.com/bayes-by-backprop.

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
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score
import uncertainty_metrics.numpy as um


sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices



def load_mnist(split=True):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    path = 'data_m/'
    if split:
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7, 8, 9]

    else:
        train_idx = list(range(10))
        test_idx = list(range(10))
    
    trainset = datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, train_idx))
    train_loader = torch.utils.data.DataLoader(
            train_hidden,
            batch_size=100,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True)

    valset = datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    val_hidden = torch.utils.data.Subset(valset, get_classes(valset, train_idx))
    val_loader = torch.utils.data.DataLoader(
            val_hidden,
            batch_size=100,
            shuffle=True,
            **kwargs)

    testset = datasets.MNIST(path,
            train=False,
            transform=transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, test_idx))
    test_loader = torch.utils.data.DataLoader(
            test_hidden,
            batch_size=100,
            shuffle=False,
            **kwargs)

    return train_loader, test_loader, val_loader

train_loader, test_loader, val_loader = load_mnist(True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())


BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(val_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(val_loader)

CLASSES = 6
TRAIN_EPOCHS = 10
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
        self.l1 = BayesianLinear(28*28, 400)
        self.l2 = BayesianLinear(400, 400)
        self.l3 = BayesianLinear(400, 6)
    
    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
        return x
    
    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior+ self.l2.log_variational_posterior+ self.l2.log_variational_posterior
    
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
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()

def test_ensemble(loader):
    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES, dtype=int)
    outputs_out = torch.zeros(TEST_SAMPLES, len(loader.dataset), 6)
    outs = []
    targets = []
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            targets.append(target.view(-1))
            for i in range(TEST_SAMPLES):
                outputs[i] = F.softmax(net(data, sample=True), -1)
            outputs_out[:, i*100:(i+1)*100, :] = outputs
            outs.append(outputs)
            #outputs[TEST_SAMPLES] = net(data, sample=False)
            output = outputs.mean(0)
            preds = preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1] # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
        for index, num in enumerate(corrects):
            if index < TEST_SAMPLES:
                print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
            else:
                print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
        print('Ensemble Accuracy: {}: [{}/{}['.format(correct/TEST_SIZE, correct, TEST_SIZE))
    
    return torch.cat(outs, dim=1), torch.cat(targets, -1).view(-1)

def auc_score(known, unknown):
    """ Computes the AUROC for the given predictions on `known` data
        and `unknown` data.
    """
    y_true = np.array([0] * len(known) + [1] * len(unknown))
    y_score = np.concatenate([known, unknown])
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score

def ece_score(labels, probs, bins=15):
    labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    return um.ece(labels, probs, num_bins=bins)

def uncertainty(outputs): 
    """ outputs (torch.tensor): class probabilities, 
        in practice these are given by a softmax operation
        * Soft voting averages the probabilties across the ensemble
            dimension, and then takes the maximal predicted class
            Taking the entropy of the averaged probabilities does not 
            yield a valid probability distribution, but in practice its ok
    """
    # Soft Voting (entropy and var in confidence)
    preds_soft = outputs.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]
    
    preds_hard = outputs.var(0).cpu()  # [data, 10]
    variance = preds_hard.max(-1)[0].numpy()  # [data]
    return (entropy, variance)

optimizer = optim.Adam(net.parameters())
for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer, epoch)
    outputs_in, labels = test_ensemble(val_loader)
    outputs_out , _ = test_ensemble(test_loader)
    un_in, un_in_var = uncertainty(outputs_in)
    un_out, un_out_var = uncertainty(outputs_out)
    auc_var = auc_score(un_in_var, un_out_var)
    print ('auc var: ', auc_var)
    auc = auc_score(un_in, un_out)
    print ('auc: ', auc)
    ece = ece_score(labels, outputs_in.mean(0))
    print ('ece: ', ece)
