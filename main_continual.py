import math
import pickle
import torch
import torch.cuda
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as dsets
import os
from utils.BBBConvmodel import BBBAlexNet, BBBLeNet, BBBCIFAR
from utils.BBBlayers import GaussianVariationalInference
import datagen
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score
import uncertainty_metrics.numpy as um
import numpy as np

cuda = torch.cuda.is_available()

'''
HYPERPARAMETERS
'''
is_training = True  # set to "False" for evaluation of network ability to remember previous tasks
pretrained = False  # change pretrained to "True" for continual learning
task='CIFAR10'
test='oc'

num_samples = 5  # because of Casper's trick
batch_size = 16
beta_type = "Blundell"
net = BBBCIFAR   # LeNet, BBB3Conv3FC, or AlexNet
dataset = 'CIFAR-10'  # MNIST, CIFAR-10, or CIFAR-100
num_epochs = 200
p_logvar_init = 0
q_logvar_init = -10
lr = 0.001
weight_decay = 0.0005


# dimensions of input and output
if dataset is 'MNIST':    # train with MNIST
    outputs = 10
    inputs = 1
elif dataset is 'CIFAR-10':    # train with CIFAR-10
    outputs = 10
    inputs = 3
elif dataset is 'CIFAR-100':    # train with CIFAR-100
    outputs = 100
    inputs = 3
elif dataset is 'CIFAR-100-classes':    # train with 3 CIFAR-100-classes classes
    outputs = 3
    inputs = 3


if net is BBBLeNet or BBBCIFAR:
    resize = 32
elif net is BBBAlexNet:
    resize = 227

'''
LOADING DATASET
'''

if dataset is 'MNIST':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x + noise * torch.randn(x.size())),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = dsets.MNIST(root="data", download=True, transform=transform)
    val_dataset = dsets.MNIST(root="data", download=True, train=False, transform=transform)

elif dataset is 'CIFAR-10':
    if test == 'clean':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        train_dataset = dsets.CIFAR10(root="data", download=True, transform=transform)
        val_dataset = dsets.CIFAR10(root="data", download=True, train=False, transform=transform)
    else:
        loader_train, loader_test, loader_val = datagen.load_cifar10()


'''
MAKING DATASET ITERABLE
'''
if task == 'clean':
    print('length of training dataset:', len(train_dataset))
    n_iterations = num_epochs * (len(train_dataset) / batch_size)
    n_iterations = int(n_iterations)
    print('Number of iterations: ', n_iterations)

    loader_train = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    loader_val = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# enable loading of weights to transfer learning
def cnnmodel():
    model = net(outputs=outputs, inputs=inputs, task=test)
    return model

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
    # Soft Voting (entropy and var in confidence)
    preds_soft = outputs.mean(0)  # [data, 10]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]

    preds_hard = outputs.var(0).cpu()  # [data, 10]
    variance = preds_hard.max(-1)[0].numpy()  # [data]
    return (entropy, variance)


'''
INSTANTIATE MODEL
'''

model = cnnmodel()

if cuda:
    model.cuda()

'''
INSTANTIATE VARIATIONAL INFERENCE AND OPTIMISER
'''
vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

'''
check parameter matrix shapes
'''

# how many parameter matrices do we have?
print('Number of parameter matrices: ', len(list(model.parameters())))

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

'''
TRAIN MODEL
'''

if is_training:
    logfile = os.path.join('diagnostics_{}.txt'.format(task))
else:
    logfile = os.path.join('diagnostics_{}_eval.txt'.format(task))

with open(logfile, 'w') as lf:
    lf.write('')


def run_epoch(loader, epoch, is_training=False):
    m = math.ceil(len(loader.dataset) / loader.batch_size)

    accuracies = []
    likelihoods = []
    kls = []
    losses = []
    targets = []
    outs = []
    for i, (images, labels) in enumerate(loader):
        # Repeat samples (Casper's trick)
        targets.append(labels.view(-1))
        x = images.view(-1, inputs, resize, resize).repeat(num_samples, 1, 1, 1)
        y = labels.repeat(num_samples)

        if cuda:
            x = x.cuda()
            y = y.cuda()

        if beta_type is "Blundell":
            beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
        elif beta_type is "Soenderby":
            beta = min(epoch / (num_epochs//4), 1)
        elif beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0
        
        logits, kl = model.probforward(x)
        if test == 'clean':
            loss = vi(logits, y, kl, beta)
            ll = -loss.data.mean() + beta*kl.data.mean()
            if is_training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        else:
            loss = torch.zeros(1)
            ll = torch.zeros(1)


        _, predicted = logits.max(1)
        outputs = torch.nn.functional.softmax(logits, -1).view(num_samples, len(images), -1)
        outs.append(outputs)
        accuracy = (predicted.data.cpu() == y.cpu()).float().mean()

        accuracies.append(accuracy)
        losses.append(loss.data.mean())
        kls.append(beta*kl.data.mean())
        likelihoods.append(ll)

    diagnostics = {'loss': sum(losses)/len(losses),
                   'acc': sum(accuracies)/len(accuracies),
                   'kl': sum(kls)/len(kls),
                   'likelihood': sum(likelihoods)/len(likelihoods)}

    return diagnostics, torch.cat(targets, -1).view(-1), torch.cat(outs,  1)


for epoch in range(num_epochs):
    if is_training is True:
        diagnostics_train, targets, out_in = run_epoch(loader_train, epoch, is_training=True)
        with torch.no_grad():
            diagnostics_val, _, _ = run_epoch(loader_val, epoch)
            diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
            diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
            print(diagnostics_train)
            print(diagnostics_val)
        if test == 'oc':
            with torch.no_grad():
                diagnostics_val, _, out_out = run_epoch(loader_test, epoch)
                un_in, un_in_var = uncertainty(out_in.detach())
                un_out, un_out_var = uncertainty(out_out.detach())
                auc = auc_score(un_in, un_out)
                print ('auc: ', auc)
                ece = ece_score(targets, out_in.mean(0))
                print ('ece:', ece) 

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_train))
            lf.write(str(diagnostics_val))
    else:
        diagnostics_val = run_epoch(loader_val, epoch)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_val))

'''
SAVE PARAMETERS
'''

if is_training:
    weightsfile = os.path.join("weights_{}.pkl".format(task))
    with open(weightsfile, "wb") as wf:
        pickle.dump(model.state_dict(), wf)

