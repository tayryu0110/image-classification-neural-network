import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
# from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import os
import h5py
import math
import matplotlib.pyplot as plt
import seaborn as sns
# from cnn_utils import *
# hello
save_to_file_dir = './tay_log/saved_logs'

h5f = h5py.File('saved_gray.h5', 'r')

X_train = h5f['X_train'][:1000]
y_train = h5f['y_train'][:1000]
X_valid = h5f['X_valid'][:100]
y_valid = h5f['y_valid'][:100]
X_test = h5f['X_test'][:100]
y_test = h5f['y_test'][:100]

h5f.close()

X_train = np.swapaxes(X_train, 1, 3)
X_valid = np.swapaxes(X_valid, 1, 3)
X_test = np.swapaxes(X_test, 1, 3)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

def show_img(img):
    fig,ax = plt.subplots()
    if img.shape == (32,32,3):
        ax.imshow(img)
    else:
        ax.imshow(img[:,:,0])

class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=1),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=1),
            torch.nn.ReLU())
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1))
        
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1))
        
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1))
        
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1))

        self.fc1 = torch.nn.Linear(4 * 4 * 4, 16, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer7 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU())
        self.fc2 = torch.nn.Linear(16, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def get_batch2(X,Y,M,dtype):
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    batch_ys = torch.FloatTensor(Y[batch_indices]).type(dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

model = cnn()
criterion = nn.MSELoss()

learning_rate = 0.0005
num_epoches = 50
minibatch_size = 32
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(len(list(model.parameters())))

print(list(model.parameters())[0].size())
print(list(model.parameters())[1].size())
print(list(model.parameters())[2].size())
print(list(model.parameters())[3].size())
print(list(model.parameters())[4].size())
print(list(model.parameters())[5].size())
print(list(model.parameters())[6].size())
print(list(model.parameters())[7].size())
print(list(model.parameters())[8].size())
print(list(model.parameters())[9].size())
print(list(model.parameters())[10].size())
print(list(model.parameters())[11].size())
print(list(model.parameters())[12].size())
print(list(model.parameters())[13].size())
print(list(model.parameters())[14].size())
print(list(model.parameters())[15].size())

iter = 0
seed = 5
dtype = torch.FloatTensor
for epoch in range(num_epoches):
    print ('Training .........\n')
    print('Epoch', epoch+1, ': ........ \n')
    
    m = X_train.shape[0]
    num_minibatches = int(m / minibatch_size)
    
    for i in range(num_minibatches):
        epoch_x, epoch_y = get_batch2(X_train, y_train, minibatch_size, dtype)
        epoch_x = epoch_x.requires_grad_()
        optimizer.zero_grad()
        outputs = model(epoch_x)
        y_ = np.argmax(epoch_y, axis=1).reshape(minibatch_size, 1).float()
        loss = criterion(outputs, y_)
        loss.backward()
        optimizer.step()

        iter += 1

    correct = 0
    total = 0
    for i in range(10):
        valid_x, valid_y = get_batch2(X_valid, y_valid, 10, dtype)
        valid_x = valid_x.requires_grad_()
        outputs = model(valid_x)
        _, predicted = torch.max(outputs.data, 1)
        valid_y = np.argmax(valid_y, axis=1)
        correct += (predicted == valid_y).sum().float()
    total = float(len(valid_y))
    accuracy = 100 * correct / total

    print('Iteration: {}. Loss: {}'.format(iter, loss.item()))






