import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
from functools import partial
import math
import pickle
from scipy.optimize import minimize
from numpy.linalg import norm
from numpy.linalg import det
from numpy.linalg import inv
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


### Data Simulation ###
def row(n, coor, i):
# outputs the i-th row of the squared distance matrix; coor are the x,y coordinates
    sigmai = [0] * n
    #print('row %d' % i)
    for j in range(0, n):
        # print('column %d' % j)
        sigmai[j] = sum(pow(coor[i] - coor[j], 2))
    return sigmai

# the squared distance matrix
def Dist(layer):
    n = layer.shape[0]  # dimension
    coor = np.array(layer[:,[0,1]])
    my_pool = Pool()
    func = partial(row,n,coor)
    sigma = my_pool.imap(func,range(n))
    my_pool.close()
    my_pool.join()
    return list(sigma)

# generate data
def Simu(grid, seed):
    np.random.seed(seed)
    n = grid * grid
    inter = 2/grid  # length of each grid
    X = np.zeros((n, 10))  # cordinate x[-1,1]
            
    for i in range(grid):   # which row
        for j in range(grid):  # column
            X[grid*i+j , 0] = inter * random.random() - 1 + inter * j
            X[grid*i+j , 1] = inter * random.random() - 1 + inter * i
    X[:,2:10] = np.random.normal(0, 1, [n,8])
    
    mean = X[:,0] + X[:,1] + 0.8 * X[:,2] + 1.2 * X[:,3] - (1.1 * X[:,4] + 0.9 * X[:,5]) + np.sin(3*X[:,6]) - np.sin(3*X[:,7]) + np.cos(3*X[:,8]) - np.cos(3*X[:,9]) + X[:,0]* X[:,2] - X[:,1]* X[:,5] + X[:,3]* X[:,7] - X[:,4]* X[:,9]
    
    dist = Dist(X)
    dist = np.array(dist)  
    theta = np.array([0.9, 100, 0.4])  # 5 neighbors with cov>e3
    cov = pow(theta[0],2) *  np.exp(-dist*theta[1])  + pow(theta[2],2) * np.identity(len(dist))
    
    Y = np.random.multivariate_normal(mean, cov, 1)
    Y = Y.reshape(n,1)
    
    layer = np.hstack((X, Y))
    return (dist,layer)   

n_layer = 100
grid = 40
n = grid * grid
dist = np.zeros((n_layer,n,n))
data = np.zeros((n_layer,n,11))
for seed in range(n_layer):
    dist[seed,:,:],data[seed,:,:] = Simu(grid, seed)
    print(seed)
    
# save simulated data
file_name = 'dist100_est.pickle'   # 40*40 100 layers  for par estimation  # layer: (n_layer, n=grid**2, dim=11)
open_file = open(file_name, "wb")
pickle.dump(dist, open_file)
open_file.close()

file_name = 'data100_est.pickle'
open_file = open(file_name, "wb")
pickle.dump(data, open_file)
open_file.close()



### Model Estimation ###
# load data
n_layer = 100
n = 1600
data = pd.read_pickle("data100_est.pickle")
dist = pd.read_pickle("dist100_est.pickle")

device = torch.device("cpu")

## NN model
class NN(nn.Module):   
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=10, out_features=32),
            nn.ReLU()
        )  
             
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
       
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU()
        )
   
        self.fc4 = nn.Linear(in_features=8, out_features=1)
        
    def forward(self, x):
        z = self.fc1(x)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc4(z)
        z = torch.squeeze(z)    
        return z

model = NN()
model = model.float()
model.to(device)

error = nn.MSELoss()
learning_rate = 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataNN = data.reshape(n_layer*1600,11)
train_data = torch.from_numpy(dataNN)
train_data = train_data.float()
train_data = train_data.to(device)

batch_size = 1000
batches = int(np.ceil(n_layer*n/batch_size))
num_epochs = 300

loss_hist = np.zeros(num_epochs)

## initial NN
for epoch in range(num_epochs):
    for index in range(batches):
        x = train_data[index*batch_size:(index+1)*batch_size,0:10]   
        y = train_data[index*batch_size:(index+1)*batch_size,10]
        
        # Forward pass 
        outputs = model(x)
        loss = error(outputs, y)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
        
        loss_hist[epoch] += loss.data
    loss_hist[epoch] = loss_hist[epoch]/batches
    #print("Epoch: {}, Loss: {:.7f}".format(epoch, loss_hist[epoch])) 
      
# get initial residual
res = train_data[:,10] - model(train_data[:,0:10])
res = res.detach().numpy()
mse = np.mean(pow(res,2))
print(mse)

# function for cov estimation
def ParLayerL(theta,dist,resi,l):   #l for each layer  theta: theta, pho, tao
  sigmal = pow(theta[0],2) *  np.exp(-dist[l]*theta[1])  + pow(theta[2],2) * np.identity(len(dist[l]))
  detl = max(det(sigmal),1e-10)
  if l==0:
    print('theta = %d, %d, %d \n' % (theta[0], theta[1], theta[2]))
  Ll = 1/2 * (np.log(detl) + resi[l].T @ inv(sigmal) @ resi[l])
  #print(Ll)
  return Ll

def NegLogl(theta,dist,resi):
    L = 0
    for l in range(100):  # n_layer=100
        L += ParLayerL(theta,dist,resi,l)
        #print(L[0,0])
    print(L[0,0])
    return L[0,0]

n_iter = 20
for i in range(n_iter):
    print('iteration number = %d \n' % (i+1))
    # 1. cov estimation
    resiline = res.reshape(n_layer,1600,1)
    thetahat = minimize(NegLogl, np.array([1,100,1]), args=(dist,resiline), method='nelder-mead', options={'fatol': 5, 'maxiter': 30, 'disp': True})
    theta = thetahat.x
    print(theta)
    
    if i==n_iter-1:
        break
    
    # 2. iterate NN
    def decor_loss(ypre, ytrue, dsigma):
            dif = torch.matmul(dsigma , (ytrue - ypre))
            return (dif**2).mean()

    num_epochs = 50
    learning_rate = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_hist = np.zeros(num_epochs)

    batch_size = n  #size of a layer
    batches = n_layer

    for epoch in range(num_epochs):
        for index in range(batches):
            x = train_data[index*batch_size:(index+1)*batch_size,0:10]   
            y = train_data[index*batch_size:(index+1)*batch_size,10]

            # Forward pass 
            outputs = model(x)

            sigmal = pow(theta[0],2) *  np.exp(-dist[index]*theta[1])  + pow(theta[2],2) * np.identity(len(dist[index]))
            dematl = sqrtm(inv(sigmal))
            dematl = torch.from_numpy(dematl).float()

            loss = decor_loss(outputs, y, dematl)
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            #Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            loss_hist[epoch] += loss.data
        loss_hist[epoch] = loss_hist[epoch]/batches
        print("Epoch: {}, Loss: {:.7f}".format(epoch, loss_hist[epoch]))
    
    # save the nn model
    torch.save(model.state_dict(), f"model_{i}.pth") 
    # get new mse
    res = train_data[:,10] - model(train_data[:,0:10])
    res = res.detach().numpy()
    mse = np.mean(pow(res,2))
    print(mse)
    