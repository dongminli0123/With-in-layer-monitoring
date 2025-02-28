import pickle
import pandas as pd
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from numpy.linalg import norm
from numpy.linalg import det
from numpy.linalg import inv
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

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
    #coor = np.array(layer[[0, 1]])
    coor = np.array(layer[:,[0,1]])
    # sigma = [[]] * n
    my_pool = Pool()
    func = partial(row,n,coor)
    sigma = my_pool.imap(func,range(n))
    my_pool.close()
    my_pool.join()
    return list(sigma)

def covar(theta, dist):
    cov = pow(theta[0],2) *  np.exp(-dist * theta[1])  + pow(theta[2],2) * np.identity(len(dist))
    return cov

def weight(dist, lamda = 0.2, gamma = 500):
    w = np.power((1 - lamda), gamma * dist)
    return w

def OC_loc(bottomleft,length):
    """
    Generate indices for all OC locations 40 * (x - 1) + y
    
    Args:
        bottomleft: The (x, y) coordinate of the bottom-left corner.
        length: length of OC square
    """
    x_start, y_start = bottomleft
    indices = [
        40 * (x_start + i - 1) + (y_start + j) - 1
        for i in range(length)
        for j in range(length)
    ]
    return indices

def SeqOrder(mat_dim):
    """
    mat_dim: dimension of input matrix (nroww=ncol)
    original order: row by row, from left to right, down to up (simulated data)
    output order: from downright corner (up first) to upleft corner
    original order: row by row, from left to right, up to down (default of matrix)
    output order: from upright corner (down first) to downleft corner (default of matrix)
    index is the orginal index order, value of the given index is the new index order
    """
    n_tot = mat_dim*mat_dim
    in_order = np.array(range(n_tot))
    out_order = in_order
    x = np.zeros(n_tot) # record
    y = np.zeros(n_tot)
    # initial coordinate indices (note: index = dim-1)
    # print("point No. 1")
    x[0], y[0] = mat_dim-1, 1-1
    oo = x[0] + y[0] * mat_dim
    out_order[int(oo)] = 0
    # print("coordinate:(", x[0], ",", y[0], ")")
    
    # print("point No. 2")
    x[1], y[1] = mat_dim-1, 1
    oo = x[1] + y[1] * mat_dim
    out_order[int(oo)] = 1
    # print("coordinate:(", x[1], ",", y[1], ")")
    
    for iter in range(1,n_tot-1): # double check
        # print("point No. ", iter+2)
        # track (coordinate of next point)
        if x[iter] == mat_dim-1 and y[iter] == 0:
            # print("first point")
            pass

        # special case: upleft & downright corner
        elif x[iter] == 0 and y[iter] == 0:
            if (mat_dim % 2) == 0: x[iter+1], y[iter+1] = x[iter], y[iter]+1 # straight down
            else: x[iter+1], y[iter+1] = x[iter]+1, y[iter]+1
        elif x[iter] == mat_dim-1 and y[iter] == mat_dim-1:
            if (mat_dim % 2) == 0: x[iter+1], y[iter+1] = x[iter]-1, y[iter]-1 #
            else: x[iter+1], y[iter+1] = x[iter]-1, y[iter] # straight left
        # right boundary
        elif x[iter] == mat_dim-1:
            # print("right boundary")
            if x[iter-1] == mat_dim-1: x[iter+1], y[iter+1] = x[iter]-1, y[iter]-1
            else: x[iter+1], y[iter+1] = x[iter], y[iter]+1
        # up boundary
        elif y[iter] == 0: 
            if y[iter-1] == 0: x[iter+1], y[iter+1] = x[iter]+1, y[iter]+1 
            else: x[iter+1], y[iter+1] = x[iter]-1, y[iter]
        # left boundary
        elif x[iter] == 0: 
            if x[iter-1] == 0: x[iter+1], y[iter+1] = x[iter]+1, y[iter]+1
            else: x[iter+1], y[iter+1] = x[iter], y[iter]+1
        # down boundary
        elif y[iter] == mat_dim-1: 
            if y[iter-1] == mat_dim-1: x[iter+1], y[iter+1] = x[iter]-1, y[iter]-1 
            else: x[iter+1], y[iter+1] = x[iter]-1, y[iter]     
        # otherwise
        else: #if x[iter] < mat_dim-1 and y[iter] > 0:
            x[iter+1], y[iter+1] = x[iter]+x[iter]-x[iter-1], y[iter]+y[iter]-y[iter-1]
        # print("coordinate:(", x[iter+1], ",", y[iter+1], ")")
        # original order index
        oo = x[iter+1] + y[iter+1] * mat_dim
        out_order[int(oo)] = iter+1
    return out_order


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
#model.to(device)

state_dict = torch.load('model_1.pth')
model.load_state_dict(state_dict)
model.eval()


# generate data
def Simu_OC(grid, seed, shift):
    np.random.seed(seed)
    random.seed(seed)
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
    
    bottomleft = (15, 15)
    OC_indices = OC_loc(bottomleft,10)
    Y[OC_indices,0] = Y[OC_indices,0] + shift
    
    layer = np.hstack((X, Y))
    return (dist,layer)   

# Online monitoring process
def mon_t(layer, p, h):
    seq_order = SeqOrder(40)
    reorder= []
    for i in range(len(list(seq_order))):
        indx = np.where(seq_order==i)
        reorder.append(layer[indx[0][0],:])
    reorder = np.array(reorder)

    n = layer.shape[0]  # total data points in one layer
    res = np.zeros(n)
    d = 5  # how many previous points for decorrelation
    dist = Dist(reorder)
    dist = np.array(dist)
    theta = np.array([0.90220131, 100.40453363, 0.85088577])
    cov = covar(theta, dist)
    w = weight(dist, 0.2, p)  #p is the parameter in weight

    t = 0
    input_x = torch.from_numpy(reorder[t,0:10]).float()
    with torch.no_grad():  
        output = model(input_x)
    output = output.detach().numpy()
    res[t] = reorder[t,10] - output
    Q = res[t] * np.ones(n) # local monitoring statistics for all locations
    Q_top = Q * np.ones(n)  # global monitoring statistic of all times

    for t in range(1,n): # t+1 is actual time
        #print(t)
        ## get the prediction residual
        input_x = torch.from_numpy(reorder[t,0:10]).float()
        with torch.no_grad():  
            output = model(input_x)
        output = output.detach().numpy()
        res[t] = reorder[t,10] - output

        ## get the decorrelated residual res_decov
        if (t<=d):
            de = sqrtm(inv(cov[0:(t+1),0:(t+1)]))
            res_de_all = de @ res[0:(t+1)]
            res_decov = res_de_all[t]
        else:
            de = sqrtm(inv(cov[(t-d):(t+1),(t-d):(t+1)]))
            res_de_all = de @ res[(t-d):(t+1)]
            res_decov = res_de_all[d]

        ## update local monitoring statistic Q for all locations
        standard = w[:,0:(t+1)].sum(1)
        Q = w[:,0:t].sum(1) * Q + w[:,t] * res_decov
        Q = np.divide(Q,standard)
        Q[np.isnan(Q)]=0

        Q_top[t] = np.max(Q[0:t])   
        if ( Q_top[t] > h):
            break  
    return (t+1, Q_top[0:t+1])

n_layer = 1000
grid = 40
theta = np.array([0.90220131, 100.40453363, 0.85088577])
h500 = 2.1494957217550947 
h800 = 2.29
h1000 = 2.417656151229066  
h2000 = 2.8699390011809403
h3000 = 3.1586675919844875
h5000 = 3.50580094382226

t500 = np.zeros(n_layer)
t800 = np.zeros(n_layer)
t1000 = np.zeros(n_layer)
t2000 = np.zeros(n_layer)
t3000 = np.zeros(n_layer)
t5000 = np.zeros(n_layer)
 
for i in range(n_layer):
    seed = i
    #print(seed)
    (dist,layer) = Simu_OC(grid, seed, shift)
    (t500[i], Q) = mon_t(layer,500,h500)
    (t800[i], Q) = mon_t(layer,800,h800)
    (t1000[i], Q) = mon_t(layer,1000,h1000)
    (t2000[i], Q) = mon_t(layer,2000,h2000)
    (t3000[i], Q) = mon_t(layer,3000,h3000)
    (t5000[i], Q) = mon_t(layer,5000,h5000)
    
