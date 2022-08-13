import torch
import numpy as np
import pickle
from scipy import interpolate
import random
import time

# The single-cell dataset used in this work is stored in "single_cell_data_set.xls", which is derived from reference:
#"Guo, G. ,Huss, M., Tong, G. Q.,Wang, C., Li Sun, L., Clarke, N. D. and Robson, P. (2010) 
# Resolution of Cell Fate Decisions Revealed by Single-Cell Gene Expression Analysis from Zygote to Blastocyst. 
# Development Cell, 18, 675-685".
# As described in "Small_scale_single_cell_network_Fig4/readme.txt".

# The dimension of this part is 48, and the column names are the same as the "Small_scale_single_cell_network_Fig4/single_cell_data_set.xls" table,
# i.e., Actb, Ahcy, Aqp3, Atp12a, Bmp4, Cdx2, Creb312, Cebpa, Dab2, DppaI, Eomes, Esrrb,
# Fgf4, Fgfr2, Fn1, Gapdh, Gata3, Gata4, Gata6, Grhl1, Grhl2, Hand1, Hnf4a, Id2,
# Klf2, Klf4, Klf5, Krt8, Lcp1, Mbnl3, Msc, Msx2, Nanog, Pdgfa, Pdgfra, Pecam1,
# Pou5f1, Runx1, Sox2, Sall4, Sox17, Snail, Sox13, Tcfap2a, Tcfap2c, Tcf23, Utf1, Tspan8

# Generate time series data (single cell data, and the number of dimensions is 48)
print('\n----------   Begin generating time series data for single cell ----------')
print('\n--   You need to wait about 30 minutes to get a data.pickle file ')
print('\n--   each interpolation cost about 10 minutes ')
time1 = time.time()

# import the single cell date from state 16
with open('state_16.pickle','rb+') as f:
    state_16 = pickle.load(f)
# import the single cell date from state 32
with open('state32_ICM.pickle','rb+') as f:
    state_32ICM = pickle.load(f)
with open('state32_TE.pickle','rb+') as f:
    state_32TE = pickle.load(f)
# import the single cell date from state 64 
with open('state64_EPI.pickle','rb+') as f:
    state_64EPI = pickle.load(f)
with open('state64_PE.pickle','rb+') as f:
    state_64PE = pickle.load(f)
with open('state64_TE.pickle','rb+') as f:
    state_64TE = pickle.load(f)

print("\n 16-cell state: state_16:",state_16.shape)
print("\n 32-cell state: state_32ICM:",state_32ICM.shape)
print("\n 32-cell state: state_32TE:",state_32TE.shape)
print("\n 64-cell state: state_64EPI:",state_64EPI.shape)
print("\n 64-cell state: state_64PE:",state_64PE.shape)
print("\n 64-cell state: state_64TE:",state_64TE.shape)

random.seed(2021)

x1x = np.linspace(1,3,3)
x1x_new = np.linspace(1,3,9)

print('\n----------   Begin interpolation of TE data ----------')
m_TE = 50000
n = 13
youtcome_TE=np.zeros([m_TE,48,n,1])

for kk in range(m_TE):
    a11_TE = random.randint(0,74)
    a22_TE = random.randint(0,61)
    a33_TE = random.randint(0,95)
    
    you_media = np.zeros([48,3])
    you_media[:,0] = state_16[a11_TE, :]
    you_media[:,1] = state_32TE[a22_TE, :]
    you_media[:,2] = state_64TE[a33_TE, :]
    
    for i in range(48):
        f = interpolate.interp1d(x1x,you_media[i,:],kind='quadratic')
        y_ls = f(x1x_new)
        youtcome_TE[kk,i,0:4,0] = y_ls[0:4]
        youtcome_TE[kk,i,4,0] = y_ls[4] + np.random.random(1)[0]*0.01
        youtcome_TE[kk,i,5,0] = y_ls[4]
        youtcome_TE[kk,i,6,0] = y_ls[4] + np.random.random(1)[0]*0.01
        youtcome_TE[kk,i,7:10,0] = y_ls[5:8]
        youtcome_TE[kk,i,10,0] = y_ls[-1] + np.random.random(1)[0]*0.01
        youtcome_TE[kk,i,11,0] = y_ls[-1] 
        youtcome_TE[kk,i,12,0] = y_ls[-1] + np.random.random(1)[0]*0.01

print('\n----------   Begin interpolation of EPI data ----------')
m_EPI = 50000
n = 13
youtcome_EPI=np.zeros([m_EPI,48,n,1])

for kk in range(m_EPI):
    a11_EPI = random.randint(0,74)
    a22_EPI = random.randint(0,50)
    a33_EPI = random.randint(0,18)
    
    you_media = np.zeros([48,3])
    you_media[:,0] = state_16[a11_EPI, :]
    you_media[:,1] = state_32ICM[a22_EPI, :]
    you_media[:,2] = state_64EPI[a33_EPI, :]
    
    for i in range(48):
        f = interpolate.interp1d(x1x,you_media[i,:],kind='quadratic')
        y_ls = f(x1x_new)
        youtcome_EPI[kk,i,0:4,0] = y_ls[0:4]
        youtcome_EPI[kk,i,4,0] = y_ls[4] + np.random.random(1)[0]*0.01
        youtcome_EPI[kk,i,5,0] = y_ls[4]
        youtcome_EPI[kk,i,6,0] = y_ls[4] + np.random.random(1)[0]*0.01
        youtcome_EPI[kk,i,7:10,0] = y_ls[5:8]
        youtcome_EPI[kk,i,10,0] = y_ls[-1] + np.random.random(1)[0]*0.01
        youtcome_EPI[kk,i,11,0] = y_ls[-1] 
        youtcome_EPI[kk,i,12,0] = y_ls[-1] + np.random.random(1)[0]*0.01

print('\n----------   Begin interpolation of PE data ----------')
m_PE = 50000
n = 13
youtcome_PE=np.zeros([m_PE,48,n,1])

for kk in range(m_PE):
    a11_PE = random.randint(0,74)
    a22_PE = random.randint(0,50)
    a33_PE = random.randint(0,43)
    
    you_media = np.zeros([48,3])
    you_media[:,0] = state_16[a11_PE, :]
    you_media[:,1] = state_32ICM[a22_PE, :]
    you_media[:,2] = state_64PE[a33_PE, :]
    
    for i in range(48):
        f = interpolate.interp1d(x1x,you_media[i,:],kind='quadratic')
        y_ls = f(x1x_new)
        youtcome_PE[kk,i,0:4,0] = y_ls[0:4]
        youtcome_PE[kk,i,4,0] = y_ls[4] + np.random.random(1)[0]*0.01
        youtcome_PE[kk,i,5,0] = y_ls[4]
        youtcome_PE[kk,i,6,0] = y_ls[4] + np.random.random(1)[0]*0.01
        youtcome_PE[kk,i,7:10,0] = y_ls[5:8]
        youtcome_PE[kk,i,10,0] = y_ls[-1] + np.random.random(1)[0]*0.01
        youtcome_PE[kk,i,11,0] = y_ls[-1] 
        youtcome_PE[kk,i,12,0] = y_ls[-1] + np.random.random(1)[0]*0.01

# Divide the data into three parts:
# train 5/7, validation 1/7, test 1/7
print('\n----------   Begin dividing the data ----------')
youtcome=np.zeros([150000,48,13,1])
youtcome[0:50000,:,:,:] = youtcome_TE[0:50000,:,:,:]
youtcome[50000:100000,:,:,:] = youtcome_EPI[0:50000,:,:,:]
youtcome[100000:150000,:,:,:] = youtcome_PE[0:50000,:,:,:]
index=[]
index=[o for o in range(youtcome.shape[0])]
np.random.shuffle(index)
ynew=youtcome[index,:,:,:]
train_data = []
val_data = []
test_data = []
train_data = ynew[: youtcome.shape[0] // 7 * 5, :, :, :]
val_data = ynew[youtcome.shape[0] // 7 * 5:youtcome.shape[0] // 7 * 6, :, :, :]
test_data = ynew[youtcome.shape[0] // 7 * 6:, :, :, :]

train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data )
test_data = torch.tensor(test_data)
print('\n Train data size:', train_data.shape)
print('\n Val data size:' , val_data.shape)
print('\n Test data size:' , test_data.shape)
time2 = time.time()
print('\n----------   Finsh generating time series for single cell data  ----------')
results = [train_data, val_data, test_data]
with open('data.pickle','wb') as f:
    pickle.dump(results,f)
print('\n--The code finishes running, and cost %d seconds' % (time2-time1))
