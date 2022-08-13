import numpy as np
import random
import pickle
import torch
from sklearn import preprocessing
import time

# The single-cell dataset used in this work is stored in "single_cell_data_set.xls", which is derived from reference:
#"Guo, G. ,Huss, M., Tong, G. Q.,Wang, C., Li Sun, L., Clarke, N. D. and Robson, P. (2010) 
# Resolution of Cell Fate Decisions Revealed by Single-Cell Gene Expression Analysis from Zygote to Blastocyst. 
# Development Cell, 18, 675-685".
# As described in "Small_scale_single_cell_network_Fig4/readme.txt".

# Generate time series data (single cell data, and the number of dimensions is 48)
print('\n----------   Begin generating time series data for single cell ----------')
print('\n--   You need to wait about 10 seconds to get a data.pickle file ')
time1 = time.time()
cell_data = np.loadtxt('4gene_raw_data.txt',skiprows=1)
all_cell = np.array(cell_data)

all_cell=np.array(all_cell)
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(all_cell)

cell_1 = data[0:9]
cell_2 = data[9:28]
cell_4 = data[28:51]
cell_8 = data[51:95]
cell_16 = data[95:170]
cell_32 = data[170:283]
cell_64 = data[283:442]

m = 360000  
n = 7
random.seed(1998)
youtcome=np.zeros([m,4,n,1])
for kk in range(m):
    a11=random.randint(0,8)
    a22=random.randint(0,18)
    a33 = random.randint(0,22)
    a44 = random.randint(0,43)
    a55 = random.randint(0,74)
    a66 = random.randint(0,112)
    a77 = random.randint(0,158)
    youtcome[kk,:,0,:]=cell_1[a11,:].reshape(4,1)
    youtcome[kk,:,1,:]=cell_2[a22,:].reshape(4,1)
    youtcome[kk,:,2,:]=cell_4[a33,:].reshape(4,1)
    youtcome[kk,:,3,:]=cell_8[a44,:].reshape(4,1)
    youtcome[kk,:,4,:]=cell_16[a55,:].reshape(4,1)
    youtcome[kk,:,5,:]=cell_32[a66,:].reshape(4,1)
    youtcome[kk,:,6,:]=cell_64[a77,:].reshape(4,1)

index=[]
index=[o for o in range(youtcome.shape[0])]
np.random.shuffle(index)
ynew=youtcome[index,:,:,:]

# Divide the data into three parts:
# train 5/7, validation 1/7, test 1/7
train_data=[]
val_data=[]
test_data=[]
train_data =ynew[: youtcome.shape[0] // 7 * 5, :, :, :]
val_data = ynew[youtcome.shape[0] // 7 * 5:youtcome.shape[0] // 7 * 6, :, :, :]
test_data =ynew[youtcome.shape[0] // 7 * 6:, :, :, :]

train_data=torch.tensor(train_data)
val_data =torch.tensor(val_data )
test_data=torch.tensor(test_data)

print('\n Train data size:', train_data.shape)
print('\n Val data size:' , val_data.shape)
print('\n Test data size:' , test_data.shape)

time2 = time.time()
print('\n----------   Finsh generating time series for single cell data  ----------')

results = [train_data, val_data, test_data]
with open('data.pickle','wb') as f:
    pickle.dump(results,f)
print('\n--The code finishes running, and cost %d seconds' % (time2-time1))
