import torch
import numpy as np
from torch import nn
import pickle
import time
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import rcParams

print('\n---------- Use this part to get Figure 5B,  ----------')
print('\n--   You need to wait about 30 seconds to obtain picture PC1_PC2.png for model 4')
# import the single cell date from state 16
with open('state_16.pickle','rb+') as f:
    cell_16 = pickle.load(f)
time1 =time.time()
# it contains 75 cells in 16-cell state
list1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
      31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,
      59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74]

# Initialization
def init_node():
    initials = np.random.random(48)
    number= random.choice(list1)
    z11 = np.zeros([1, 48])
    z11[0,:] = initials*0.01+cell_16[number]
    return z11

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=48, out_features=32, bias=True),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=48, bias=True)
            )

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        output1 = torch.sigmoid(output)
        return output1

def predict_expression(hypar_k, dynamics_learner):
    random.seed(2020)
    np.random.seed(2020)
    xx1=torch.zeros([NN,Steps,48])
    for i in range(NN):
        x2 = init_node()
        for j in range(Steps-1):
            x1 = torch.as_tensor(x2, dtype=torch.float32)
            xx1[i,j,:]=x1[0].clone()
            x3 = dynamics_learner(x1)
            x2 = x1 + x3 * tt - x1 * tt * hypar_k
        xx1[i,Steps-1,:]=x2[0].clone()
    return xx1

NN = 1000
Steps = 13
tt = 0.5
# model 4 is a ensemble model for k=0.05,0.15,0.25 and 0.35
dynamics_learner_k005 = FullyConnected()
dynamics_learner_k005.load_state_dict(torch.load('Parameters_saved_k005.pickle'))
dynamics_learner_k005.eval()
data_k005 = predict_expression(0.05, dynamics_learner_k005)

dynamics_learner_k015 = FullyConnected()
dynamics_learner_k015.load_state_dict(torch.load('Parameters_saved_k015.pickle'))
dynamics_learner_k015.eval()
data_k015 = predict_expression(0.15, dynamics_learner_k015)

dynamics_learner_k025 = FullyConnected()
dynamics_learner_k025.load_state_dict(torch.load('Parameters_saved_k025.pickle'))
dynamics_learner_k025.eval()
data_k025 = predict_expression(0.25, dynamics_learner_k025)

dynamics_learner_k035 = FullyConnected()
dynamics_learner_k035.load_state_dict(torch.load('Parameters_saved_k035.pickle'))
dynamics_learner_k035.eval()
data_k035 = predict_expression(0.35, dynamics_learner_k035)

data_time_16=torch.zeros([4000,48])
data_time_32=torch.zeros([4000,48])
data_time_64=torch.zeros([4000,48])

data_time_16[0:1000,:]=data_k005[:,0,:]
data_time_16[1000:2000,:]=data_k015[:,0,:]
data_time_16[2000:3000,:]=data_k025[:,0,:]
data_time_16[3000:4000,:]=data_k035[:,0,:]

data_time_32[0:1000,:]=data_k005[:,5,:]
data_time_32[1000:2000,:]=data_k015[:,5,:]
data_time_32[2000:3000,:]=data_k025[:,5,:]
data_time_32[3000:4000,:]=data_k035[:,5,:]

data_time_64[0:1000,:]=data_k005[:,12,:]
data_time_64[1000:2000,:]=data_k015[:,12,:]
data_time_64[2000:3000,:]=data_k025[:,12,:]
data_time_64[3000:4000,:]=data_k035[:,12,:]

x_data_16 = np.array(data_time_16.detach())
x_data_32 = np.array(data_time_32.detach())
x_data_64 = np.array(data_time_64.detach())
m_x ,n_x = x_data_16.shape

# Dimensionality reduction according to 64-cell stage data
K=2# the dimension
model = PCA(n_components=K).fit(x_data_64) 
Z = model.transform(x_data_64) 

# Clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=2018)
kmeans.fit(x_data_64)
pre_y_64 = kmeans.predict(x_data_64)

# three type in 64-cell stage and the corresponding value in 32-cell stage
type1_64=[]
type2_64=[]
type3_64=[]
for i in range(m_x):
    if pre_y_64[i]==0:
        type1_64.append(x_data_64[i])
    elif pre_y_64[i]==1:
        type2_64.append(x_data_64[i])
    else:
        type3_64.append(x_data_64[i])

x_type1_64 = np.array(type1_64)
x_type2_64 = np.array(type2_64)
x_type3_64 = np.array(type3_64)
print('\n type1 size in 64-cell state:', x_type1_64.shape)
print('\n type2 size in 64-cell state:' , x_type2_64.shape)
print('\n type3 size in 64-cell state:' , x_type3_64.shape)

cell_64_type1=model.transform(x_type1_64)
cell_64_type2=model.transform(x_type2_64)
cell_64_type3=model.transform(x_type3_64)

type1_32 = []
type2_32 = []
type3_32 = []
for i in range(m_x):
    if pre_y_64[i]==0:
        type1_32.append(x_data_32[i])
    elif pre_y_64[i]==1:
        type2_32.append(x_data_32[i])
    else:
        type3_32.append(x_data_32[i])

x_type1_32 = np.array(type1_32)
x_type2_32 = np.array(type2_32)
x_type3_32 = np.array(type3_32)

cell_64_type1_32=model.transform(x_type1_32)
cell_64_type2_32=model.transform(x_type2_32)
cell_64_type3_32=model.transform(x_type3_32)
time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))

plt.figure(figsize=(10, 4))
config = {
    "font.family": 'Arial',
    "font.size": 8,
}
rcParams.update(config)
plt.subplot(1, 2, 1)
area = np.pi * 1.6**2  # the area of the point
plt.scatter(cell_64_type1_32[:,0],cell_64_type1_32[:,1],color='#6464ff',s=area, alpha=0.8,label='TE')
plt.scatter(cell_64_type2_32[:,0],cell_64_type2_32[:,1],color='#0dbc3e',s=area, alpha=0.8,label='ICM')
plt.scatter(cell_64_type3_32[:,0],cell_64_type3_32[:,1],color='#fd8d3c',s=area, alpha=0.8,label='ICM')
plt.ylabel('PC$_2$', fontsize=8)
plt.xlabel('PC$_1$', fontsize=8)
plt.ylim(-0.3,0.7)
plt.xlim(-0.5, 1.25)
plt.yticks([-0.3,0,0.7],['-0.3','0','0.7'])
plt.xticks([-0.5,0,1.25],['-0.5','0','1.25'])
plt.title('32-cell stage (model 4)', fontsize=12)
plt.tick_params(labelsize=8)
plt.legend(fontsize=8,frameon=True,loc='lower right')

plt.subplot(1, 2, 2)
area = np.pi * 1.6**2 
plt.scatter(cell_64_type1[:,0],cell_64_type1[:,1],color='#6464ff',s=area, alpha=0.8,label='TE')
plt.scatter(cell_64_type2[:,0],cell_64_type2[:,1],color='#0dbc3e',s=area, alpha=0.8,label='EPI')
plt.scatter(cell_64_type3[:,0],cell_64_type3[:,1],color='#fd8d3c',s=area, alpha=0.8,label='PE')
plt.ylabel('PC$_2$', fontsize=8)
plt.xlabel('PC$_1$', fontsize=8)
plt.ylim(-0.35,0.5)
plt.xlim(-0.65, 0.75)
plt.yticks([-0.35,0,0.5],['-0.35','0','0.5'])
plt.xticks([-0.65,0,0.75],['-0.65','0','0.75'])
plt.title('64-cell stage (model 4)', fontsize=12)
plt.tick_params(labelsize=8)
plt.legend(fontsize=8,frameon=True,loc='lower right')
plt.savefig('PC1_PC2.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
