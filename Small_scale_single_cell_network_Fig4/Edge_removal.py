import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# in this part, we set x1:CDX2, x2:GATA6, x3:NANOG, x4:OCT4
config = {
    "font.family": 'Arial',  
    "font.size": 8,
}
rcParams.update(config)


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=4, out_features=32, bias=True),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=4, bias=True)
            )

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        output1 = torch.sigmoid(output)
        return output1

print('\n---------- Use this part to infer the structure of GRNs ----------')
print('\n--   You need to wait about 10 seconds to obtain picture The_role_X1.png')
print('\n----------   import the DNN model  ----------')
time1 = time.time()
dynamics_learner = FullyConnected()
dynamics_learner.load_state_dict(torch.load('Parameters_saved_k09.pickle'))
dynamics_learner.eval()

# the value of corresponding hyperparameter k,
# this part we default k=0.9, you can give it other values
adj = torch.zeros([1, 4])
adj[0][0] = 0.9
adj[0][1] = 0.9
adj[0][2] = 0.9
adj[0][3] = 0.9
# set initial points
m1 = 0.5
m2 = 0.5
m3 = 0.5
m4 = 0.5
x1 = np.array([[m1, m2, m3, m4]])
x2 = np.array([[m1, m2, m3, m4]])
x3 = np.array([[m1, m2, m3, m4]])
x4 = np.array([[m1, m2, m3, m4]])
x5 = np.array([[m1, m2, m3, m4]])
x6 = np.array([[m1, m2, m3, m4]])

# NN is the length of sequence we predict
NN = 20
x_1 = np.zeros([1, NN])
x_2 = np.zeros([1, NN])
x_3 = np.zeros([1, NN])
x_4 = np.zeros([1, NN])
x_5 = np.zeros([1, NN])

xx_11_1 = np.zeros([1, NN])
xx_11_2 = np.zeros([1, NN])
xx_11_3 = np.zeros([1, NN])
xx_11_4 = np.zeros([1, NN])
xx_11_5 = np.zeros([1, NN])

xx_12_1 = np.zeros([1, NN])
xx_12_2 = np.zeros([1, NN])
xx_12_3 = np.zeros([1, NN])
xx_12_4 = np.zeros([1, NN])
xx_12_5 = np.zeros([1, NN])

xx_13_1 = np.zeros([1, NN])
xx_13_2 = np.zeros([1, NN])
xx_13_3 = np.zeros([1, NN])
xx_13_4 = np.zeros([1, NN])
xx_13_5 = np.zeros([1, NN])

xx_14_1 = np.zeros([1, NN])
xx_14_2 = np.zeros([1, NN])
xx_14_3 = np.zeros([1, NN])
xx_14_4 = np.zeros([1, NN])
xx_14_5 = np.zeros([1, NN])

xx_15_1 = np.zeros([1, NN])
xx_15_2 = np.zeros([1, NN])
xx_15_3 = np.zeros([1, NN])
xx_15_4 = np.zeros([1, NN])
xx_15_5 = np.zeros([1, NN])

# get the original sequence 
print('\n----------   get the original sequence  ----------')
for i in range(NN):
    x1_1 = torch.as_tensor(x1, dtype=torch.float32)
    x1_3 = dynamics_learner(x1_1)
    x1 = x1_1 + x1_3 * 1 - x1_1 * adj * 1
    x_1[0][i] = x1[0][0].detach().numpy()
    x_2[0][i] = x1[0][1].detach().numpy()
    x_3[0][i] = x1[0][2].detach().numpy()
    x_4[0][i] = x1[0][3].detach().numpy()

# edge removal
# in this part, we set x1:CDX2, x2:GATA6, x3:NANOG, x4:OCT4
var_number=0 # 0: the role of x1, and 1 the role x2.
print('\n----------   block the link from X1 (CDX2) to X1 (CDX2)  ----------')
for i in range(NN):
    x2_1 = torch.as_tensor(x2, dtype=torch.float32)
    x2_3 = dynamics_learner(x2_1)

    output2_cut_11 = x2_1.clone()
    output2_cut_11[0][var_number] = 0
    output3_cut_11 = dynamics_learner(output2_cut_11)
    x2_3[0][0] = output3_cut_11[0][0]

    x2 = x2_1 + x2_3 * 1 - x2_1 * adj * 1
    xx_11_1[0][i] = x2[0][0].detach().numpy()
    xx_11_2[0][i] = x2[0][1].detach().numpy()
    xx_11_3[0][i] = x2[0][2].detach().numpy()
    xx_11_4[0][i] = x2[0][3].detach().numpy()

print('\n----------   block the link from X1 (CDX2) to X2 (GATA6)  ----------')
for i in range(NN):
    x3_1 = torch.as_tensor(x3, dtype=torch.float32)
    x3_3 = dynamics_learner(x3_1)

    output2_cut_11 = x3_1.clone()
    output2_cut_11[0][var_number] = 0 
    output3_cut_11 = dynamics_learner(output2_cut_11)
    x3_3[0][1] = output3_cut_11[0][1]

    x3 = x3_1 + x3_3 * 1 - x3_1 * adj * 1
    xx_12_1[0][i]=x3[0][0].detach().numpy()
    xx_12_2[0][i]=x3[0][1].detach().numpy()
    xx_12_3[0][i]=x3[0][2].detach().numpy()
    xx_12_4[0][i] = x3[0][3].detach().numpy()

print('\n----------   block the link from X1 (CDX2) to X3 (NANOG)  ----------')
for i in range(NN):
    x4_1 = torch.as_tensor(x4, dtype=torch.float32)
    x4_3 = dynamics_learner(x4_1)

    output2_cut_11 = x4_1.clone()
    output2_cut_11[0][var_number] = 0 
    output3_cut_11 = dynamics_learner(output2_cut_11)
    x4_3[0][2] = output3_cut_11[0][2] 

    x4 = x4_1 + x4_3 * 1 - x4_1 * adj * 1
    xx_13_1[0][i]=x4[0][0].detach().numpy()
    xx_13_2[0][i]=x4[0][1].detach().numpy()
    xx_13_3[0][i]=x4[0][2].detach().numpy()
    xx_13_4[0][i] = x4[0][3].detach().numpy()

print('\n----------   block the link from X1 (CDX2) to X4 (OCT4)  ----------')
for i in range(NN):
    x5_1 = torch.as_tensor(x5, dtype=torch.float32)
    x5_3 = dynamics_learner(x5_1)

    output2_cut_11 = x5_1.clone()
    output2_cut_11[0][var_number] = 0
    output3_cut_11 = dynamics_learner(output2_cut_11)
    x5_3[0][3] = output3_cut_11[0][3]

    x5 = x5_1 + x5_3 * 1 - x5_1 * adj * 1
    xx_14_1[0][i]=x5[0][0].detach().numpy()
    xx_14_2[0][i]=x5[0][1].detach().numpy()
    xx_14_3[0][i]=x5[0][2].detach().numpy()
    xx_14_4[0][i] = x5[0][3].detach().numpy()

print('\n----------   begin picture ----------')
tt = np.linspace(0, NN, NN)
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
x_1=torch.tensor(x_1)
plt.plot(tt, x_1[0].detach().numpy(), label='CDX2-before', color='#ADD8E6', linewidth=2.0)  # pale green
plt.plot(tt, xx_11_1[0], label='CDX2-after', color='#6464ff', linewidth=2.0)  # dark green
plt.title("Block the link from CDX2 to CDX2")
plt.legend()

plt.subplot(2, 2, 2)
x_2=torch.tensor(x_2)
plt.plot(tt, x_2[0].detach().numpy(), label='GATA6-before', color='#ADD8E6', linewidth=2.0)  # light blue
plt.plot(tt, xx_12_2[0], label='GATA6-after', color='#6464ff', linewidth=2.0)  # dark blue
plt.title("Block the link from CDX2 to GATA6")
plt.legend()

plt.subplot(2, 2, 3)
x_3=torch.tensor(x_3)
plt.plot(tt, x_3[0].detach().numpy(), label='NANOG-before', color='#ADD8E6', linewidth=2.0)  # pale green
plt.plot(tt, xx_13_3[0], label='NANOG-after', color='#6464ff', linewidth=2.0)  # dark green
plt.xlabel("Time", fontsize=12)
plt.title("Block the link from CDX2 to NANOG")
plt.legend()

plt.subplot(2, 2, 4)
x_4=torch.tensor(x_4)
plt.plot(tt, x_4[0].detach().numpy(), label='OCT4-before', color='#ADD8E6', linewidth=2.0)  # light blue
plt.plot(tt, xx_14_4[0], label='OCT4-after', color='#6464ff', linewidth=2.0)  # dark blue
plt.xlabel("Time", fontsize=12)
plt.title("Block the link from CDX2 to OCT4")
plt.legend()

plt.subplots_adjust(hspace=0.2)
plt.savefig('The_role_of_CDX2_with_k09.png', dpi=300, bbox_inches='tight')
plt.show()

print('\n----------   picture finish ----------')

time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))
