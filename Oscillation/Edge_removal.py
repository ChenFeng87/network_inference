import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = {
    "font.family": 'Arial',  # 设置字体类型
    "font.size": 8,
}
rcParams.update(config)


def init_node():
    initials = np.random.random(3)
    z11 = np.zeros([1, 3])
    z11[0][0] = initials[0]
    z11[0][1] = initials[1]
    z11[0][2] = initials[2]
    return z11


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=3, out_features=32, bias=True),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=3, bias=True)
            )

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        output1 = torch.sigmoid(output)
        return output1


print('\n---------- Use this Use this part to infer the structure of GRNs ----------')
print('\n--   You need to wait about 10 seconds to obtain picture The_role_Lacl.png, The_role_TetR.pn, The_role_CI.pn')
print('\n----------   import the DNN model  ----------')
time1 = time.time()
dynamics_learner = FullyConnected()

dynamics_learner.load_state_dict(torch.load('Parameters_saved.pickle'))
dynamics_learner.eval()
print('\n----------   import DNN is finish  ----------')

# set initial points
# X1: Lacl;
# X2: TetR;
# X3: CI;
kk = 0.5
x1 = np.array([[1, 1, 1]])*kk
x2 = np.array([[1, 1, 1]])*kk
x3 = np.array([[1, 1, 1]])*kk
x4 = np.array([[1, 1, 1]])*kk
x5 = np.array([[1, 1, 1]])*kk
# the length of sequence we predict
NK = 1000

x_1 = []
x_2 = []
x_3 = []

xx_11_1 = []
xx_11_2 = []
xx_11_3 = []

xx_21_1 = []
xx_31_1 = []

xx_12_2 = []
xx_22_2 = []
xx_32_2 = []

xx_13_3 = []
xx_23_3 = []
xx_33_3 = []

# get the original sequence 
print('\n----------   get the original sequence  ----------')
x_1.append(x1[0][0])
x_2.append(x1[0][1])
x_3.append(x1[0][2])

for i in range(NK):
    x1_1 = torch.as_tensor(x1, dtype=torch.float32)
    x1_3 = dynamics_learner(x1_1)
    x1 = x1_1+x1_3 * 0.05 - x1_1 * 0.05 * 1
    x_1.append(x1[0][0])
    x_2.append(x1[0][1])
    x_3.append(x1[0][2])

# edge removal

kcut = 0 # 0 represent X1, i.e.Lacl

print('\n----------   block the link from X_kcut to X1  ----------')
xx_11_1.append(x2[0][0])
xx_11_2.append(x2[0][1])
xx_11_3.append(x2[0][2])

x2_5 = np.zeros([1, 3])
for i in range(NK):
    x2_1 = torch.as_tensor(x2, dtype=torch.float32)
    x2_2 = x2_1.clone()
    x2_3 = dynamics_learner(x2_1)
    x2_2[0][kcut] = 0
    x2_4 = dynamics_learner(x2_2)

    x2_5[0][0] = x2_4[0][0]
    x2_5[0][1] = x2_3[0][1]
    x2_5[0][2] = x2_3[0][2]

    x2 = x2_1 + x2_5 * 0.05 - x2_1 * 0.05 * 1
    xx_11_1.append(x2[0][0])
    xx_11_2.append(x2[0][1])
    xx_11_3.append(x2[0][2])


print('\n----------   block the link from X_kcut to X2  ----------')
xx_12_2.append(x3[0][1])

x3_5 = np.zeros([1, 3])
for i in range(NK):
    x3_1 = torch.as_tensor(x3, dtype=torch.float32)
    x3_2 = x3_1.clone()
    x3_3 = dynamics_learner(x3_1)
    x3_2[0][kcut] = 0
    x3_4 = dynamics_learner(x3_2)

    x3_5[0][0] = x3_3[0][0]
    x3_5[0][1] = x3_4[0][1]
    x3_5[0][2] = x3_3[0][2]

    x3 = x3_1 + x3_5 * 0.05 - x3_1 * 0.05 * 1
    xx_12_2.append(x3[0][1])

print('\n----------   block the link from X_kcut to X3  ----------')
xx_13_3.append(x4[0][2])

x4_5 = np.zeros([1, 3])
for i in range(NK):
    x4_1 = torch.as_tensor(x4, dtype=torch.float32)
    x4_2 = x4_1.clone()
    x4_3 = dynamics_learner(x4_1)
    x4_2[0][kcut] = 0
    x4_4 = dynamics_learner(x4_2)

    x4_5[0][0] = x4_3[0][0]
    x4_5[0][1] = x4_3[0][1]
    x4_5[0][2] = x4_4[0][2]

    x4 = x4_1 + x4_5 * 0.05 - x4_1 * 0.05 * 1
    xx_13_3.append(x4[0][2])

print('\n----------   begin picture ----------')

tt = np.linspace(0, NK, NK+1)
plt.figure(figsize=(7, 2))

plt.subplot(1, 3, 1)
plt.plot(tt, x_1, label='Lacl-before', color='#98FB98', linewidth=2.0)
plt.plot(tt, xx_11_1, label='Lacl-after', color='#0dbc3e', linewidth=2.0)
plt.xlabel("Time", fontsize=8)
plt.ylim(-0.1, 1.1)
plt.xlim(-10, 1010)
plt.xticks([0, 500, 1000])
plt.yticks([0, 0.5, 1.0])
plt.xlabel("Time (min)", fontsize=8, family='Arial')
plt.ylabel("Proteins per cell (normalization)", fontsize=8, family='Arial')
plt.title("Block the link from Lacl to Lacl")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(tt, x_2, label='TetR-before', color='#ADD8E6', linewidth=2.0)
plt.plot(tt, xx_12_2, label='TetR-after', color='#6464ff', linewidth=2.0)
plt.xlabel("Time", fontsize=8)
plt.ylim(-0.1, 1.1)
plt.xlim(-10, 1010)
plt.xticks([0, 500, 1000])
plt.yticks([])
plt.xlabel("Time (min)", fontsize=8, family='Arial')
plt.title("Block the link from Lacl to TetR")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(tt, x_3, label='CI-before', color='#fec194', linewidth=2.0)
plt.plot(tt, xx_13_3, label='CI-after', color='#fd8e3c', linewidth=2.0)
plt.xlabel("Time", fontsize=8)
plt.ylim(-0.1, 1.1)
plt.xlim(-10, 1010)
plt.xticks([0, 500, 1000])
plt.yticks([])
plt.xlabel("Time (min)", fontsize=8, family='Arial')
plt.title("Block the link from Lacl to CI")
plt.legend()

plt.subplots_adjust(hspace=0.3)
plt.savefig('The_role_Lacl.png', dpi=300, bbox_inches='tight')
plt.show()

print('\n----------   picture finish ----------')

time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))
