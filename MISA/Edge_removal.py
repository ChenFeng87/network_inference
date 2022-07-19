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
    initials = np.random.random(2)
    z11 = np.zeros([1, 2])
    z11[0][0] = initials[0]
    z11[0][1] = initials[1]
    return z11


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
            nn.Linear(in_features=32, out_features=2, bias=True)
            )

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        output1 = torch.sigmoid(output) * 2
        return output1


print('\n---------- Use this Use this part to infer the structure of GRNs ----------')
print('\n--   You need to wait about 10 seconds to obtain picture The_role_X1.png')
print('\n----------   import the DNN model  ----------')
time1 = time.time()
dynamics_learner = FullyConnected()
dynamics_learner.load_state_dict(torch.load('Parameters_saved.pickle'))
dynamics_learner.eval()

print('\n----------   import DNN is finish  ----------')

# parameters a and b, different values correspond to different number of steady states
# when a=0.5 and b=0.8, it has 2 steady states
ab = np.array([[0.5, 0.8]])
x_ab = torch.as_tensor(ab, dtype=torch.float32)

# set initial points
kk = 0.5
x1 = np.array([[1, 1]]) * kk
x2 = np.array([[1, 1]]) * kk
x3 = np.array([[1, 1]]) * kk
x4 = np.array([[1, 1]]) * kk
x5 = np.array([[1, 1]]) * kk

# NK is the length of sequence we predict
NK = 20


x_1 = []
x_2 = []

xx_11_1 = []
xx_21_1 = []
xx_12_2 = []
xx_22_2 = []

x_1.append(x1[0][0])
x_2.append(x1[0][1])

# get the original sequence 
print('\n----------   get the original sequence  ----------')
for i in range(NK):
    x1_1 = torch.as_tensor(x1, dtype=torch.float32)
    x1_2 = torch.cat([x1_1, x_ab], dim=1)
    x1_3 = dynamics_learner(x1_2)
    x1 = x1_1 + x1_3 * 0.5 - x1_1 * 0.5 * 1
    x_1.append(x1[0][0])
    x_2.append(x1[0][1])

# edge removal

print('\n----------   block the link from X1 to X1  ----------')
xx_11_1.append(x2[0][0])

x2_5 = np.zeros([1, 2])
for i in range(NK):
    x2_1 = torch.as_tensor(x2, dtype=torch.float32)
    x2_2 = torch.cat([x2_1, x_ab], dim=1)
    x2_3 = dynamics_learner(x2_2)
    x2_2[0][0] = 0
    x2_4 = dynamics_learner(x2_2)
    x2_5[0][0] = x2_4[0][0]
    x2_5[0][1] = x2_3[0][1]
    x2 = x2_1 + x2_5 * 0.5 - x2_1 * 0.5 * 1
    xx_11_1.append(x2[0][0])


print('\n----------   block the link from X1 to X2  ----------')
xx_12_2.append(x3[0][1])

x3_5 = np.zeros([1, 2])
for i in range(NK):
    x3_1 = torch.as_tensor(x3, dtype=torch.float32)
    x3_2 = torch.cat([x3_1, x_ab], dim=1)
    x3_3 = dynamics_learner(x3_2)
    x3_2[0][0] = 0
    x3_4 = dynamics_learner(x3_2)
    x3_5[0][0] = x3_3[0][0]
    x3_5[0][1] = x3_4[0][1]
    x3 = x3_1 + x3_5 * 0.5 - x3_1 * 0.5 * 1
    xx_12_2.append(x3[0][1])

print('\n----------   block the link from X2 to X1  ----------')
xx_21_1.append(x4[0][0])

x4_5 = np.zeros([1, 2])
for i in range(NK):
    x4_1 = torch.as_tensor(x4, dtype=torch.float32)
    x4_2 = torch.cat([x4_1, x_ab], dim=1)
    x4_3 = dynamics_learner(x4_2)
    x4_2[0][1] = 0
    x4_4 = dynamics_learner(x4_2)
    x4_5[0][0] = x4_4[0][0]
    x4_5[0][1] = x4_3[0][1]
    x4 = x4_1 + x4_5 * 0.5 - x4_1 * 0.5 * 1
    xx_21_1.append(x4[0][0])


print('\n----------   block the link from X2 to X2  ----------')
xx_22_2.append(x5[0][1])

x5_5 = np.zeros([1, 2])
for i in range(NK):
    x5_1 = torch.as_tensor(x5, dtype=torch.float32)
    x5_2 = torch.cat([x5_1, x_ab], dim=1)
    x5_3 = dynamics_learner(x5_2)
    x5_2[0][1] = 0
    x5_4 = dynamics_learner(x5_2)
    x5_5[0][0] = x5_3[0][0]
    x5_5[0][1] = x5_4[0][1]
    x5 = x5_1 + x5_5 * 0.5 - x5_1 * 0.5 * 1
    xx_22_2.append(x5[0][1])

print('\n----------   begin picture ----------')

tt = np.linspace(0, NK, NK+1)
plt.figure(figsize=(4.5, 2))
plt.subplot(1, 2, 1)
x_1=torch.tensor(x_1)
plt.plot(tt, x_1.detach().numpy(), label='X1-before', color='#98FB98', linewidth=4.0)  # pale green
plt.plot(tt, xx_11_1, label='X1-after', color='#0dbc3e', linewidth=4.0)  # dark green
plt.ylim(0, 1.4)
plt.xlim(-0.5, 21)
plt.yticks([0, 0.7, 1.4])
plt.tick_params(labelsize=8)
plt.xticks([0, 10, 21])
plt.xlabel("Time", fontsize=8, family='Arial')
plt.ylabel("Expression", fontsize=8, family='Arial')
plt.title("Block the link from X1 to X1")
plt.legend()

plt.subplot(1, 2, 2)
x_2=torch.tensor(x_2)
plt.plot(tt, x_2.detach().numpy(), label='X2-before', color='#ADD8E6', linewidth=4.0)  # light blue
plt.plot(tt, xx_12_2, label='X2-after', color='#6464ff', linewidth=4.0)  # dark blue
plt.xlabel("Time", fontsize=8)
plt.ylim(0, 1.4)
plt.xticks([0, 10, 21])
plt.xlim(-0.5, 21)
plt.yticks([0, 0.7, 1.4])
plt.tick_params(labelsize=8)
plt.yticks([])
plt.title("Block the link from X1 to X2")
plt.legend()

plt.subplots_adjust(hspace=0.1)
plt.savefig('The_role_X1.png', dpi=300, bbox_inches='tight')
plt.show()
print('\n----------   picture finish ----------')

time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))
