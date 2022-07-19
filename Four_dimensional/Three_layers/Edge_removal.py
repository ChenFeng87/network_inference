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
    initials = np.random.random(4)
    z11 = np.zeros([1, 4])
    z11[0][0] = initials[0]
    z11[0][1] = initials[1]
    z11[0][2] = initials[2]
    z11[0][3] = initials[3]
    return z11


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=6, out_features=32, bias=True),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=4, bias=True)
            )

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        fc3 = self.layer3(fc2)
        output = self.layer4(fc3)
        output1 = torch.sigmoid(output)
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
# when a=0.35 and b=0.3, it has 7 steady states
ab = np.array([[0.35, 0.3]])
x_ab = torch.as_tensor(ab, dtype=torch.float32)

# set initial points
kk = 0.5
x1 = np.array([[1, 1, 1, 1]]) * kk
x2 = np.array([[1, 1, 1, 1]]) * kk
x3 = np.array([[1, 1, 1, 1]]) * kk
x4 = np.array([[1, 1, 1, 1]]) * kk
x5 = np.array([[1, 1, 1, 1]]) * kk

x_1 = []
x_2 = []
x_3 = []
x_4 = []

xx_11_1 = []
xx_11_2 = []
xx_11_3 = []
xx_11_4 = []
xx_12_1 = []
xx_12_2 = []
xx_12_3 = []
xx_12_4 = []
xx_13_1 = []
xx_13_2 = []
xx_13_3 = []
xx_13_4 = []
xx_14_1 = []
xx_14_2 = []
xx_14_3 = []
xx_14_4 = []

x_1.append(x1[0][0])
x_2.append(x1[0][1])
x_3.append(x1[0][2])
x_4.append(x1[0][3])

# get the original sequence 
print('\n----------   get the original sequence  ----------')
for i in range(20):
    x1_1 = torch.as_tensor(x1, dtype=torch.float32)
    x1_2 = torch.cat([x1_1, x_ab], dim=1)
    x1_3 = dynamics_learner(x1_2)
    x1 = x1_1 + x1_3 * 0.5 - x1_1 * 0.5 * 1
    x_1.append(x1[0][0])
    x_2.append(x1[0][1])
    x_3.append(x1[0][2])
    x_4.append(x1[0][3])

# edge removal

print('\n----------   block the link from X1 to X1  ----------')

xx_11_1.append(x2[0][0])
xx_11_2.append(x2[0][1])
xx_11_3.append(x2[0][2])
xx_11_4.append(x2[0][3])
x2_5 = np.zeros([1, 4])
for i in range(20):
    x2_1 = torch.as_tensor(x2, dtype=torch.float32)
    x2_2 = torch.cat([x2_1, x_ab], dim=1)
    x2_3 = dynamics_learner(x2_2)
    x2_2[0][0] = 0
    x2_4 = dynamics_learner(x2_2)
    x2_5[0][0] = x2_4[0][0]
    x2_5[0][1] = x2_3[0][1]
    x2_5[0][2] = x2_3[0][2]
    x2_5[0][3] = x2_3[0][3]
    x2 = x2_1 + x2_5 * 0.5 - x2_1 * 0.5 * 1
    xx_11_1.append(x2[0][0])
    xx_11_2.append(x2[0][1])
    xx_11_3.append(x2[0][2])
    xx_11_4.append(x2[0][3])

print('\n----------   block the link from X1 to X2  ----------')
xx_12_1.append(x3[0][0])
xx_12_2.append(x3[0][1])
xx_12_3.append(x3[0][2])
xx_12_4.append(x3[0][3])
x3_5 = np.zeros([1, 4])
for i in range(20):
    x3_1 = torch.as_tensor(x3, dtype=torch.float32)
    x3_2 = torch.cat([x3_1, x_ab], dim=1)
    x3_3 = dynamics_learner(x3_2)
    x3_2[0][0] = 0
    x3_4 = dynamics_learner(x3_2)

    x3_5[0][0] = x3_3[0][0]
    x3_5[0][1] = x3_4[0][1]
    x3_5[0][2] = x3_3[0][2]
    x3_5[0][3] = x3_3[0][3]
    x3 = x3_1 + x3_5 * 0.5 - x3_1 * 0.5 * 1
    xx_12_1.append(x2[0][0])
    xx_12_2.append(x2[0][1])
    xx_12_3.append(x2[0][2])
    xx_12_4.append(x2[0][3])

print('\n----------   block the link from X1 to X3  ----------')
xx_13_1.append(x4[0][0])
xx_13_2.append(x4[0][1])
xx_13_3.append(x4[0][2])
xx_13_4.append(x4[0][3])
x4_5 = np.zeros([1, 4])
for i in range(20):
    x4_1 = torch.as_tensor(x4, dtype=torch.float32)
    x4_2 = torch.cat([x4_1, x_ab], dim=1)
    x4_3 = dynamics_learner(x4_2)
    x4_2[0][0] = 0
    x4_4 = dynamics_learner(x4_2)

    x4_5[0][0] = x4_3[0][0]
    x4_5[0][1] = x4_3[0][1]
    x4_5[0][2] = x4_4[0][2]
    x4_5[0][3] = x4_3[0][3]
    x4 = x4_1 + x4_5 * 0.5 - x4_1 * 0.5 * 1
    xx_13_1.append(x4[0][0])
    xx_13_2.append(x4[0][1])
    xx_13_3.append(x4[0][2])
    xx_13_4.append(x4[0][3])

print('\n----------   block the link from X1 to X4  ----------')
xx_14_1.append(x5[0][0])
xx_14_2.append(x5[0][1])
xx_14_3.append(x5[0][2])
xx_14_4.append(x5[0][3])
x5_5 = np.zeros([1, 4])
for i in range(20):
    x5_1 = torch.as_tensor(x5, dtype=torch.float32)
    x5_2 = torch.cat([x5_1, x_ab], dim=1)
    x5_3 = dynamics_learner(x5_2)
    x5_2[0][0] = 0
    x5_4 = dynamics_learner(x5_2)

    x5_5[0][0] = x5_3[0][0]
    x5_5[0][1] = x5_3[0][1]
    x5_5[0][2] = x5_3[0][2]
    x5_5[0][3] = x5_4[0][3]
    x5 = x5_1 + x5_5 * 0.5 - x5_1 * 0.5 * 1
    xx_14_1.append(x5[0][0])
    xx_14_2.append(x5[0][1])
    xx_14_3.append(x5[0][2])
    xx_14_4.append(x5[0][3])

print('\n----------   begin picture ----------')

tt = np.linspace(0, 20, 21)
plt.figure(figsize=(2, 10))
plt.subplot(4, 1, 1)
x_1=torch.tensor(x_1)
plt.plot(tt, x_1.detach().numpy(), label='X$_1$-before', color='#98FB98', linewidth=2.0)
plt.plot(tt, xx_11_1, label='X$_1$-after', color='#0dbc3e', linewidth=2.0)
plt.xlabel("")
plt.ylim(-0.1, 0.9)
plt.xlim(-1, 21)
plt.yticks([0, 0.5, 0.9])
plt.xticks([0, 10, 21])
plt.title("Block the link from X1 to X1")
plt.legend()

plt.subplot(4, 1, 2)
x_2=torch.tensor(x_2)
plt.plot(tt, x_2.detach().numpy(), label='X$_2$-before', color='#ADD8E6', linewidth=2.0)
plt.plot(tt, xx_12_2, label='X$_2$-after', color='#6464ff', linewidth=2.0)
plt.xlabel("")
plt.ylim(-0.1, 0.9)
plt.xlim(-1, 21)
plt.yticks([0, 0.5, 0.9])
plt.xticks([0, 10, 21])
plt.title("Block the link from X1 to X2")
plt.legend()

plt.subplot(4, 1, 3)
x_3=torch.tensor(x_3)
plt.plot(tt, x_3.detach().numpy(), label='X$_3$-before', color='#fec194', linewidth=2.0)
plt.plot(tt, xx_13_3, label='X$_3$-after', color='#fd8e3c', linewidth=2.0)
plt.xlabel("")
plt.ylim(-0.1, 0.9)
plt.xlim(-1, 21)
plt.yticks([0, 0.5, 0.9])
plt.xticks([0, 10, 21])
plt.title("Block the link from X1 to X3")
plt.legend()

plt.subplot(4, 1, 4)
x_4=torch.tensor(x_4)
plt.plot(tt, x_4.detach().numpy(), label='X$_4$-before', color='#d5b7db', linewidth=2.0)
plt.plot(tt, xx_14_4, label='X$_4$-after', color='#9451a1', linewidth=2.0)
plt.xlabel("Time", fontsize=8)
plt.ylim(-0.1, 0.9)
plt.xlim(-1, 21)
plt.yticks([0, 0.5, 0.9])
plt.xticks([0, 10, 21])
plt.title("Block the link from X1 to X4")
plt.legend()

plt.subplots_adjust(hspace=0.3)
plt.savefig('The_role_X1.png', dpi=300, bbox_inches='tight')
plt.show()
print('\n----------   picture finish ----------')

time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))
