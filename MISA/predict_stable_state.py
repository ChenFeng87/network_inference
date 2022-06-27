import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import time


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
print('\n---------- Use this part to predict the steady state ----------')
print('\n--   You need to wait about 120 seconds')

time1 = time.time()
print('\n----------   import the DNN model  ----------')
dynamics_learner = FullyConnected()
dynamics_learner.load_state_dict(torch.load('Parameters_saved.pickle'))
dynamics_learner.eval()
print('\n----------   import DNN is finish  ----------')

# parameters a and b, different values correspond to different number of steady states
# when a=0.5 and b=0.8, it has 2 steady states
ab = np.array([[0.5, 0.8]])
x_ab = torch.as_tensor(ab, dtype=torch.float32)
y = []
z = []
# NN is the number of random initial points
# different initial points will fall into different basins
NN = 10000

print('\n----------   prediction the time serie by DNN  ----------')
for i in range(NN):
    x2 = init_node()
    for j in range(40):
        x1 = torch.as_tensor(x2, dtype=torch.float32)
        x12 = torch.cat([x1, x_ab], dim=1)
        x3 = dynamics_learner(x12)
        x2 = x1 + x3 * 0.5 - x1 * 0.5 * 1
    y.append(x2[0][1])
    z.append(x2[0][0])
yy1 = np.array(y)
zz1 = np.array(z)

print('\n----------   start counting steady states  ----------')
sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0
steady_state = np.zeros([4,2])
for i in range(NN):
    if yy1[i] > 0.7 and zz1[i] > 0.7:
        sum1 = sum1+1
        if sum1>10 and sum1<12:
            steady_state[0,0]=yy1[i]
            steady_state[0,1]=zz1[i]
    if yy1[i] > 0.7 and zz1[i] < 0.2:
        sum2 = sum2+1
        if sum2>10 and sum2<12:
            steady_state[1,0]=yy1[i]
            steady_state[1,1]=zz1[i]

    if yy1[i] < 0.2 and zz1[i] > 0.7:
        sum3 = sum3 + 1
        if sum3>10 and sum3<12:
            steady_state[2,0]=yy1[i]
            steady_state[2,1]=zz1[i]

    if yy1[i] < 0.2 and zz1[i] < 0.2:
        sum4 = sum4 + 1
        if sum4>10 and sum4<12:
            steady_state[3,0]=yy1[i]
            steady_state[3,1]=zz1[i]

sum = np.zeros([4,1])
sum[0,0]=sum1
sum[1,0]=sum2
sum[2,0]=sum3
sum[3,0]=sum4

for os in range(4):
    if sum[os,0]>0:
        print('\n- the steady state is (%s,%s), and the Ratio is %s' % (steady_state[os,0], steady_state[os,1], sum[os,0]/NN))
time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))
