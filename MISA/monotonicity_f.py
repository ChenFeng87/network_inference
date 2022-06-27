# %%
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

print('\n---------- Use this part to predict the steady state ----------')
print('\n--   You need to wait about 10 seconds to obtain the monotonicity of f1 and f2 with respect to X2, i.e. f1_f2_X2.png')
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

# 0~1 divide NK=20 
NK = 20
x_inital = np.zeros([1, 2])

# fix x1=0.5, no change
x_inital[0][0] = 0.5
tt = np.linspace(0, 1, NK)
f1 = []
f2 = []

print('\n----------   begin calculating f1 and f2  ----------')
for i in range(NK):
    x_inital[0][1] = tt[i]
    x1_1 = torch.as_tensor(x_inital, dtype=torch.float32)
    x1_2 = torch.cat([x1_1, x_ab], dim=1)
    x1_3 = dynamics_learner(x1_2)
    f1.append(x1_3[0][0].detach().numpy())
    f2.append(x1_3[0][1].detach().numpy())
print('\n----------   calculating f1 and f2 finish ----------')

print('\n----------   begin picture ----------')
plt.figure(figsize=(2, 2))
plt.scatter(tt, f1, label='$f_1$', color='', marker='o', edgecolors='#0dbc3e', s=20)  # dark green
plt.scatter(tt, f2, label='$f_2$', color='', marker='o', edgecolors='#6464ff', s=20)  # dark blue
plt.yticks([0, 0.8, 1.6])
plt.tick_params(labelsize=8)
plt.xticks([0, 0.5, 1.0])
plt.xlabel("X$_2$", fontsize=8, family='Arial')
plt.legend()
plt.savefig('f1_f2_X2.png', dpi=300, bbox_inches='tight')
plt.show()
print('\n----------   picture finish ----------')

time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))

# %%
