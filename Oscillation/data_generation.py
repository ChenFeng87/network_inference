import numpy as np
import torch
import pickle
from sklearn import preprocessing
import time


# Generate time series data (Oscillation system)
def move(point, steps, sets):
    alfa, alfa0, n, beta = sets
    p1, p2, p3, m1, m2, m3 = point

    if p1 < 0:
        p1 = 0
    if p2 < 0:
        p2 = 0
    if p3 < 0:
        p3 = 0

    dm1 = -m1 + alfa / (1 + p3) ** 2 + alfa0
    dm2 = -m2 + alfa / (1 + p1) ** 2 + alfa0
    dm3 = -m3 + alfa / (1 + p2) ** 2 + alfa0
    m1_a = m1 + dm1 * steps
    m2_a = m2 + dm2 * steps
    m3_a = m3 + dm3 * steps

    dp1 = -beta * (p1 - m1_a)
    dp2 = -beta * (p2 - m2_a)
    dp3 = -beta * (p3 - m3_a)
    p1_a = p1 + dp1 * steps + np.random.normal(loc=0, scale=sigma, size=1)
    p2_a = p2 + dp2 * steps + np.random.normal(loc=0, scale=sigma, size=1)
    p3_a = p3 + dp3 * steps + np.random.normal(loc=0, scale=sigma, size=1)
    return [p1_a, p2_a, p3_a, m1_a, m2_a, m3_a]


def init_node():
    initials = np.random.random(6)
    return initials


x1 = []
x2 = []
x3 = []

# set random seeds
np.random.seed(2020)
# NN is the length of each sequence
NN = 4000
# tt is the step length
tt = 0.05
# D characterizes the level of noise
D = 0.004
sigma = np.sqrt(2*tt*D)

# generate 20*4000 time series
print('\n----------   Begin generating time series data ----------')
print('\n--   You need to wait about 10 seconds to get a data.pickle file ')
time1 = time.time()
for i in range(20):
    P0 = init_node()
    P = P0
    xx1 = np.zeros([NN])
    xx2 = np.zeros([NN])
    xx3 = np.zeros([NN])
    for v in range(NN):
        P = move(P, tt, np.array([0.5*40*20, 0.0005*40*20, 2, 5]))
        xx1[v] = P[0]
        xx2[v] = P[1]
        xx3[v] = P[2]
    x1.append(xx1)
    x2.append(xx2)
    x3.append(xx3)

datax1 = np.array(x1)
m, n = datax1.shape
datax2 = np.array(x2)
datax3 = np.array(x3)

# Normalize the data to 0~1
min_max_scaler = preprocessing.MinMaxScaler()
data1 = min_max_scaler.fit_transform(datax1)
min_max_scaler = preprocessing.MinMaxScaler()
data2 = min_max_scaler.fit_transform(datax2)
min_max_scaler = preprocessing.MinMaxScaler()
data3 = min_max_scaler.fit_transform(datax3)

# reshape the sequences into m*3*n*1: 
# m is the number of sequences, 3 is the dim, n is the length of each sequence
youtcome=np.zeros([76020, 3, 200, 1])
for i in range(76020):
    j = i//3801
    k = i % 3801
    youtcome[i, 0, :, :] = data1[j, k:k+200].reshape(200, 1)
    youtcome[i, 1, :, :] = data2[j, k:k+200].reshape(200, 1)
    youtcome[i, 2, :, :] = data3[j, k:k+200].reshape(200, 1)

# Divide the data into three parts:
# train 80%, validation 10%, test 10%
index = [o for o in range(youtcome.shape[0])]
np.random.shuffle(index)
ynew = youtcome[index, :, :, :]
train_data = ynew[: youtcome.shape[0] // 10 * 8, :, :, :]
val_data = ynew[youtcome.shape[0] // 10 * 8:youtcome.shape[0] // 10 * 9, :, :, :]
test_data = ynew[youtcome.shape[0] // 10 * 9:, :, :, :]

train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data)
test_data = torch.tensor(test_data)

print('\n Train data size:', train_data.shape)
print('\n Val data size:' , val_data.shape)
print('\n Test data size:' , test_data.shape)
time2 = time.time()
print('\n----------   Finsh generating time series data ----------')

results = [train_data, val_data, test_data]
with open('data.pickle', 'wb') as f:
    pickle.dump(results, f)

print('\n--The code finishes running, and cost %d seconds' % (time2-time1))
