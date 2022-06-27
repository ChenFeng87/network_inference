import numpy as np
import torch
import pickle
import time

# Generate time series data (MISA model)
def move(point, steps, sets):
    a, b, s, n, k = sets
    x1, x2 = point

    dx1 = a*x1**n / (s**n + x1**n) + b*s**n/(x2**n+s**n)-k*x1
    dx2 = a*x2**n / (s**n + x2**n) + b*s**n/(x1**n+s**n)-k*x2
    return [x1 + dx1 * steps + np.random.normal(loc=0, scale=sigma, size=1),
            x2 + dx2 * steps + np.random.normal(loc=0, scale=sigma, size=1)]

def init_node():
    initials = np.random.random(2)
    return initials


x1 = []
x2 = []
ab_data = np.zeros([80000, 2])

# set random seeds
np.random.seed(2020)
# NN is the length of each sequence
NN = 20
# tt is the step length
tt = 0.5
# D characterizes the level of noise
D = 0.0004
sigma = np.sqrt(2*tt*D)

# generate 200*400 time series
print('\n----------   Begin generating time series data ----------')
print('\n--   You need to wait about 60 seconds to get a data.pickle file ')
time1 = time.time()
for j in range(200):
    ab = np.random.random(2)
    # a is the strength of activation
    a = ab[0]
    # b is the strength of inhibition 
    b = ab[1]
    # for each set of parameters a and b, produce 400 sequences
    for i in range(400):
        ab_data[j*400+i, 0] = a
        ab_data[j*400+i, 1] = b
        P0 = init_node()
        P = P0
        xx1 = np.zeros([NN])
        xx2 = np.zeros([NN])
        for v in range(NN):
            P = move(P, tt, np.array([a, b, 0.5, 4.0, 1.0]))
            xx1[v] = P[0]
            xx2[v] = P[1]
        x1.append(xx1)
        x2.append(xx2)

# reshape the sequences into m*2*n*1: 
# m is the number of sequences, 2 is the dim, n is the length of each sequence
datax1 = np.array(x1)
m, n = datax1.shape
datax2 = np.array(x2)
youtcome = np.zeros([m, 2, n, 1])
for i in range(m):
    youtcome[i, 0, :, :] = datax1[i, :].reshape(n, 1)
    youtcome[i, 1, :, :] = datax2[i, :].reshape(n, 1)

# Divide the data into three parts:
# train 80%, validation 10%, test 10%
index = [o for o in range(youtcome.shape[0])]
np.random.shuffle(index)
ynew = youtcome[index, :, :, :]
ab_newdata = ab_data[index, :]

train_data = ynew[: youtcome.shape[0] // 10 * 8, :, :, :]
val_data = ynew[youtcome.shape[0] // 10 * 8:youtcome.shape[0] // 10 * 9, :, :, :]
test_data = ynew[youtcome.shape[0] // 10 * 9:, :, :, :]
train_ab = ab_newdata[: youtcome.shape[0] // 10 * 8, :]
val_ab = ab_newdata[youtcome.shape[0] // 10 * 8:youtcome.shape[0] // 10 * 9, :]
test_ab = ab_newdata[youtcome.shape[0] // 10 * 9:, :]

train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data)
test_data = torch.tensor(test_data)
train_ab = torch.tensor(train_ab)
val_ab = torch.tensor(val_ab)
test_ab = torch.tensor(test_ab)

print('\n Train data size:', train_data.shape)
print('\n Val data size:' , val_data.shape)
print('\n Test data size:' , test_data.shape)
print('\n Train ab size:' , train_ab.shape)
print('\n Val ab size:' , val_ab.shape)
print('\n Test ab size:' , test_ab.shape)

time2 = time.time()
print('\n----------   Finsh generating time series data ----------')
results = [train_data, val_data, test_data, train_ab, val_ab, test_ab]
with open('data.pickle', 'wb') as f:
    pickle.dump(results, f)

print('\n--The code finishes running, and cost %d seconds' % (time2-time1))
