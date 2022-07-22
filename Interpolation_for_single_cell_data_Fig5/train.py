import torch
import numpy as np
from torch import nn
import pickle
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import time

# train DNN for different hyperparameter k, you can change the value of k in line 186
# and this file we set k=0.15 default, after train you can get a Parameters_saved_k015.file
# for model 4 in our work, it is a ensemble model for k=0.05,0.15,0.25 and 0.35
def load_data(batch_size=256):
    data_path = 'data.pickle'
    with open(data_path, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    print('\n Train data size:', train_data.shape)
    print('\n Val data size:' , val_data.shape)
    print('\n Test data size:' , test_data.shape)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_dynamics(args, dynamics_learner,
                   optimizer, device, train_loader, use_cuda, adj):

    # dynamics_learner is the DNN model
    loss_records = []
    mse_records = []
    out_loss_record = []

    # train sub_epochs times before every validation
    for step in range(1, args.sub_epochs + 1):
        loss_record = []
        mse_record = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            loss, mse = train_dynamics_learner(optimizer, dynamics_learner, data, args.prediction_steps, use_cuda,
                                               device, adj)
            loss_record.append(loss.item())
            mse_record.append(mse.item())
        loss_records.append(np.mean(loss_record))
        mse_records.append(np.mean(mse_record))
        print('\nTraining %d/%d before validation, loss: %f, MSE: %f' % (step,args.sub_epochs, np.mean(loss_record), np.mean(mse_record)))
        
        out_loss_record.append(np.mean(loss_record))
    return out_loss_record


def val_dynamics(args, dynamics_learner, device, val_loader, best_val_loss, use_cuda, adj):

    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, data, args.prediction_steps, use_cuda, device, adj)
        loss_record.append(loss.item())
        mse_record.append(mse.item())

    print('\nValidation: loss: %f, MSE: %f' % (np.mean(loss_record), np.mean(mse_record)))
    
    if best_val_loss > np.mean(loss_record):
        torch.save(dynamics_learner.state_dict(), args.dynamics_path)

    return np.mean(loss_record)


def train_dynamics_learner(optimizer, dynamics_learner, data, steps, use_cuda, device, adj):
    optimizer.zero_grad()

    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]
    output = input1

    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    # Make a prediction with steps-1，output：batchsize, num_nodes, time_steps, dimension
    for t in range(steps - 1):
        output1 = torch.as_tensor(output, dtype=torch.float32).view(-1, data.size(1)).to(device)
        output3 = dynamics_learner(output1)
        output = output1 + output3 * 0.5 - output1 * 0.5 * adj
        out11 = output.view(-1, data.size(1), 1)
        outputs[:, :, t, :] = out11

    loss = torch.mean(torch.abs(outputs - target))
    loss.backward()
    optimizer.step()
    mse = F.mse_loss(outputs, target)
    if use_cuda:
        loss = loss.cpu()
        mse = mse.cpu()

    return loss, mse


def val_dynamics_learner(dynamics_learner, data, steps, use_cuda, device, adj):

    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]
    output = input1

    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    for t in range(steps - 1):
        output1 = torch.as_tensor(output, dtype=torch.float32).view(-1, data.size(1)).to(device)
        output3 = dynamics_learner(output1)

        output = output1 + output3 * 0.5 - output1 * 0.5 * adj
        out11 = output.view(-1, data.size(1), 1)
        outputs[:, :, t, :] = out11

    loss = torch.mean(torch.abs(outputs - target))
    mse = F.mse_loss(outputs, target)

    return loss, mse


def test(args, dynamics_learner, device, test_loader, use_cuda, adj):
    # load model
    dynamics_learner.load_state_dict(torch.load(args.dynamics_path))
    dynamics_learner.eval()
    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, data, args.prediction_steps, use_cuda, device, adj)
        loss_record.append(loss.item())
        mse_record.append(mse.item())
    print('loss: %f, mse: %f' % (np.mean(loss_record), np.mean(mse_record)))


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


def main():

    # Training settings

    parser = argparse.ArgumentParser(description='DNN_FOR_NETWORK_REFERENCE')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1998,
                        help='random seed (default: 1998)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs, default: 50)')
    parser.add_argument('--sub-epochs', type=int, default=10,
                        help='i.e. train 10 times before every Validation (default: 10)')
    parser.add_argument('--prediction-steps', type=int, default=13,
                        help='prediction steps in data (default: 13)')
    parser.add_argument('--dynamics-path', type=str, default='Parameters_saved_k015.pickle',
                        help='path to save dynamics learner (default: ./saved/Parameters_saved_k015.pickle)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    torch.manual_seed(args.seed)
    adj = torch.ones([1, 48])
    # the value of hyperparameter k we set for this specific DNN model
    mkm = 0.15
    adj = adj * mkm
    adj = adj.to(device)

    # Loading data
    print('\n----------   Loading data ----------')
    train_loader, val_loader, test_loader = load_data(batch_size=256)
    print('\n----------   loading data is finished ----------')

    # move network to gpu
    dynamics_learner = FullyConnected().to(device)
    # Adam optimizer and the learning rate is 1e-4
    optimizer = torch.optim.Adam(dynamics_learner.parameters(), lr=0.0001)

    # Initialize the best validation error and corresponding epoch
    best_val_loss = np.inf
    best_epoch = 0

    loss_out = []
    print('\n----------   Parameters of each layer  ----------')
    for name, parameters in dynamics_learner.named_parameters():
        print(name,":",parameters.shape)

    print('\n----------   begin training  ----------')
    print('\n--   You need to wait about 10 minutes for each epoch ')    

    for epoch in range(1, args.epochs + 1):
        time1 = time.time()
        print(device)
        print('\n----------   Epoch %d/%d ----------' % (epoch,args.epochs))
        out_loss = train_dynamics(args, dynamics_learner, optimizer, device, train_loader,
                                  use_cuda, adj)
        val_loss = val_dynamics(args, dynamics_learner, device, val_loader,
                                best_val_loss, use_cuda, adj)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        print('\nCurrent best epoch: %d, best val loss: %f' % (best_epoch, best_val_loss))

        loss_out.append(out_loss)
        time2 =time.time()
        print("it spends %d seconds for trainsing this epoch" % (time2-time1))

    print('\nBest epoch: %d' % best_epoch)

    test(args, dynamics_learner, device, test_loader, use_cuda, adj)

    # loss_address = 'loss.pickle'
    # with open(loss_address, 'wb') as f:
    #     pickle.dump(loss_out, f)
    print('\n-----The code finishes running' )

if __name__ == '__main__':
    main()

