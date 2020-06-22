import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from scipy.io import loadmat


def load_mnist(args):
    # set args
    tr = args.train_size
    te = args.test_size
    train_dset = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
    test_dset = datasets.MNIST('data', train=False,
                               transform=transforms.Compose([transforms.ToTensor()]))
    # preparing data

    ##
    if tr != -1:
        ind = np.random.permutation(range(len(train_dset)))[:tr]
        train_dset.data = train_dset.data[ind]
        train_dset.targets = train_dset.targets[ind]
    if te != -1:
        ind = np.random.permutation(range(len(test_dset)))[:te]
        test_dset.data = test_dset.data[ind]
        test_dset.targets = test_dset.targets[ind]

    return train_dset, test_dset


def load_notmnist(args):
    # set args
    tr = args.train_size
    te = args.test_size

    link_to_data = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat'
    path_to_data = 'data/notMNIST'
    # start processing
    if not os.path.exists(path_to_data):
        datasets.utils.download_url(link_to_data, root = path_to_data,
                                                filename='notMNIST_small.mat')
    out = loadmat(os.path.join(path_to_data, 'notMNIST_small.mat'))
    X, Y = out['images'], out['labels']
    Y =  np.array(Y, dtype=int)
    X = X.transpose(2, 0, 1)
    # X = X.reshape(X.shape[0], -1)
    X = np.array(X, dtype=np.float32)
    X /= 255.0
    # idx = np.std(X, axis=1) >= 0.08
    # X = X[idx]
    # Y = Y[idx]

    seed = 0
    np.random.seed(seed)
    N_train = int(X.shape[0] * 0.9)
    N_test = -1
    ind = np.random.permutation(range(X.shape[0]))
    if tr != -1:
        N_train = tr
    if te != -1:
        N_test = N_train + te

    x_train = X[ind[:N_train]]
    y_train = Y[ind[:N_train]]
    x_test = X[ind[N_train:N_test]]
    y_test = Y[ind[N_train:N_test]]
    # pytorch dataset
    train_dset = TensorDataset(torch.from_numpy(x_train).float().unsqueeze(1),
                               torch.from_numpy(y_train))
    test_dset = TensorDataset(torch.from_numpy(x_test).float().unsqueeze(1),
                              torch.from_numpy(y_test))
    return train_dset, test_dset


