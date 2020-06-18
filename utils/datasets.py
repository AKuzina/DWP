import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from scipy.io import loadmat


def load_mnist():
    # set args
    train_dset = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
    test_dset = datasets.MNIST('data', train=False,
                               transform=transforms.Compose([transforms.ToTensor()]))
    # preparing data
    x_train = train_dset.data.float().numpy() / 255.
    x_train = np.reshape(x_train, (-1, 28 * 28))

    x_test = test_dset.data.float().numpy() / 255.
    x_test = np.reshape(x_test, (-1, 28 * 28))

    # pytorch dataset
    train_dset = TensorDataset(torch.from_numpy(x_train).float(), train_dset.targets)
    test_dset = TensorDataset(torch.from_numpy(x_test).float(), test_dset.targets)
    return train_dset, test_dset


def load_notmnist():
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
    X = X.reshape(X.shape[0], -1)
    X = np.array(X, dtype=np.float32)
    X /= 255.0
    idx = np.std(X, axis=1) >= 0.08
    X = X[idx]
    Y = Y[idx]

    seed = 0
    np.random.seed(seed)
    N_train = int(X.shape[0] * 0.9)
    ind = np.random.permutation(range(X.shape[0]))
    x_train = X[ind[:N_train]]
    y_train = Y[ind[:N_train]]
    x_test = X[ind[N_train:]]
    y_test = Y[ind[N_train:]]
    # pytorch dataset
    train_dset = TensorDataset(torch.from_numpy(x_train).float(),
                               torch.from_numpy(y_train))
    test_dset = TensorDataset(torch.from_numpy(x_test).float(),
                              torch.from_numpy(y_test))
    return train_dset, test_dset


