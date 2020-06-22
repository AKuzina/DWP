
import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision


class KernelDataset3D(data.Dataset):
    def __init__(self, root, layer=['0.weight'], norm_thr=1e-2, test_ratio=0, mod='last_model'):
        super().__init__()
        """
        Take all kernels from a given layer of all NNs (with the same architecture, but trained with different initialization)

        root, string --- path to folder with model weights
        layer, list --- names of the layers in weight dictionary, e.g. ['0.weight', '3.weight', '5.weight'] 
        """

        self.root = root
        folders = [os.path.join(root, x, '{}.pth'.format(mod)) for x in os.listdir(root)
                   if 'SMALL' in x]
        all_weights = list(
            map(lambda x: torch.load(x, map_location=lambda storage, loc: storage),
                folders))

        def extract_layer(weight_dict):
            all_kernels = [weight_dict[k] for k in layer]
            all_kernels = torch.cat(
                [x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], -1) for x in
                 all_kernels])
            return all_kernels

        self.kernels = torch.cat(list(map(extract_layer, all_weights)))

        # throw away tensors of small norm
        self.kernels = self.kernels[np.where(
            np.array(list(map(torch.norm, torch.unbind(self.kernels, 0)))) > norm_thr)[0]]

        self.train_mode = True
        N = self.kernels.shape[0]
        self.train_idx = np.random.choice(np.arange(N), size=int((1 - test_ratio) * N),
                                          replace=False)
        self.test_idx = np.delete(np.arange(N), self.train_idx)

    def __getitem__(self, index):
        if self.train_mode:
            ker = self.kernels[self.train_idx[index]]
        else:
            ker = self.kernels[self.test_idx[index]]
        return ker.unsqueeze(0), 0

    def __len__(self):
        if self.train_mode:
            return len(self.train_idx)
        else:
            return len(self.test_idx)


class KernelDataset(data.Dataset):
    def __init__(self, root, layer=['net.net.0.weight', 'net.net.3.weight', 'net.net.5.weight'],
                 norm_thr=1e-2, test_ratio=0):
        super().__init__()
        """
        Take all kernels from a given layer of all NNs (with the same architecture, but trained with different initialization)

        root --- path to folder with model weights
        layer --- name of the layer in weight dictionary, e.g. ['0.weight', '3.weight', '5.weight'] for CIFAR conv net
        """

        self.root = os.path.join(root, 'lightning_logs')
        folders = [os.path.join(self.root, x, 'checkpoints/last.ckpt')
                   for x in os.listdir(self.root) if 'version' in x]
        # folders = [os.path.join(root, x, 'last.ckpt') for x in os.listdir(root)]
        all_weights = list(map(lambda x: torch.load(x)['state_dict'], folders))

        def extract_layer(weight_dict):
            all_kernels = [weight_dict[k] for k in layer]
            all_kernels = [x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) for x in
                           all_kernels]
            all_labels = torch.cat([torch.zeros(all_kernels[i].shape[0] ) +i for i in range(len(all_kernels))])
            all_kernels = torch.cat(all_kernels)

            return all_kernels, all_labels

        self.tot = list(map(extract_layer, all_weights))
        self.kernels = torch.cat([self.tot[i][0] for i in range(len(self.tot))])
        self.labels = torch.cat([self.tot[i][1] for i in range(len(self.tot))])
        # print(self.kernels.shape, self.labels.shape)
        # throw away tensors of small norm
        mask = np.where(np.array(list(map(torch.norm,
                                          torch.unbind(self.kernels, 0)))) > norm_thr)[0]
        self.kernels = self.kernels[mask]
        self.labels = self.labels[mask]

        self.train_mode = True
        N = self.kernels.shape[0]
        self.train_idx = np.random.choice(np.arange(N), size=int(( 1-test_ratio )*N),
                                          replace=False)
        self.test_idx = np.delete(np.arange(N), self.train_idx)

    def __getitem__(self, index):
        if self.train_mode:
            idx = self.train_idx[index]
        else:
            idx = self.test_idx[index]
        return self.kernels[idx].unsqueeze(0), self.labels[idx]

    def __len__(self):
        if self.train_mode:
            return len(self.train_idx)
        else:
            return len(self.test_idx)

    def curr_idx(self):
        if self.train_mode:
            return self.train_idx
        else:
            return self.test_idx
