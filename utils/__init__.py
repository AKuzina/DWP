import os
import numpy as np
from sklearn.model_selection import train_test_split

from utils.datasets import *
from utils.mri_datasets import *


def load_dataset(args, bootstrap=False):
    root = os.path.join(args.root, args.dataset_name)

    if 'notMNIST' in args.dataset_name:
        train_dset, test_dset = load_notmnist(args)
        print(len(train_dset))
        args.task = 'clf'
        args.n_classes = 10

    elif 'MNIST' in args.dataset_name:
        train_dset, test_dset = load_mnist(args)
        print(len(train_dset))
        args.task = 'clf'
        args.n_classes = 10


    # elif 'BRATS18' in args.dataset_name:
    #     dset = BRATS18(root=root, im_types=['{}.'.format(args.data_type)],
    #                    proper_shape=np.array([152, 184, 144]))
    #     args.task = 'seg'
    #     args.n_classes = 2
    # elif 'MS' in args.dataset_name:
    #     root = os.path.join(args.root, 'MS_Dataset_full/MS_nii')
    #     dset = MS(root=root, proper_shape=np.array([24, 352, 400]))
    #     args.task = 'seg'
    #     args.n_classes = 2
    #
    # if args.train_size > 0:
    #     all_patients = dset.all_patients
    #     if 'clf' in args.task:
    #         strat = np.array([x.item() for x in dset.y])
    #     else:
    #         strat = None
    #     tr_idx, te_idx = train_test_split(np.arange(0, len(all_patients)),
    #                                       test_size=args.test_size,
    #                                       random_state=42 + args.iter, stratify=strat)
    #     assert args.test_size + args.train_size <= len(all_patients), \
    #         'test and train size is more than {} (number of images in dataset)'.format(
    #             len(all_patients))
    #     np.random.seed(42 + args.iter)
    #     if bootstrap:
    #         tr_idx = tr_idx[
    #             np.random.choice(len(tr_idx), size=args.train_size, replace=True)]
    #     else:
    #         tr_idx = tr_idx[
    #             np.random.permutation(np.arange(0, len(tr_idx)))[:args.train_size]]
    #
    #     if 'clf' in args.task:
    #         print('Train:', np.array(dset.y)[tr_idx])
    #         print('Test:', np.array(dset.y)[te_idx])
    #     print(tr_idx, te_idx)
    #     dset.train_patients = tr_idx
    #     dset.test_patients = te_idx
    #
    return train_dset, test_dset, args


def create_model_name(args):
    if not args.resume:
        args.resume = None
    root = '/'.join(args.root.split('/')[:-1])

    if 'BRATS' in str(args.dataset_name):
        args.model_name = '{}/'.format('BRATS')
    else:
        args.model_name = '{}/'.format(str(args.dataset_name))
    if args.pretrain or args.dwp:
        args.model_name += 'from_{}/'.format(args.prior)
        if args.pretrain:
            args.model_name += 'pretrain'
            if args.freeze:
                args.model_name += '_freeze'

            args.resume = os.path.join(root,
                                       'runs/weights/{}/prior/'.format(args.prior))
            if 'seg' in args.task:
                if args.short:
                    args.resume += 'short_unet_{}'.format(args.n_classes)
                else:
                    args.resume += 'full_unet_{}'.format(args.n_classes)

            elif 'clf' in args.task:
                if args.short:
                    args.resume += 'short_clf'
                else:
                    args.resume += 'full_clf'

        else:
            args.model_name += 'dwp'
    else:
        args.model_name += 'no_prior/nb'

    if args.data_type is not None:
        args.model_name += '_{}'.format(args.data_type)
    args.model_name += '_{}'.format(args.train_size)
    if args.short:
        args.model_name += 'SMALL'
    return args