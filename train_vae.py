import argparse
import torch
import utils
import models
import os
import pytorch_lightning as pl

parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='./runs', help='root folder')
parser.add_argument('--dataset_name', type=str, default='notMNIST', help='name of the dataset')
parser.add_argument('--resume', action='store_true', default=False, help='Continue training')


# arguments for optimization
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 5)')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
parser.add_argument('--patience', type=float, default=15, help='Patients for lr scheduler')
parser.add_argument('--warmup', type=int, default=1, help='')
parser.add_argument('--f', type=int, default=32, help='Floating point precision')
# parser.add_argument('--anneal', type=float, default=1e-9, help='Patients for lr scheduler')

# cuda
parser.add_argument('--device', type=str, default='cuda:0', help='enables CUDA training')

# kernel
parser.add_argument('--kernel_size', type=int, default=5, help='')
parser.add_argument('--kernel_dimention', type=int, default=2, help='2D or 3D kernels')
parser.add_argument('--dim_latent', type=int, default=2, help='')
parser.add_argument('--norm_thr', type=float, default=5e-1, help='')



# experiment
# parser.add_argument('--iter', type=int, default=0, help='Train/test split iteration')


def main(args):
    if args.kernel_dimention == 2:
        if args.kernel_size == 7:
            args.norm_thr = 1e-1
            args.dim_latent = 2
            max_ep = 300
        elif args.kernel_size == 5:
            args.norm_thr = 5e-2
            args.dim_latent = 4
            max_ep = 50


        mod = models.kernel_vae.KernelVAE(args)
        args.model_name = '{}/prior/kernel_{}'.format(args.dataset_name, args.kernel_size)


    elif args.kernel_dimention == 3:
        mod = models.kernel_vae.KernelVAE3D(args)



    print('Model name:', args.model_name)

    early_stop_callback = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True
    )

    trainer = pl.Trainer(gpus=1, show_progress_bar=True,
                         default_root_dir=os.path.join('runs', args.model_name),
                         early_stop_callback=early_stop_callback,
                         precision=args.f, terminate_on_nan=True,
                         checkpoint_callback=checkpoint_callback, max_epochs=max_ep)
    trainer.fit(mod)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)