import math

import torch
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler

import Lars_optimzer
import loading_data
from vicreg_architecture import VICReg
from visulalize import visualize
from torch.utils.data import DataLoader


def exclude_bias_and_norm(p):
    return p.ndim == 1


def main():
    # Load the dataset this is lite version of imagenet.
    train_dataset = loading_data.get_data()

    # optional - visualize the data, every couple of pictures that was augmented by the same original picture.
    # visualize(train_dataset)

    # train the model
    train(train_dataset, 50)


def default_args():
    args = {}
    args["arch"] = "resnet50"
    args["mlp"] = "8192-8192-8192"
    args["sim_coeff"] = 25.0
    args["std_coeff"] = 25.0
    args["cov_coeff"] = 1.0
    args["batch_size"] = 128
    args["wd"] = 1e-6
    args["lr"] = 0.2
    args["base_lr"] = 0.2
    args["epochs"] = 100
    return args


def train(training_dataset):
    # create the model default args.
    args = default_args()
    model = VICReg(args)
    lars = Lars_optimzer.LARS(
        model.parameters(), lr=args['lr'], weight_decay=args['wd'], weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm
    )
    # Initialize GradScaler
    scaler = GradScaler()
    # Create a DataLoader to handle batching
    train_loader = DataLoader(training_dataset, batch_size=args["batch_size"], shuffle=True)
    # Create a learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        lars,  # Optimizer
        T_0=10,  # Number of iterations until the first restart
        T_mult=2,  # Factor by which T_0 is multiplied after each restart
        eta_min=0.001  # Minimum learning rate
    )
    # train the model
    for epoch in range(args['epochs']):
        for i, batch in enumerate(train_loader):
            scheduler.step(epoch + i / len(train_loader))
            # zero the parameter gradients
            lars.zero_grad()
            # forward + backward + optimize
            with torch.cuda.amp.autocast():
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(lars)
            scaler.update()
            # print statistics
            print(f'epoch: {epoch}, batch: {i}, loss: {loss.item()}')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args['epochs'] * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args['base_lr'] * args['batch_size'] / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    main()
