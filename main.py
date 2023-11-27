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
    train(train_dataset)


def default_args():
    args = {}
    args["arch"] = "resnet50"
    args["mlp"] = "8192-8192-8192"
    args["sim_coeff"] = 25.0
    args["std_coeff"] = 25.0
    args["cov_coeff"] = 1.0
    args["batch_size"] = 64
    args["wd"] = 1e-6
    args["base_lr"] = 0.2
    args["epochs"] = 100  # should be 1000
    # args["lr"] = args["base_lr"] * args["batch_size"] / 256
    args["lr"] = 1.6
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
        lars,
        T_0=100,  # Number of iterations until the first restart (10 warmup epochs with batch size 2048)
        T_mult=1,  # Factor by which T_0 is multiplied after each restart (no change)
        eta_min=0.002  # Minimum learning rate
    )
    # Define the number of batches to accumulate gradients over
    accumulation_steps = 4  # You can adjust this value

    # train the model
    for epoch in range(args['epochs']):
        accumulated_loss = 0
        for i, batch in enumerate(train_loader):
            # zero the parameter gradients
            lars.zero_grad()

            # forward + backward + optimize
            with torch.cuda.amp.autocast():
                loss = model(batch)
            loss = loss / accumulation_steps  # Divide the loss by the accumulation steps
            scaler.scale(loss).backward()

            accumulated_loss += loss.item()

            if (i + 1) % accumulation_steps == 0:
                # Update the model parameters after accumulating gradients
                scaler.step(lars)
                scaler.update()

                # Print statistics
                print(f'epoch: {epoch}, batch: {i}, accumulated loss: {accumulated_loss}')

                accumulated_loss = 0  # Reset accumulated_loss

            # Update the learning rate
            scheduler.step()


if __name__ == '__main__':
    main()
