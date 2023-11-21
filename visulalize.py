import matplotlib.pyplot as plt
import data_transform


def visualize(train_loader):
    for batch_data, batch_labels in train_loader:
        # batch_labels is a batch of labels corresponding to the images

        # If you want to do further processing, you can do it here

        # Visualize the data
        plt.imshow(batch_data[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(batch_data[1].permute(1, 2, 0))
        plt.show()

