import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import math


def normalize(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / (tensor.max() + 1e-08)
    return tensor


class FeatureDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        if self.transform:
            x = self.transform(x)
        return x, y


def test_classification_accuracy(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    model.train()


def reconstruct(model, test_batch_size, test_loader, normalize_tensor=False):
    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images_flatten = images.view(images.size(0), -1)
    # get sample outputs
    output = model(images_flatten)
    # prep images for display
    images = images.numpy()

    # output is resized into a batch of images
    output = output.view(test_batch_size, 1, 28, 28)

    # use detach when it's an output that requires_grad
    output = output.data.numpy()
    if normalize_tensor:
        normalize(output)

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


def convert_batch_to_image_grid(image_batch, model_folder, model_num, reshape_size):
    image_batch_norm = normalize(image_batch)
    print("image batch shape: ", image_batch_norm.shape)
    nrows = 8
    nmaps = image_batch.size(0)
    xmaps = min(nrows, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    fig = plt.figure(figsize=(ymaps, xmaps), facecolor='w', edgecolor='k')
    for t in range(nmaps):
        ax1 = fig.add_subplot(xmaps, ymaps, t + 1)
        ax1.imshow(image_batch_norm[t, :].reshape(reshape_size, reshape_size), cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        t += 1
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.axis('off')
    filename = model_folder + '_' + model_num + '.png'
    plt.savefig(filename)
    plt.show()


def plot_weights(decoder_layer_data, model_folder, model_num, reshape_size=28, start_idx=400, end_idx=600, transpose=True):
    if transpose:
        decoder_layer_data = decoder_layer_data.T[start_idx:end_idx]
    else:
        decoder_layer_data = decoder_layer_data[start_idx: end_idx]
    convert_batch_to_image_grid(decoder_layer_data, model_folder, model_num, reshape_size)
