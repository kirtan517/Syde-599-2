import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Setup transforms
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Downloading the dataset
train_dataset = torchvision.datasets.MNIST(root="Data/", train=True,
                                           download=True, transform=train_transform)

test_dataset = torchvision.datasets.MNIST(root="Data/", train=False,
                                          download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


def plotTensorImages(images, labels):
    """Input images are Tensors """
    fig, axarr = plt.subplots(2, 5)

    # Loop through the image paths and display them in the grid
    for i in range(2):
        for j in range(5):
            index = i * 5 + j
            if index < len(images):
                axarr[i, j].imshow(images[index].permute(1, 2, 0))
                axarr[i, j].set_title(f"Label : {labels[index]}")
                axarr[i, j].axis('off')

    fig.suptitle('All the Images')
    plt.tight_layout()
    plt.show()


def ComputeMeanAndStd(train_dataset):
    """Compute mean and std"""
    mean = torch.stack([image for image, _ in train_dataset]).mean()
    std = torch.stack([image for image, _ in train_dataset]).std()

    return mean, std


def plotTenDigits(dataset):
    """Plot 10 images"""
    images = []
    labels = []
    index = 0
    for i in dataset:
        if (index == i[1]):
            images.append(i[0])
            labels.append(i[1])
            index += 1
        if (index > 9):
            break
    plotTensorImages(images, labels)


def plotImage(image):
    """plot single image """
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Turn off the axes
    save_path = 'mnist_image.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.show()


if __name__ == "__main__":
    images, labels = zip(*[train_dataset[i] for i in range(10)])
    plotTenDigits(train_dataset)
    print(next(iter(train_loader))[1])
    plotImage(images[0].permute(1, 2, 0))
