import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#Setup transforms
#TODO: create two transforms one for train and another for test
custom_transform = transforms.Compose([
    transforms.ToTensor(),
])

# TODO: set the transform function in collate function property in trainLoader


# Downloading the dataset
train_dataset = torchvision.datasets.MNIST(root = "Data/",train = True,
                                        download=True,transform=custom_transform)

test_dataset = torchvision.datasets.MNIST(root = "Data/",train = False,
                                          download=True,transform=custom_transform)

#TODO: implement train and test loader
train_loader = DataLoader(train_dataset,batch_size=1024,shuffle=True,)
test_loader = DataLoader(test_dataset)


def plotTensorImages(images,labels):
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
    mean = 0
    std = 0



def plotTenDigits(dataset):
    images = []
    labels = []
    index = 0
    for i in dataset:
        if(index == i[1]):
            images.append(i[0])
            labels.append(i[1])
            index+=1
        if(index > 9):
            break
    plotTensorImages(images,labels)


if __name__ == "__main__":
    # images,labels = zip(*[train_dataset[i] for i in range(10)])
    # plotTenDigits(train_dataset)
    # print(next(iter(train_loader))[1])
    # k = 0
    # for X,y in train_loader:
    #     k+=1
    #     if(k > 10):
    #         break
    #     print(y)


    pass