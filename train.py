import torch
import torch.nn as nn
import torch.optim as optim
from models import CustomModel
from DataLoader import train_loader,test_loader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: Effect of augmentation
# TODO: Effect of regularizaiton drop out, early stopping, l2 norm
# TODO: Different Architectures

EPOCHS = 3


def getModel():
    """
    :return: (model,loss function, optimizer)
    """
    model = CustomModel()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    return model,loss_function,optimizer

def train_batch(X,y,model,loss_function,optimizer):
    """
    :return: (loss , accuracy)
    """
    model.train()
    predicitons = model(X)
    batch_loss = loss_function(predicitons,y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item() * X.shape[0],accuray(y,predicitons)


@torch.no_grad()
def Inference(X,y,model,loss_function):
    predictions = model(X)
    loss = loss_function(predictions,y)
    accuracy = torch.sum(torch.argmax(predictions) == y)
    return loss.item() * X.shape[0],accuracy.item()

def train(train_loader,test_loader,model,loss_function,optimizer):
    """
    :return: (plot nothing to return )
    """
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    with tqdm(total=EPOCHS, desc='Training') as epoch_bar:
        for epoch in range(EPOCHS):
            train_loss_epoch,train_accuracy_epoch = [],[]
            test_loss_epoch, test_accuracy_epoch = [],[]
            for X,y in train_loader:
                loss,accuray = train_batch(X,y,model,loss_function,optimizer)
                train_loss_epoch.append(loss)
                train_accuracy_epoch.append(accuray)


            for X,y in test_loader:
                loss,accuray = Inference(X,y,model,loss_function)
                test_loss_epoch.append(loss)
                test_accuracy_epoch.append(accuray)

            train_losses.append(np.sum(np.array(train_loss_epoch)) / len(train_loader.dataset)  )
            train_accuracies.append(np.sum(np.array(train_accuracy_epoch)) / len(train_loader.dataset) )
            test_losses.append(np.sum(np.array(test_loss_epoch))/ len(test_loader.dataset) )
            test_accuracies.append(np.sum(np.array(test_accuracy_epoch)) / len(test_loader.dataset) )

            epoch_bar.set_postfix(
                loss=f'{np.sum(np.array(train_loss_epoch)) / len(train_loader.dataset):.4f}',
                accuracy=f'{100 * np.sum(np.array(train_accuracy_epoch)) / len(train_loader.dataset):.2f}%'
            )
            epoch_bar.set_description(f'Epoch {epoch + 1}')
            epoch_bar.update(1)
    plot(train_losses,train_accuracies,test_losses,test_accuracies)


def accuray(y_true,y_predictions):
    """
    :return: accuracy
    """
    final_predicitons = torch.argmax(y_predictions,dim = 1)
    total_accuracy = torch.sum(y_true == final_predicitons)
    return total_accuracy.item()

def plot(train_losses,train_accuracies,test_losses,test_accuraies):
    """
    :return:
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='o')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(test_accuraies, label='Test Accuracy', marker='o')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    pass

def saveModel():
    pass

def main():
    model,loss_function,optimizer = getModel()
    train(train_loader,test_loader,model,loss_function,optimizer)


if __name__ == "__main__":
    main()
    pass
