import torch
import torch.nn as nn
import torch.optim as optim
from models import CustomModel
from DataLoader import train_loader,test_loader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna

# TODO: Effect of augmentation
# TODO: Effect of regularizaiton drop out, early stopping, l2 norm
# TODO: Different Architectures

EPOCHS = 3
if  torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    x = torch.ones(1, device=DEVICE)
else:
    DEVICE = torch.device("cpu")

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.device = DEVICE


def getModel(params,trail):
    """
    :return: (model,loss function, optimizer)
    """
    model = CustomModel()
    model.to(DEVICE)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=params["Learning Rate"])
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
    accuracy = torch.sum(torch.argmax(predictions,dim = 1) == y)
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
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                loss,accuray = train_batch(X,y,model,loss_function,optimizer)
                train_loss_epoch.append(loss)
                train_accuracy_epoch.append(accuray)


            for X,y in test_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                loss,accuray = Inference(X,y,model,loss_function)
                test_loss_epoch.append(loss)
                test_accuracy_epoch.append(accuray)

            train_losses.append(np.sum(np.array(train_loss_epoch)) / len(train_loader.dataset)  )
            train_accuracies.append(np.sum(np.array(train_accuracy_epoch)) / len(train_loader.dataset) )
            test_losses.append(np.sum(np.array(test_loss_epoch))/ len(test_loader.dataset) )
            test_accuracies.append(np.sum(np.array(test_accuracy_epoch)) / len(test_loader.dataset) )

            epoch_bar.set_postfix(
                loss=f'{np.sum(np.array(train_loss_epoch)) / len(train_loader.dataset):.4f}, Test : {np.sum(np.array(test_loss_epoch))/ len(test_loader.dataset)}',
                accuracy=f'{100 * np.sum(np.array(train_accuracy_epoch)) / len(train_loader.dataset):.2f}% , Test : {100 * np.sum(np.array(test_accuracy_epoch)) / len(test_loader.dataset) :.2f}%'
            )
            epoch_bar.set_description(f'Epoch {epoch + 1}')
            epoch_bar.update(1)
    plot(train_losses,train_accuracies,test_losses,test_accuracies)

    # get the final loss
    loss,accuray = Inference(X,y,model,loss_function)
    return loss



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

def main(params,trail):
    model,loss_function,optimizer = getModel(params,trail)
    return train(train_loader,test_loader,model,loss_function,optimizer)


def objective(trail):
    params = {
        # "Number of Convolution layers" : trail.suggest_int("numberOfConvolutionLayers",2,5),
        # "Number of Linear layers" : trail.suggest_int("numberOfConvolutionLayers",1,3),
        "Learning Rate" : trail.suggest_loguniform("learningRate",1e-8,1e-2),
        # "Optimizer" : trail.suggest_categorical("optimizer", ["Adam", "SGD"]),
    }
    loss = main(params,trail)
    return loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=3)
    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))
    # print(DEVICE)
    # main()
