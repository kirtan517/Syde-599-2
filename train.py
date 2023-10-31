import torch
import torch.nn as nn
import torch.optim as optim
from models import CustomModel

# TODO: Effect of augmentation
# TODO: Effect of regularizaiton drop out, early stopping, l2 norm
# TODO: Different Architectures


def getModel():
    """
    :return: (model,loss function, optimizer)
    """
    model = CustomModel("config.json")
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    return model,loss_function,optimizer

def train_Batch(X,y,model,loss_function,optimizer):
    """
    :return: (loss , accuracy)
    """
    model.train()
    predicitons = model(X)
    loss = loss_function(predicitons,y)
    



    pass

def train():
    """
    :return: (plot nothing to return )
    """
    pass

def loss():
    """
    :return: loss
    """
    pass

def accuray():
    """
    :return: accuracy
    """
    pass

def plot():
    """
    :return:
    """
    pass

def saveModel():
    pass

def main():
    pass

if __name__ == "__main__":
    pass
