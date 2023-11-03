import torch
import torch.optim as optim
from models import CustomModel
from DataLoader import train_loader, test_loader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

EPOCHS = 10

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.device = DEVICE


def getModel(params, trail):
    """
    Create and configure a deep learning model for a given set of parameters and a Optuna trial.
    :param params: A dictionary containing hyperparameters for configuring the model.
    :param trail: A Optuna trial object used for hyperparameter optimization.

    :return: A tuple containing the deep learning model, loss function, and optimizer.
    """
    model = CustomModel(params, trail)
    model.to(DEVICE)
    loss_function = torch.nn.CrossEntropyLoss()
    if params is not None and params["Optimizer"] == "Adam":
        beta_2 = trail.suggest_loguniform("beta_2", 1e-1, 9e-1)
        beta_1 = trail.suggest_loguniform("beta_1", 1e-2, 9e-2)
        optimizer = optim.Adam(model.parameters(), lr=params["Learning Rate"], betas=(beta_1, beta_2))
    elif params is not None and params["Optimizer"] == "SGD":
        momentun = trail.suggest_loguniform("beta_2", 1e-7, 1e-2)
        weight_decay = trail.suggest_loguniform("beta_2", 1e-7, 1e-2)
        optimizer = optim.SGD(model.parameters(), lr=params["Learning Rate"], momentum=momentun,
                              weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-2)


    return model, loss_function, optimizer


def train_batch(X, y, model, loss_function, optimizer):
    """
    Train a deep learning model on a single batch of data.

    :param X: Input data for the batch.
    :param y: Target labels for the batch.
    :param model: The deep learning model to train.
    :param loss_function: The loss function used to compute the loss.
    :param optimizer: The optimizer responsible for updating the model's parameters.

    :return: A tuple containing the total batch loss and accuracy.
    """
    model.train()
    predicitons = model(X)
    batch_loss = loss_function(predicitons, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item() * X.shape[0], accuray(y, predicitons)


@torch.no_grad()
def Inference(X, y, model, loss_function):
    """
    Perform inference (evaluation) using a trained deep learning model on a batch of data.

    :param X: Input data for the batch.
    :param y: Target labels for the batch.
    :param model: The trained deep learning model.
    :param loss_function: The loss function used to compute the loss.

    :return: A tuple containing the total batch loss and accuracy.
    """
    predictions = model(X)
    loss = loss_function(predictions, y)
    accuracy = torch.sum(torch.argmax(predictions, dim=1) == y)
    return loss.item() * X.shape[0], accuracy.item()


def train(train_loader, test_loader, model, loss_function, optimizer, trail):
    """
    Train a deep learning model over multiple epochs using the provided training and test data loaders.

    :param train_loader: Data loader for training data.
    :param test_loader: Data loader for test data.
    :param model: The deep learning model to train.
    :param loss_function: The loss function used for training and evaluation.
    :param optimizer: The optimizer responsible for updating the model's parameters.
    :param trail: A HyperOpt trial object used for reporting intermediate results.

    :return: Nothing to return. This function logs training and test statistics, and reports accuracy to the HyperOpt trial.
    """

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    with tqdm(total=EPOCHS, desc='Training') as epoch_bar:
        for epoch in range(EPOCHS):
            train_loss_epoch, train_accuracy_epoch = [], []
            test_loss_epoch, test_accuracy_epoch = [], []
            for X, y in train_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                loss, accuray = train_batch(X, y, model, loss_function, optimizer)
                train_loss_epoch.append(loss)
                train_accuracy_epoch.append(accuray)

            for X, y in test_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                loss, accuray = Inference(X, y, model, loss_function)
                test_loss_epoch.append(loss)
                test_accuracy_epoch.append(accuray)

            train_losses.append(np.sum(np.array(train_loss_epoch)) / len(train_loader.dataset))
            train_accuracies.append(np.sum(np.array(train_accuracy_epoch)) / len(train_loader.dataset))
            test_losses.append(np.sum(np.array(test_loss_epoch)) / len(test_loader.dataset))
            test_accuracies.append(np.sum(np.array(test_accuracy_epoch)) / len(test_loader.dataset))

            epoch_bar.set_postfix(
                loss=f'{np.sum(np.array(train_loss_epoch)) / len(train_loader.dataset):.4f}, Test : {np.sum(np.array(test_loss_epoch)) / len(test_loader.dataset)}',
                accuracy=f'{100 * np.sum(np.array(train_accuracy_epoch)) / len(train_loader.dataset):.2f}% , Test : {100 * np.sum(np.array(test_accuracy_epoch)) / len(test_loader.dataset) :.2f}%'
            )
            epoch_bar.set_description(f'Epoch {epoch + 1}')
            epoch_bar.update(1)

            if trail is not None:
                trail.report(np.sum(np.array(train_accuracy_epoch)) / len(train_loader.dataset), epoch)

                # Handle pruning based on the intermediate value.
                if trail.should_prune():
                    raise optuna.exceptions.TrialPruned()

    plot(train_losses, train_accuracies, test_losses, test_accuracies)

    # get the final loss
    loss, accuray = Inference(X, y, model, loss_function)
    return accuray / len(test_loader.dataset)


def accuray(y_true, y_predictions):
    """
    Calculate the accuracy of predicted labels compared to true labels.

    :param y_true: True labels.
    :param y_predictions: Predicted labels.

    :return: The accuracy as a floating-point number.
    """
    final_predicitons = torch.argmax(y_predictions, dim=1)
    total_accuracy = torch.sum(y_true == final_predicitons)
    return total_accuracy.item()


def plot(train_losses, train_accuracies, test_losses, test_accuraies):
    """
    Create and display a plot showing training and test loss, as well as training and test accuracy over epochs.

    :param train_losses: List of training losses for each epoch.
    :param train_accuracies: List of training accuracies for each epoch.
    :param test_losses: List of test losses for each epoch.
    :param test_accuracies: List of test accuracies for each epoch.

    :return: Nothing is returned. The function displays the plot.
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


def main(params=None, trail=None):
    """
    Main function for training and evaluating a deep learning model with specified hyperparameters.

    :param params: A dictionary containing hyperparameters for configuring the model and training process.
    :param trail: A Optuna trial object for reporting and optimization.

    :return: The final accuracy of the trained model on the test data.
    """
    model, loss_function, optimizer = getModel(params, trail)
    return train(train_loader, test_loader, model, loss_function, optimizer, trail)


def objective(trail):
    """
    Define the optimization objective for HyperOpt.

    :param trail: A Optuna trial object for configuring and tracking the optimization process.

    :return: The accuracy of the trained model as a value to be optimized.
    """
    params = {
        "Number of Linear layers": trail.suggest_int("numberOfLinearLayers", 1, 4),
        # "Number of Convolutional layers": trail.suggest_int("numberOfConvolutionLayers", 1, 3),
        "Learning Rate": trail.suggest_loguniform("learningRate", 1e-6, 1e-2),
        "Optimizer": trail.suggest_categorical("optimizer", ["Adam", "SGD"]),
    }
    accuracy = main(params, trail)
    return accuracy


if __name__ == "__main__":
    Optuna = False
    if Optuna == True:
        study = optuna.create_study(direction="maximize", storage="sqlite:///mnsit.db", study_name="Final_Run_1",
                                    load_if_exists=True)
        study.optimize(objective, n_trials=100)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        main()
