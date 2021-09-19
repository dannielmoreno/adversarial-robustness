import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# THIS VARIABLE CAN TAKE "train", "plot_error", "best_model", "examples" AND "autoattack" AS VALUES. DURING THE
# EVALUATION OF YOUR CODE, IT WILL BE EXECUTED 5 TIMES (ONCE WITH EACH POSSIBLE VALE OF mode). EXECUTION WILL
# BE GRADED TAKING THE CAPACITY OF YOUR CODE TO BE EXECUTED UNDER THIS 5 POSIBILITIES

mode = "examples"

# Loads MNIST train and test datasets, and their corresponding dataloaders

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 1000, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 1000, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

# Creates a two instances of equal simple CNN architectures that would be trained under standard conditions and with
# adversarial examples.

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

model_standard_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)

model_robust_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)

# Construct FGSM adversarial examples on the examples X (IMPORTANT: It returns the perturbation (i.e. delta))
def fgsm(model, X, y, epsilon=0.1):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return X + epsilon * delta.grad.detach().sign()

# Construct PGD in L_inf ball adversarial examples on the examples X (IMPORTANT: It returns the perturbation (i.e. delta))

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):

    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

# Standard training/evaluation epoch over the dataset

def epoch(loader, model, opt=None):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Adversarial training/evaluation epoch over the dataset

def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


# Performs standard training over an architecture and evaluates it with the tests set and with adversarial examples
# generated from the test set.

def main_standard(model, train_loader, test_loader, attack, lr, epochs, save_path_model='model_standard.pt',
                  save_path_error='model_standard_error.npy'):
    opt = optim.SGD(model.parameters(), lr=lr)
    train_err_list, test_err_list, adv_err_list = [], [], []
    for t in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
        test_err, test_loss = epoch(test_loader, model)
        adv_err, adv_loss = epoch_adversarial(test_loader, model, attack)
        print("Epoch "+ str(t+1) + "/" +str(epochs) + ": " ,*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")
        train_err_list.append(train_err)
        test_err_list.append(test_err)
        adv_err_list.append(adv_err)
    if save_path_model != None:
        torch.save(model.state_dict(), save_path_model)
    if save_path_error != None:
        np.save(save_path_error, np.array([list(range(1, epochs+1)), train_err_list, test_err_list, adv_err_list]))

if mode == 'train':
    print("\nStandard Model")
    main_standard(model_standard_cnn, train_loader, test_loader, attack=pgd_linf, lr=1e-1, epochs=10)

"""
TODO: 4. Based on the main_standard() function, create a main_adversarial() function that performs adversarial
training over an architecture and evaluates it with the tests set and with adversarial examples generated from the
test set. Hint: You just have to change one line of code. Implement this code to train a model with adversarial training.
"""

def main_adversarial(model, train_loader, test_loader, attack, lr, epochs, save_path_model='model_adversarial.pt',
                  save_path_error='model_adversarial_error.npy'):
    return None

if mode == 'train':
    print("\nRobust Model")
    main_adversarial(model_robust_cnn, train_loader, test_loader, attack=pgd_linf, lr=1e-1, epochs=10)

"""
TODO: 5. Generate a 1x2 subplot in which you visualize the train, test and adversarial error as a function of the number
of training epochs for a) the model with standard training, b) the model with adversarial training. When you run the
code with mode="plot_error", this figure should be shown and saved with the path "errors.png".
"""

if mode == "plot_error":
    print("Plot your errors here and save the figure as errors.png ")

"""
TODO: 6. Experiment by training various models with adversarial examples. Compare their errors when varying the
 following conditions:
      a) The type of attack and its parameters (fgsm and pgd - epsilon, alpha, number of iterations, etc.)
      b) PGD with zero and random initialization.
      c) The model architecture.
      d) The training epochs.
You should only report your experiments and results in the article. However, when you run the code with mode="best_model",
your best model should be trained and its parameters should be saved as 'best_model.pt'.
"""

if mode == "best_model":
    print("Plot your errors here and save the figure as errors.png ")

"""
TODO: 7. Visualize 3 adversarial examples that your baseline standard model misclassifies and your model with the lowest
adversarial error classifies correctly as a 1x3 subplot. Indicate the classes predicted by each model among with its 
probability. When you run the code with mode="examples", this figure should be shown and it should be saved with the
path examples.png
"""

if mode == "examples":
    print("Visualize your examples and save the figure as examples.png ")

"""
BONUS:
Run Autoattack on your best model. Use norm='Linf' and eps=0.3 as parameters. This should be executed when you run
the code with mode = "autoattack"
"""

if mode == "autoattack":
    print("Run Autoattack on your best model.")




