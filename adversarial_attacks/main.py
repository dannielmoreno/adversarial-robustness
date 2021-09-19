
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet50
import json
import torch.optim as optim
import numpy as np

"""
TODO: 0. Import an image using the Image.open function and save it as a variable with the name example_img.
(Change None for the image path)
"""
example_img = None

# Defines a preprocess that resizes the image to the default input size of a ResNet50 and turns it into a PyTorch Tensor.
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# Applies preprocessing to the example image
example_tensor = preprocess(example_img)

# Saves what would be considered as the original image received as input by the neural network. It should be a resized
# version of the image saved as example_img

plt.imsave("original.jpg", example_tensor.numpy().transpose(1,2,0))

# Defines a class that implements normalization as a PyTorch layer in order
# to allow to directly feed the original image first.

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None] / self.std.type_as(x)[None,:,None, None])

# Creates an instance of the normalization layer based on  the normalization values
# fro ImageNet dataset

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Loads a pre-trained ResNet50 and puts it in evaluation mode
model = resnet50(pretrained=True)
model.eval()

# Forms predictions. pred now contains a 1000 dimensional vector containing the class
# logits for the 1000 imagenet classes

example_pred = model(norm(example_tensor))

# Creates a dictionary that maps the class that maps the logit index position to the name of the imagenet class

with open("imagenet_class_index.json") as f:
    imagenet_classes = {int(i): x[1] for i, x in json.load(f).items()}

# This function returns the class index of the highest probability class when receiving a vector of logits. Although
# this is trivial, when the argument print_stats = True, it returns a description of the (num_classes) classes with the
# higher probabilities.

def most_probable_classes(pred, num_classes, imagenet_classes, print_stats=False):

    logits_sorted, indices = torch.sort(pred, descending=True)
    prob_sorted = nn.Softmax(dim=1)(logits_sorted)

    logits_sorted = torch.squeeze(logits_sorted).detach().numpy()
    indices = torch.squeeze(indices).detach().numpy()
    prob_sorted = torch.squeeze(prob_sorted).detach().numpy()

    pred_dict = {"Class index": indices,
                 "Class":[imagenet_classes[i] for i in indices],
                 "Logit": logits_sorted,
                 "Probability": prob_sorted}

    pred_df = pd.DataFrame(pred_dict)
    if print_stats:
        print(pred_df.head(num_classes))
    return indices[0]

print("\nMost probable classes for original image:")
example_class = most_probable_classes(example_pred, 5, imagenet_classes, print_stats=True)


"""
TODO: 1. Tune the epsilon, alpha and num_iter parameters in order to achieve an adversarial example that induces a
misclassification of your original image. REMEMBER TO COMPLETE THE MISSING STEPS IN THE untargeted_attack FUNCTION.
"""

epsilon = None
alpha = None
num_iter = None

# The following function should produce an untargetted adversarial in al L_inf ball defined by epsilon through PGD. It
# receives as arguments the tensor associated to the input original image, the radius of the Lp-ball (epsilon), the
# "learning rate" of the stochastic gradient descent (alpha), and the number of iterations. It returns the tensor of the
# generated adversarial example.

def untargeted_attack(example_tensor, epsilon, alpha, num_iter):

    # Defines a tensor of the same shape as the example tensor, which will allocate the perturbation associated to the
    # adversarial attack
    delta = torch.zeros_like(example_tensor, requires_grad=True)
    # Creates an instance of an stochastic gradient descend optimizer that will optimize the values of delta with a
    # learning rate defined by parameter.
    opt = optim.SGD([delta], lr=alpha)
    # Obtains the original classification
    example_pred = model(norm(example_tensor))
    example_class = most_probable_classes(example_pred, 1, imagenet_classes)

    print("\nUntargeted PDG Attack")

    for t in range(num_iter):
        # Calculates the logits returned by the model when the original image plus the perturbation are received as inputs
        pred = model(norm(example_tensor + delta))
        # Calculates the Cross Entropy Loss of the logits with respect to the original class
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([example_class]))
        if t % 5 == 0:
            print("Iterations: " + str(t) + " - Loss: " + str(loss.item()))
        # Restarts the gradients of the loss function with respect to delta and iterates the optimization
        opt.zero_grad()
        loss.backward()
        opt.step()

        """
        CHANGE THE None's IN ORDER TO SATISFY THE FOLLOWING STEPS:

        1. Project the perturbation delta to a L_inf ball of radius epsilon centered in 0
        2. Calculate the resulting adversarial example (changed_tensor) as the sum of the original image and the perturbation.
        3. Guarantee that the resulting adversarial example (changed_tensor) lies between the same intensity range
        of the original image. Hint: Check the intensity range of the example_tensor
        """
        delta.data.clamp_(None, None)
        changed_tensor = None
        changed_tensor.data.clamp_(None, None)

    print("True class probability:", nn.Softmax(dim=1)(pred)[0, example_class].item())
    return changed_tensor, delta

uattack_tensor, uattack_delta = untargeted_attack(example_tensor, epsilon, alpha, num_iter)
uattack_pred = model(norm(uattack_tensor))
uattack_class = most_probable_classes(uattack_pred, 5, imagenet_classes, print_stats=True)

"""
TODO: 2. Write a function that allows you to visualize your original image, a magnification of the perturbation, and the
resulting image of the previous adversarial attack in a 1x3 subplot. The title of each subplot must be the most probable
class of the image according to the model among with its probability. (Similar to the Panda Example of the presentation)
"""

def visualize_attack(example_tensor, delta, changed_tensor, model, imagenet_classes, save_path=None):
    return None

visualize_attack(example_tensor, uattack_delta, uattack_tensor, model, imagenet_classes, save_path="uattack.png")

"""
TODO: 3. Based on the untargeted_attack() function written previously, create an adapted function for a targeted attack.
This function will also receive as a parameter the index of one of the 1000 ImageNet classes. Implement this function and
generate a successful targeted attack to a class of your interest which is not among the top 3 most probable classes
according to the initial inference. Visualize this attack.
"""

def targeted_attack(example_tensor, epsilon, alpha, num_iter, target_class):
    changed_tensor = None
    delta = None
    return changed_tensor, delta

target_class = None
epsilon = None
alpha = None
num_iter = None
tattack_tensor, tattack_delta = targeted_attack(example_tensor, epsilon, alpha, num_iter, target_class)
tattack_pred = model(norm(tattack_tensor))
tattack_class = most_probable_classes(tattack_pred, 5, imagenet_classes, print_stats=True)

visualize_attack(example_tensor, tattack_delta, tattack_tensor, model, imagenet_classes, save_path="tattack.png")










