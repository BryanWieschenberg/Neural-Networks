# CONVOLUTIONAL NEURAL NETWORK - NUMBER CLASSIFICATION
# Bryan Wieschenberg

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import sys

# --- CNN ARCHITECTURE ---
# Conv1:
#    - Learn 6 different features from 6 filters (5x5 each) from the input images instead of just 1
#    - This helps the model to learn more complex patterns in the images, like edges, corners, distinct shapes, etc.
#    - You need reLU's after each conv layer to introduce non-linearity, which allows the model to learn more complex patterns
# Conv2:
#    - Learn 16 different features from the 6 features learned in conv1
#    - This allows the model to learn even more complex patterns, like combinations of edges and shapes
# Fully connected layer:
#    - We only do 1 FC layer, as a linear layer since its a much better fit to actually make the final predictions, and linear layers are a good fit for this since it can tell pytorch how much an input looks like each class
#    - Takes the 16 features from conv2 and maps them to 10 output classes (digits 0-9)
#    - This is the final layer that outputs the predicted class for each input image
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # Must call to properly initialize the pytorch nn.Module, which is required for all PyTorch models to work correctly
        self.conv1 = nn.Conv2d(1, 6, 5) # For finding features in the input image, 1 input channel (grayscale), 6 output channels to find more features, kernel size of 5x5
        self.conv2 = nn.Conv2d(6, 16, 5) # Takes 6 input channels from conv1, outputs 16 channels for more features, kernel size of 5x5
        self.pool = nn.MaxPool2d(2, 2) # Pooling layer to half size of the feature dimensions, kernel size of 2x2 and stride of 2 to half the size
        self.relu = nn.ReLU() # Activation function to introduce non-linearity for better learning, applied after each layer (if x < 0, x = 0, else x)
        self.fc = nn.Linear(16 * 4 * 4, 10) # nn.Linear connects all inputs to all outputs, input size is 16 channels, each 4x4 after conv2 and pooling, flatten to 1D tensor

    def forward(self, x):
        x = self.conv1(x) # Forward pass thru 1st convolutional layer (28x28 -> 24x24)
        x = self.relu(x) # Apply ReLU activation function
        x = self.pool(x) # Pool halves size of feature dimensions (24x24 -> 12x12)
        x = self.conv2(x) # Forward pass thru 2nd convolutional layer (12x12 -> 8x8)
        x = self.relu(x) # Apply ReLU activation function
        x = self.pool(x) # Halves size again, (8x8 -> 4x4)
        x = x.view(-1, 16 * 4 * 4) # Flatten to 1D tensor based on batch size (-1 gets batch size of 120) since FCs expect 1D input, so x shape goes from [120, 1, 28, 28] -> [120, 256] as FC input
        x = self.fc(x) # Forward pass thru fully connected layer (256 features -> 10 classes)
        return x # Returns the output logits (raw unnormalized prediction where higher logit = higher probability) for each class, which will be used to compute the loss during training

def train(model, train_loader, criterion, optimizer, epochs, batch_size, losses):
    model.train() # Set the model to training mode, prepares it to learn & update params
    
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images) # Forward pass: use the CNN to get output logits for the batch of images. Outputs logits for each of the 120 images, where outputs[i][j] is class j logit for image i
            loss = criterion(outputs, labels) # Compute loss by applying softmax (all logits together = 1 where higher logits mean value closer to 1) then compute cross-entropy loss by applying -log to the softmax output for the correct class, which is our loss (-log(.5) = loss of .3)
            loss.backward() # Backpropagation, applies chain rule to compute gradients wrt each param in each layer of the model & stores gradients in the .grad field of each param to be applied by the optimizer
            optimizer.step() # Applies the gradients in each .grad field to update each model param (where the learning actually happens)
            optimizer.zero_grad() # Reset gradients for next batch, REQUIRED to prevent past gradients from effecting new batches

            losses.append(loss.item())

            if (i + 1) % 100 == 0:
                max_logits, predicted_classes = torch.max(outputs, dim=1) # Get the index of the max log-probability, dim=1 because we want the predicted classes of the images, not the image itself (outputs is [images][classes])
                num_correct = (predicted_classes == labels).sum().item() # Compares predicted classes to actual labels (since both are 1D tensors of size batch_size) and sums up how many predictions were actually correct in this batch with sum() then converts the sum to an actual Python int with .item()
                batch_accuracy = (100.0 * num_correct) / batch_size
                print(f'Epoch {epoch + 1}/{epochs}, Step {i + 1}/{steps}, Accuracy: {num_correct}/{batch_size} | {batch_accuracy:.3f}%, Avg loss: {loss.item():.3f}')
                print(f'First 10 in Batch Predicted: {predicted_classes[:10].tolist()}')
                print(f'First 10 in Batch Actual   : {labels[:10].tolist()}\n')
    
    print("Training complete! Evaluating model...\n")

def eval(model, test_loader):
    model.eval() # Set the model to evaluation mode, makes its behavior consistent and won't learn or update params
    
    correct, total = 0, 0 # Counts correct predictions & total samples
    correct_classes, sample_classes = [0] * 10, [0] * 10 # Corrects correct counts for each class
    
    for images, labels in test_loader:
        outputs = model(images) # Forward pass: the model gets output logits for each image in the batch
        max_logits, predicted_classes = torch.max(outputs, dim=1) # Get the index of the max log-probability

        total += len(labels) # Update total number of samples, we can't just inc with batch_size since last batch isn't divisible by 120 (10000 % 120 != 0)
        num_correct = (predicted_classes == labels).sum().item()
        correct += num_correct
        
        for i in range(len(labels)): # Loop thru each image in the batch
            label = labels[i] # Get the actual label for the image
            prediction = predicted_classes[i] # Get the predicted class for the image
            if (label == prediction): # If the prediction is correct, inc correct count for that class
                correct_classes[label] += 1
            sample_classes[label] += 1 # Inc sample count for that class

    for i in range(10):
        class_accuracy = (100.0 * correct_classes[i]) / sample_classes[i]
        print(f'Accuracy of {classes[i]}: {correct_classes[i]}/{sample_classes[i]} | {class_accuracy:.3f}%')

    total_accuracy = (100 * correct) / total

    print("\nEvaluation complete!")
    print(f'Accuracy of the model on the test set: {total_accuracy:.3f}%')

# ---------------------------------------------------------------------------------------------

# Hyperparams
epochs = 5 # Num of times model will train on the entire dataset, more epochs can mean greater accuracy
batch_size = 120 # Image ct per batch, larger batch size can speed up training but requires more memory
learning_rate = .01 # How much the model's params will change based on loss, smaller values can lead to more precise learning but slower convergence. I've found that typically for pytorch, .01 works well for CNNs

# Load MNIST dataset - 60,000 training images and 10,000 test images
train_dataset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor()) # train=True = get training set, transform=transforms.ToTensor() converts images to PyTorch tensors
test_dataset = torchvision.datasets.MNIST(root='./mnist', train=False, download=False, transform=transforms.ToTensor()) # train=False = get test set

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Loader for training set, shuffle=True means data will be shuffled each epoch for better generalization
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Loader for test set, shuffle=False means data will not be shuffled since it would not matter for evaluation

examples = iter(train_loader) # Get a batch of training data
batch_images, batch_labels = next(examples) # Get the first batch of images and labels

print(f'Image batch: {batch_images.shape}, Label batch: {batch_labels.shape}\n') # Batch shape, [60, 1, 28, 28] means 60 images per batch, 1 channel (grayscale), 28x28 pixel images

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Initialize model
model = CNN() # Create an instance of the CNN model
criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification, quantifies how incorrect the model's predictions are by applying softmax then computing cross-entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD optimizer, determines how weights/biases are updated based on loss, approximating gradient using smaller batches of training data (appropriate for this larger dataset). model.parameters() allows this optimizer to update the model's weights and biases during training
steps = len(train_loader) # Num of batches in the training set
losses = [] # List to store loss for each batch, used for plotting later

# Train & evaluate the model
train(model, train_loader, criterion, optimizer, epochs, batch_size, losses)
eval(model, test_loader)

# Plot for the loss over time
plot.plot(losses, label='Batch Loss')
plot.xlabel("Batch Number")
plot.ylabel("Loss")
plot.title("Loss Over Time")
plot.legend()
plot.grid(True)
plot.show()
