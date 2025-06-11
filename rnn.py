# RECURRENT NEURAL NETWORK - SPAM DETECTION
# Bryan Wieschenberg

import torch
import torch.nn as nn
import torch.optim
from rnn_utils import train_set, test_set
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import sys

# --- RNN ARCHITECTURE ---
# Embed:
#    - Converts word IDs into vectors called embeddings of size embed_size
#    - These vectors start completely random and are adjusted during training so that words with similar meanings in the sentence end up with similar vectors
#    - This helps the model understand relationships between words, like "money" and "prince" being related to spam
#    - The actual values are not important to us, but they are important to the model since they are the ones actually forming the meaning behind the words
#    - Padding index 0 does NOT get trained (padding_idx=0), which makes sure the extra padding doesn't affect learning or gradients
# RNN:
#    - Sequentially processes the encoded sentence 1 word at a time
#    - Keeps track of what it has seen so far using a hidden state that is updated at each step and passed to the next step
#    - At each time step: Takes 1 word embedding, combines it with the previous hidden state, and produces a new hidden state
#    - The hidden size is 20D, which is good for smaller data but larger data demands more dimensions. Using too big of dimensions with smaller data risks overfitting and slower training
#    - It outputs an output, which is the RNNs state after reading each word, and the final state, which has actually processed the entire sentence
# FC Layer:
#    - After processing the full sentence, the RNN returns the final hidden state, which is a 20D vector encapsulating information about the sentence that only the model understands
class RNN(nn.Module):
    def __init__(self, len_vocab, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(len_vocab, embed_size, padding_idx=0) # Prevents updating the <Padding> index 0 during training & makes sure it embedding doesn't effect learning
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True) # Make batch_first=True so input shape is [batch, sequence_length, embed_dim] instead of [sequence_length, batch, embed_dim] to align with the input shape of [batch, sequence_length]
        self.fc = nn.Linear(hidden_size, 1) # Fully connected layer to map the final hidden state to output a binary identifier of spam

    def forward(self, x):
        x = self.embed(x) # Convert word IDs the proper embedding, of shape [batch, sequence_length, embed_dim]
        output, final_hidden = self.rnn(x) # Process the embeddings through the RNN and receive output at every time step and final hidden state after the final time step of shape [num_layers * num_directions, batch, hidden_size], and our RNN only uses the def 1 layer & isn't bidirectional, so num_directions is 1
        final_hidden = self.fc(final_hidden.squeeze(0)) # Removes the 1st dimension of final_hidden, [1 x 1], making it [batch, hidden_size] so the FC layer can correctly process it
        return final_hidden # self.fc() returns the logit for spam identifying based on the final_hidden 20D vector

# Train
def train(model, sentences_tensor, spamIDs_tensor, criterion, optimizer, epochs, num_samples, losses):
    model.train()

    for epoch in range(epochs):
        output = model(sentences_tensor) # Get the model's output logits for the training sentences for the current epoch
        output = torch.sigmoid(output) # Apply sigmoid to convert logits to probabilities, where 0.5 is the threshold for spam vs not spam
        loss = criterion(output, spamIDs_tensor) # Compute loss, same idea as in CNN, only now using BCE loss b/c binary classification instead of CEL (multi-class)
        loss.backward() # Backpropagate to compute gradients
        optimizer.step() # Update model parameters based on gradients
        optimizer.zero_grad() # Reset gradients for the next batch

        losses.append(loss.item())

        predicted = (output >= 0.5).long() # Convert probabilities to binary based on the threshold of 0.5
        correct = (predicted == spamIDs_tensor).float().sum().item() # Count how many predictions were correct by comparing the predicted values to actual
        accuracy = (correct / num_samples) * 100

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {int(correct)}/{num_samples} | {accuracy:.3f}%, Avg loss: {loss.item():.3f}")
            print(f'First 10 in Epoch Predicted: {(output[:10] >= 0.5).long().squeeze().tolist()}')
            print(f'First 10 in Epoch Actual   : {spamIDs_tensor[:10].long().squeeze().tolist()}\n')

    print("Training complete! Evaluating model...\n")

def eval(model, test_set, num_samples):
    model.eval() # Set the model to evaluation mode

    correct = 0
    correct_classes, sample_classes = [0, 0], [0, 0]

    for sentence, label in test_set: # Go sentence by sentence
        x = torch.tensor([toModelReadable(sentence, max_len)], dtype=torch.long)

        output = torch.sigmoid(model(x)) # Get the model's output logits to sigmoid probabilities
        prediction = (output >= 0.5).float().item() # Conv to bin with same threshold
        label = int(label)

        if prediction == label:
            correct += 1
            correct_classes[label] += 1
        sample_classes[label] += 1

    for i in range(2):
        class_accuracy = 100.0 * correct_classes[i] / sample_classes[i]
        print(f'Accuracy of {classes[i]}: {correct_classes[i]}/{sample_classes[i]} | {class_accuracy:.3f}%')

    total_accuracy = 100.0 * correct / sum(sample_classes)

    print("\nEvaluation complete!")
    print(f'Accuracy of the model on the test set: {correct}/{num_samples} | {total_accuracy:.3f}%\n')

# We can use the model ourselves now
def predict(sentence):
    model.eval()

    x = torch.tensor([toModelReadable(sentence, max_len)], dtype=torch.long) # Conv to model-readable format
    prob = torch.sigmoid(model(x))
    if prob.item() >= 0.5:
        return "spam"
    else:
        return "not spam"

# ---------------------------------------------------------------------------------------------

# Create vocab dictionary for the dataset, which will let the model read sentences as lists of word int IDs
vocab = {"<Padding>": 0, "<Unknown>": 1} # <Padding> for padding all strings to equal len w/ 0s, <Unknown> for unknown words in the test set to 1
for sentence, isSpam in train_set:
    for word in sentence.split(): # Split sentence into word tokens
        if word not in vocab: # If word not in vocab, add it. Indices 0 and 1 will always be reserved for padding 0s & test set unknown words
            vocab[word] = len(vocab)

# Encode sentence as list of token IDs
def toModelReadable(sentence, max_len):
    tokens = sentence.split() # Split sentence into word tokens
    tokenIDs = [] # List to hold the token IDs

    for token in tokens:
        tokenIDs.append(vocab.get(token, vocab["<Unknown>"])) # Get the token ID from the vocab, or use <Unknown> if the token is not in the vocab

    padding = max_len - len(tokenIDs) # Get how many padding tokens we need to add
    tokenIDs.extend([vocab["<Padding>"]] * padding) # Add padding tokens to the end of the list to make it max_len length
    return tokenIDs

# Prepare dataset
max_len, sentences, spamIDs = 0, [], []
for sentence, isSpam in train_set: # Loop thru training set to find the max len of a sentence
    max_len = max(max_len, len(sentence.split()))
    spamIDs.append(isSpam) # Add spamIDs to a list for training
for sentence, isSpam in train_set: # Once max_len is found, loop thru training set again to encode sentences to model-readable
    sentences.append(toModelReadable(sentence, max_len))

sentences_tensor = torch.tensor(sentences, dtype=torch.int32) # Convert sentences to tensor so pytorch can work with it
spamIDs_tensor = torch.tensor(spamIDs, dtype=torch.float32).unsqueeze(1) # Convert spamIDs to tensor of float32 (needed for our BCE loss function) and unsqueeze(1) turns it into a column vector (also needed for BCE)

print(f'{sentences_tensor.shape} {spamIDs_tensor.shape}\n') # Print shapes of tensors to make sure they're correct

# Hyperparams
epochs = 10000 # We don't need a batch size since the dataset's small
learning_rate = .0025 # Learning rate for the optimizer, I found that .0025 works well for this model
embed_size = 10 # Size of the word embeddings, which are mapped to words to help the model understand words. We choose 10 since it's a small dataset and we want to keep it simple, but larger datasets may benefit from larger embeddings. Too large may lead to overfitting though
hidden_size = 20 # Size of the hidden state in the RNN, which are used to keep track of features of a sentence. We choose 20 to allow the RNN to learn more complex patterns in the data, but larger datasets may benefit from larger hidden sizes

# Initialize model
len_vocab = len(vocab)
model = RNN(len_vocab, embed_size, hidden_size) # Creates the model
classes = ["not spam", "spam"]
losses = [] # For matplotlib

# Loss & optimizer
criterion = nn.BCELoss() # Binary Cross-Entropy Loss, which is used for binary classification, and it compares the predicted labels to the actual labels and calculates the loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Can also use Adam, but SGD is simpler for this example (we can see how much more accurate Adam is)

# Train & evalute model
train(model, sentences_tensor, spamIDs_tensor, criterion, optimizer, epochs, len(train_set), losses)
eval(model, test_set, len(test_set))

# Plot for the loss over time
# Notice how the loss is MUCH smoother than in CNN since we're not using batches, and there's only 1 loss update per epoch
# Typically, real-world datasets will be more akin to the CNN plot since the loss is more erratic due to the larger dataset and batch sizes
plot.plot(losses, label='Epoch Loss')
plot.xlabel("Epoch Number")
plot.ylabel("Loss")
plot.title("Loss Over Time")
plot.legend()
plot.grid(True)
plot.show()

# We can use the model ourselves now since it's text-based
#    - Model is not super accurate since the dataset is so small, but it is somewhat accurate
while True:
    usr_input = input("Enter sentence ('quit' to quit): ")
    if usr_input == 'quit':
        break
    print(predict(usr_input))
