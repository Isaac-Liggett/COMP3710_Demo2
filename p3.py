from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

import torch
import torch.nn as nn

# Download the data , if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes ( for plotting )
n_samples, h, w = lfw_people.images.shape
# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model )
X = lfw_people.data
n_features = X.shape[1]
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print ("Total dataset size:")
print ("n_samples: %d" % n_samples)
print ("n_features: %d" % n_features)
print ("n_classes: %d" % n_classes)

# The following code could be used for data preprocessing if using pytorch
X = lfw_people.images
Y = lfw_people.target
# Verify the value range of X_train . No normalization is necessary in this case ,
# as the input values already fall within the range of 0.0 to 1.0.
print ("X_min:", X.min(), "X_train_max:", X.max())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
X_train = torch.Tensor(X_train[:, np.newaxis, :, :])
X_test = torch.Tensor(X_test[:, np.newaxis, :, :])
print ("X_train shape:", X_train.shape)

X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

### TRAINING THE MODEL ###
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.Conv2d(32, 32, 3),
    nn.Flatten(),
    nn.Linear(48576, 7),
    nn.Softmax()
)

loss = nn.CrossEntropyLoss()

EPOCHS = 10
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    pred = model.forward(X_train)
    out = loss(pred, y_train)
    train_losses.append(out.item())

    pred_test = model.forward(X_test)
    test_loss = loss(pred_test, y_test)
    test_losses.append(test_loss.item())

    out.backward()

# The below plotting code was AI generated
import matplotlib.pyplot as plt

# Generate an x-axis sequence representing epochs
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 6))

# Plot both arrays
plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
plt.plot(epochs, test_losses, label='Test Loss', marker='s', color='orange')

# Formatting
plt.title('Training and Test Loss Overlay')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.tight_layout()
plt.show()
