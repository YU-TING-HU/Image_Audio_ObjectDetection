# -*- coding: utf-8 -*-
"""
# Data EDA
"""

import torchaudio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch


"""## Plot Audio"""

def plot_waveform(waveform, sample_rate):
    """
    Plots the waveform of the audio signal.

    Args:
        waveform (Tensor): The audio waveform.
        sample_rate (int): The sample rate of the audio.
    """    
    waveform = waveform.numpy()
    # Get number of channels and frames
    num_channels, num_frames = waveform.shape
    # Create time axis for plotting
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}") # Set ylabel for multi-channel
    figure.suptitle("waveform")
    plt.show(block=False)

# Source from https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
def plot_specgram(waveform, sample_rate, file_path = 'test2.png'):
    """
    Plots and saves the spectrogram of the audio signal.

    Args:
        waveform (Tensor): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        file_path (str): The path to save the spectrogram image.
    """
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape # Get number of channels and frames

    fig, axes = plt.subplots(num_channels, 1)
    fig.set_size_inches(10, 10)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}") # Set ylabel for multi-channel
    plt.gca().set_axis_off() # Turn off axis
    plt.gca().axes.get_xaxis().set_visible(False) # Hide x-axis
    plt.gca().axes.get_yaxis().set_visible(False) # Hide y-axis
    # plt.show(block=False)
    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0)

# Exsample data: extrahls__201101070953.wav
wav_file = '/content/drive/MyDrive/065_CNN_AudioClassification/set_a/extrahls__201101070953.wav'
data_waveform, sr = torchaudio.load(wav_file) # Load waveform and sample rate
data_waveform.size()

# Plot Waveform
plot_waveform(data_waveform, sample_rate=sr)

# Calculate and plot spectrogram
spectogram = torchaudio.transforms.Spectrogram()(data_waveform)
spectogram.size()
plot_specgram(waveform=data_waveform, sample_rate=sr)


"""# Data Preprocessing"""

import torchaudio
import os
import random

wav_path = '/content/drive/MyDrive/065_CNN_AudioClassification/set_a'
wav_filenames = os.listdir(wav_path)
random.shuffle(wav_filenames)

wav_filenames[0]
wav_filenames.index("Aunlabelledtest__201106040930.wav")

ALLOWED_CLASSES = ['normal', 'murmur', 'extrahls', 'artifact']
for f in wav_filenames:

    class_type = f.split('_')[0] # Extract class type from filename
    f_index = wav_filenames.index(f) # Get index of the file
    target_path = 'train' if f_index < 140 else 'test'
    class_path = f"{target_path}/{class_type}"
    file_path = f"{wav_path}/{f}"
    f_basename = os.path.basename(f)
    f_basename_wo_ext = os.path.splitext(f_basename)[0]
    target_file_path = f"{class_path}/{f_basename_wo_ext}.png"
    if (class_type in ALLOWED_CLASSES):
        # create folder if necessary
        if not os.path.exists(class_path):
            os.makedirs(class_path)

        # extract class type from file
        data_waveform, sr = torchaudio.load(file_path)
        # create spectrogram and save it
        plot_specgram(waveform=data_waveform, sample_rate=sr, file_path=target_file_path)


"""# Model"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# Define transformations for the training and testing data
transform = transforms.Compose(
    [transforms.Resize((100,100)), # Resize images to 100x100
    transforms.Grayscale(num_output_channels=1), # Convert images to grayscale
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize((0.5, ), (0.5, ))])  # Normalize images

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

trainset
testset

CLASSES = ['artifact', 'extrahls', 'murmur', 'normal']
NUM_CLASSES = len(CLASSES)
class ImageMulticlassClassificationNet(nn.Module):
    """
    Convolutional Neural Network for multi-class image classification.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size= 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels= 16, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100*100, 128) # out: (BS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x) # out: (BS, 6, 100, 100)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 6, 50, 50)
        x = self.conv2(x) # out: (BS, 16, 50, 50)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 16, 25, 25)
        x = self.flatten(x)  # out: (BS, 10000)
        x = self.fc1(x)  # out: (BS, 128)
        x = self.relu(x)
        x = self.fc2(x)  # out: (BS, 64)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# input = torch.rand(1, 1, 100, 100) # BS, C, H, W
# model(input).shape
model = ImageMulticlassClassificationNet()

# Training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
losses_epoch_mean = [] # List to store mean losses for each epoch
NUM_EPOCHS = 100
for epoch in range(NUM_EPOCHS):
    losses_epoch = [] # List to store losses for each batch
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        losses_epoch.append(loss.item()) # Append loss to the list
    losses_epoch_mean.append(np.mean(losses_epoch)) # Append mean loss for the epoch
    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {np.mean(losses_epoch):.4f}')

# Plot the training loss
sns.lineplot(x=list(range(len(losses_epoch_mean))), y=losses_epoch_mean)

# Testing
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()

    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# Accuracy
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# confusion matrix
cm = confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
sns.heatmap(cm, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)

Counter(y_test) # Print count of each class in the test set