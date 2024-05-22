import os
import torch
import torchaudio
from torch import nn, optim
from torchaudio import transforms
import os
import matplotlib.pyplot as plt
import sys
import random
import json
import numpy as np
import sounddevice as sd
from utils import *
from skimage.transform import rescale, resize, downscale_local_mean
from matplotlib.pyplot import imshow

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

num_devices = 3
samples_folder = "./keywords"
train_samples_split = 164  # Number of samples for training of each keyword
test_samples_split = 16
keywords_buttons = {
    "montserrat": 0,
    "pedraforca": 1,
    "vermell": 2,
    "blau": 3,
}
keywords = list(keywords_buttons.keys())

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device {device}")


def sampleData(path):
    with open(path) as f:
        data = json.load(f)
        return np.array(data['payload']['values']).astype(np.int16)


def processSample(path):
    values = sampleData(path)
    waveform = torch.Tensor(values)
    transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=650,
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=13,
        f_min=300,
    )
    mfccs = transform(waveform)
    # plot_audio(waveform, mfccs)
    return torch.flatten(mfccs.to(device))


def play(path):
    values = sampleData(path)
    sd.play(values, 16000)


class NN(nn.Module):
    def __init__(self, hl_size):
        self.hl_size = hl_size
        super().__init__()
        self.linear_one = torch.nn.Linear(650, self.hl_size)
        self.linear_one.weight.data.normal_(0, 0.1)

        self.linear_two = torch.nn.Linear(self.hl_size, 4)
        self.linear_two.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = torch.sigmoid(self.linear_one(x))
        out = torch.sigmoid(self.linear_two(out))
        return out


def train(model, optimizer, loss_fn, startIndex, endIndex):
    model.train(True)
    for i in range(startIndex, endIndex):
        file_index = i % len(train_files)
        filename = train_files[file_index]
        path = f"{samples_folder}/{filename}"
        keyword = filename.split("/")[0]
        inputs = processSample(path)

        optimizer.zero_grad()
        outputs = model(inputs)
        targets = torch.Tensor([
            (1 if kw == keyword else 0) for kw in keywords_buttons.keys()
        ]).to(device)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # play(path)
        # accuracy_evolution.append(test(model))


def test(model: NN):
    correct = 0
    model.train(False)
    with torch.no_grad():
        for filename in test_files:
            keyword = filename.split("/")[0]
            inputs = processSample(f"{samples_folder}/{filename}")
            output = model(inputs)
            pred = nn.Softmax(dim=0)(output)
            y_pred = pred.argmax(0)
            correct += keyword == keywords[y_pred]
    accuracy = 100 * correct / len(test_files)
    return accuracy


def resizeModel(sourceModel: NN, targetModel: NN):
    layer1 = resize(
        sourceModel.linear_one.weight.data.detach().cpu().numpy(),
        targetModel.linear_one.weight.data.shape,
        anti_aliasing=False
    )
    layer2 = resize(
        sourceModel.linear_two.weight.data.detach().cpu().numpy(),
        targetModel.linear_two.weight.data.shape,
        anti_aliasing=False
    )
    targetModel.linear_one.weight.data = torch.Tensor(layer1).to(device)
    targetModel.linear_two.weight.data = torch.Tensor(layer2).to(device)
    # imshow(layer1, cmap='Greys') plt.show()


def mergeModels(smallModel: NN, mediumModel: NN, largeModel: NN):
    first_layers = []
    second_layers = []
    for model in [smallModel, mediumModel]:
        first_layers.append(
            resize(
                model.linear_one.weight.data.detach().cpu().numpy(),
                largeModel.linear_one.weight.data.shape,
                anti_aliasing=False
            )
        )
        second_layers.append(
            resize(
                model.linear_two.weight.data.detach().cpu().numpy(),
                largeModel.linear_two.weight.data.shape,
                anti_aliasing=False
            )
        )

    first_layers.append(largeModel.linear_one.weight.data.detach().cpu().numpy())
    second_layers.append(largeModel.linear_two.weight.data.detach().cpu().numpy())

    weights = [smallModel.hl_size, mediumModel.hl_size, largeModel.hl_size]
    # weights = [smallModel.hl_size * smallModel.hl_size, mediumModel.hl_size * mediumModel.hl_size, largeModel.hl_size * largeModel.hl_size]
    # weights = [1, 1, 1]
    big_layer_1 = np.average(first_layers, axis=0, weights=weights)
    big_layer_2 = np.average(second_layers, axis=0, weights=weights)

    medium_layer_1 = resize(big_layer_1, mediumModel.linear_one.weight.data.shape)
    medium_layer_2 = resize(big_layer_2, mediumModel.linear_two.weight.data.shape)

    small_layer_1 = resize(big_layer_1, smallModel.linear_one.weight.data.shape)
    small_layer_2 = resize(big_layer_2, smallModel.linear_two.weight.data.shape)

    return big_layer_1, big_layer_2, medium_layer_1, medium_layer_2, small_layer_1, small_layer_2


accuracy_evolution = []

files = []
test_files = []
for i, word in enumerate(keywords):
    file_list = os.listdir(f"{samples_folder}/{word}")
    if len(file_list) < train_samples_split + test_samples_split:
        sys.exit(f"[MAIN] Not enough samples for keyword {word}")
    random.shuffle(file_list)
    files.append(list(map(lambda f: f"{word}/{f}", file_list[0:train_samples_split])))
    test_files.append(
        list(map(lambda f: f"{word}/{f}", file_list[train_samples_split:(train_samples_split + test_samples_split)])))

train_files = list(sum(zip(*files), ()))
test_files = list(sum(zip(*test_files), ()))

loss_fn = nn.CrossEntropyLoss()

big_model = NN(50).to(device)
big_optimizer = optim.SGD(big_model.parameters(), lr=0.1, momentum=0.9)

medium_model = NN(40).to(device)
medium_optimizer = optim.SGD(medium_model.parameters(), lr=0.1, momentum=0.9)

small_model = NN(25).to(device)
small_optimizer = optim.SGD(small_model.parameters(), lr=0.1, momentum=0.9)

big_accuracy = []
med_accuracy = []
small_accuracy = []

for i in range(0, len(train_files), len(keywords) * 3):
    train(big_model, big_optimizer, loss_fn, i, i + len(keywords))
    train(medium_model, medium_optimizer, loss_fn, i + len(keywords), i + (len(keywords) * 2))
    train(small_model, small_optimizer, loss_fn, i + (len(keywords) * 2), i + (len(keywords) * 3))

    big_l_1, big_l_2, med_l_1, med_l_2, sma_l_1, sma_l_2 = mergeModels(small_model, medium_model, big_model)

    with torch.no_grad():
        big_model.linear_one.weight.data.copy_(torch.Tensor(big_l_1))
        big_model.linear_two.weight.data.copy_(torch.Tensor(big_l_2))
        medium_model.linear_one.weight.data.copy_(torch.Tensor(med_l_1))
        medium_model.linear_two.weight.data.copy_(torch.Tensor(med_l_2))
        small_model.linear_one.weight.data.copy_(torch.Tensor(sma_l_1))
        small_model.linear_two.weight.data.copy_(torch.Tensor(sma_l_2))

    big_accuracy.append(test(big_model))
    med_accuracy.append(test(medium_model))
    small_accuracy.append(test(small_model))

plt.title("Accuracy evolution with exp weighted FL")
plt.plot(big_accuracy, label="Larger model")
plt.plot(med_accuracy, label="Medium model")
plt.plot(small_accuracy, label="Smaller model")
plt.legend()
plt.ylim([0, 100])
plt.show()
