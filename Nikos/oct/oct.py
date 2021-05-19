import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from Nikos.oct.dataset import OCTDataset
from Nikos.oct.net import Net

data_root = os.path.join(os.path.dirname(__file__), '../data')
RANDOM_STATE = 7


def accuracy(net, dataset, device, batch_size=8):
    correct = 0
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        predictions = [torch.argmax(outputs[j]) for j in range(len(outputs))]
        for j in range(len(predictions)):
            if labels[j] == predictions[j]:
                correct += 1
    return (correct / len(dataset)) * 100


def get_entropies(net, dataset, device, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    entropies = np.zeros(len(dataset))
    counter = 0
    for i_batch, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        for i in range(len(outputs)):
            entropies[counter] = entropy(outputs[i], base=2)
            counter += 1
    return entropies


# Return the top n indices that don't already exist in labeled indices
def take_top_n(ranking, labeled_indices, n):
    indices = np.zeros(n)
    counter = 0
    for i in range(len(ranking)):
        if ranking[i] not in labeled_indices:
            indices[counter] = ranking[i]
            counter += 1
        if counter == n:
            return indices
    return indices[:counter]  # If there aren't n samples that don't exist in labeled_indices, just return the everything we've found (not needed for this project)


# Return n indices, uniformly sampled from the top m (that all aren't in labeled_indices)
def take_n_uniformly_from_top_m(ranking, labeled_indices, n, m):
    top_m = np.zeros(m, dtype=int)
    counter = 0
    for i in range(len(ranking)):
        if ranking[i] not in labeled_indices:
            top_m[counter] = ranking[i]
            counter += 1
        if counter == m:
            break
    mask = np.linspace(0, m - 1, num=n, dtype=int)
    return top_m[mask]


def take_n_randomly(train_set_size, labeled_indices, n):
    if train_set_size - len(labeled_indices) < n:
        raise ValueError("Only " + str(train_set_size - len(labeled_indices)) + " samples can be drawn, but was asked to draw " + str(n))
    indices = np.zeros(n)
    counter = 0
    while True:
        next_index = np.random.randint(0, train_set_size)
        if next_index not in labeled_indices:
            indices[counter] = next_index
            counter += 1
        if counter == n:
            return indices


def main():
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        # transforms.RandomCrop(250),
        transforms.ToTensor()
    ])
    train_set = OCTDataset(root_directory=os.path.join(data_root, 'OCT2017'), transform=transform, mode='train')
    test_set = OCTDataset(root_directory=os.path.join(data_root, 'OCT2017'), transform=transform, mode='test')

    batch_size = 8

    train_set_size = len(train_set)

    known_samples = int(0.01 * train_set_size)

    # Randomly choose the samples that we will consider labeled to start with
    initial_labeled_indices = np.random.choice(len(train_set), size=known_samples, replace=False)

    labeled_indices = initial_labeled_indices.copy()

    train_set_l = Subset(train_set, labeled_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_loader = DataLoader(train_set_l, batch_size=batch_size, shuffle=True)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    max_iterations = 100

    accuracies = np.zeros(max_iterations)

    for i in range(max_iterations):
        print("Iteration " + str(i))
        net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 0.001
        momentum = 0.9
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        epochs = 50
        model_name = 'model.pth'

        # Train loop
        for epoch in range(epochs):
            print("(Training) Epoch " + str(epoch))
            for i_batch, (images, labels) in enumerate(labeled_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            accuracies[i] = accuracy(net, test_set, device, batch_size=batch_size)
            entropies = get_entropies(net, train_set, device, batch_size=batch_size)

            ranking = (-entropies).argsort()

            new_indices = take_top_n(ranking, labeled_indices, 20)

            labeled_indices = np.append(labeled_indices, new_indices)

            train_set_l = Subset(train_set, labeled_indices)

            labeled_loader = DataLoader(train_set_l, batch_size=batch_size, shuffle=True)

        # torch.save(net.state_dict(), model_name)

    labeled_indices = initial_labeled_indices

    uniform_accuracies = np.zeros_like(accuracies)

    for i in range(max_iterations):
        print("Iteration " + str(i))
        net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 0.001
        momentum = 0.9
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        epochs = 50
        model_name = 'model.pth'

        # Train loop
        for epoch in range(epochs):
            print("(Training) Epoch " + str(epoch))
            for i_batch, (images, labels) in enumerate(labeled_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            uniform_accuracies[i] = accuracy(net, test_set, device, batch_size=batch_size)
            entropies = get_entropies(net, train_set, device, batch_size=batch_size)

            ranking = (-entropies).argsort()

            new_indices = take_n_uniformly_from_top_m(ranking, labeled_indices, 20, 200)

            labeled_indices = np.append(labeled_indices, new_indices)

            train_set_l = Subset(train_set, labeled_indices)

            labeled_loader = DataLoader(train_set_l, batch_size=batch_size, shuffle=True)

        # torch.save(net.state_dict(), model_name)

    labeled_indices = initial_labeled_indices

    random_accuracies = np.zeros_like(accuracies)

    for i in range(max_iterations):
        print("Iteration " + str(i))
        net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 0.001
        momentum = 0.9
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        epochs = 50
        model_name = 'model.pth'

        # Train loop
        for epoch in range(epochs):
            print("(Training) Epoch " + str(epoch))
            for i_batch, (images, labels) in enumerate(labeled_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            random_accuracies[i] = accuracy(net, test_set, device, batch_size=batch_size)

            new_indices = take_n_randomly(train_set_size, labeled_indices, 20)

            labeled_indices = np.append(labeled_indices, new_indices)

            train_set_l = Subset(train_set, labeled_indices)

            labeled_loader = DataLoader(train_set_l, batch_size=batch_size, shuffle=True)

        # torch.save(net.state_dict(), model_name)

    x_axis = np.linspace(0, 20 * 99, num=100, dtype=int)
    plt.plot(x_axis, np.convolve(accuracies, np.ones(7)/7, mode='valid'), label='Accuracy when selecting the top 20 samples', color='burlywood')
    plt.plot(x_axis, np.convolve(uniform_accuracies, np.ones(7)/7, mode='valid'), label='Accuracy when selecting 20 samples uniformly from the top 200', color='darkorchid')
    plt.plot(x_axis, np.convolve(random_accuracies, np.ones(7)/7, mode='valid'), label='Accuracy when selecting 20 samples randomly', color='firebrick')
    plt.xlabel('Additional labeled examples')
    plt.ylabel('Test set accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
