import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


def get_ranking(net, dataset, known_samples, device, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    entropies = np.zeros(len(dataset) - known_samples)
    counter = known_samples
    # TODO: Start from known_samples. Use a sampler
    for i_batch, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)




def main():
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        # transforms.RandomCrop(250),
        transforms.ToTensor()
    ])
    train_set = OCTDataset(root_directory=os.path.join(data_root, 'OCT2017'), transform=transform, mode='train')

    batch_size = 8

    train_set_size = len(train_set)

    known_samples = int(0.01 * train_set_size)

    # Randomly choose the samples that we will consider labeled to start with
    labeled_indices = np.random.choice(len(train_set), size=known_samples, replace=False)

    train_set_l = Subset(train_set, labeled_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_loader = DataLoader(train_set_l, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    epochs = 100
    model_name = 'model.pth'

    # Train loop
    for epoch in range(epochs):
        for i_batch, (images, labels) in enumerate(labeled_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(net.state_dict(), model_name)

    # TODO: Now test it on the pool. Test it on the entire train set (no shuffle), but consider only the unlabeled ones. Then, add the indices to labeled_indices and recreate the subset and the loader
    # (and save the accuracies)


if __name__ == '__main__':
    main()
