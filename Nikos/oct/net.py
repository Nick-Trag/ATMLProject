import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 200, 3)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(200, 100, 3)
        self.conv3 = nn.Conv2d(100, 50, 3)
        self.fc1 = nn.Linear(50 * 23 * 23, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool(f.relu(self.conv1(x)))
        x = self.maxpool(f.relu(self.conv2(x)))
        x = self.maxpool(f.relu(self.conv3(x)))
        x = x.view(-1, 50 * 23 * 23)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
