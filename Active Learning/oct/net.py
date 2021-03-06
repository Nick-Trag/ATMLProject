import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, (3, 3))
        self.maxpool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(25, 50, (3, 3))
        self.conv3 = nn.Conv2d(50, 100, (3, 3))
        self.fc1 = nn.Linear(100 * 23 * 23, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 4)

    def forward(self, x):
        x = self.maxpool(f.relu(self.conv1(x)))
        x = self.maxpool(f.relu(self.conv2(x)))
        x = self.maxpool(f.relu(self.conv3(x)))
        # x = f.avg_pool2d(x, kernel_size=x.size()[2:])
        # x = x.view(x.size()[0], -1)
        x = x.view(-1, 100 * 23 * 23)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim=1)
        return x
