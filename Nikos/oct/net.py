import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, (3, 3))
        self.maxpool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(50, 100, (3, 3))
        self.conv3 = nn.Conv2d(100, 200, (3, 3))
        self.fc1 = nn.Linear(200, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 4)

    def forward(self, x):
        x = self.maxpool(f.relu(self.conv1(x)))
        x = self.maxpool(f.relu(self.conv2(x)))
        x = self.maxpool(f.relu(self.conv3(x)))
        x = f.avg_pool2d(x, kernel_size=x.size()[2:])  # TODO: Check if this works. The model might be unable to learn with this shit
        x = x.view(x.size()[0], -1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim=1)
        return x
