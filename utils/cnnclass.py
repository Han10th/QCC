import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierCNN(nn.Module):
    def __init__(self, n_input = 2, n_output = 2, size=[128,128]):
        super().__init__()        
        h = int((((size[0]/2-4)/2)))
        self.net = nn.Sequential(
            nn.Conv2d(n_input, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(16, 32, 3),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(32, 64, 5),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(h * h * 16, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, n_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class ClassifierMINI(nn.Module):
    def __init__(self, n_input = 2, n_output = 2, size=[28,28]):
        super().__init__()
        h = int(((size[0] - 4)/2 - 4)/2)
        self.conv1 = nn.Conv2d(n_input, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(h * h * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x