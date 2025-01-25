import torch.nn as nn

class CNN1(nn.Module):
    def __init__(self,num_classes):
        super(CNN1, self).__init__()
        # No padding
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 *54 *54, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.cnn_stack(x)
