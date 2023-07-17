import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self, in_channels: int=1, num_labels: int=10):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(84, num_labels),
        )

    def forward(self, img, return_features=False):
        feature = self.feature_extractor(img)
        output = self.classifier(feature)

        if out_feature == False:
            return output
        else:
            return output, feature
    
