"""
    Baseline Convolutional Neural Network for
    RSNA Pneumonia Classification.

    Architecture

    Input (1 x 512 x 512)

    ↓

    Conv Block 1

    ↓

    Conv Block 2

    ↓

    Classifier

    ↓

    Binary Prediction

"""


import torch
import torch.nn as nn

class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Feature Extractor
        self.features = nn.Sequential(

            # Layer 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(
                64,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1
            ),

            nn.Sigmoid()
                            
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x