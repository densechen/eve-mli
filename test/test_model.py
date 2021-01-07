import utils  # pylint: disable=import-error
import eve
import torch as th
import torch.nn.functional as F
import torch.nn as nn

from eve.core.eve import Eve
from eve.app.model import Classifier


class Net(Eve):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.conv(x)


classifier = Classifier(Net())
classifier.prepare_data(data_root="/media/densechen/data/dataset")

classifier.setup_train()

# train it via trainer
from eve.app.trainer import BaseTrainer

trainer = BaseTrainer(classifier)

trainer.fit(200)