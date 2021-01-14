import utils
import eve
import eve.core.layer as elayer
from eve.core import Eve
import torch as th

class FBS(Eve):
    def __init__(self):
        super().__init__()

        self.conv1 = elayer.FBSConv2d(3, 3, 5, 1, 2)
        self.fc1 = elayer.FBSLinear(in_features=3*32*32, out_features=10)
    
    def forward(self, x):
        conv1 = self.conv1(x)

        conv1 = conv1.view(-1, 1, 3 * 32 * 32)
        return self.fc1(conv1)
    
net = FBS().cuda()

net(th.randn(10, 3, 32, 32).cuda()).mean().backward()