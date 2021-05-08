#%% Import packages
import torch
import torch.nn as nn

#%% CNN
class resnet_block(nn.Module):
    def __init__(self, in_dim, outdim, cv_1x1=False):
        super(resnet_block, self).__init__()
        self.F = nn.Sequential(
            nn.Conv2d(in_dim, outdim, kernel_size=3, padding=1),
            nn.BatchNorm2d(outdim),
            nn.ReLU(),
            nn.Conv2d(outdim, outdim, kernel_size=3, padding=1),
            nn.BatchNorm2d(outdim),
            )
        if cv_1x1:
            self.cv1 = nn.Conv2d(in_dim, outdim, kernel_size=1)
        else:
            self.cv1 = None
    def forward(self, x):
        y1 = self.F(x)
        if self.cv1:
            y2 = self.cv1(x)
        else:
            y2 = x
        out = torch.nn.functional.relu(y1+y2)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.cvrgb = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.cvgray = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.bk1 = resnet_block(64, 64)
        self.bk1_2 = resnet_block(64, 64)
        self.bk2 = resnet_block(64, 128, cv_1x1=True)
        self.bk3 = resnet_block(128, 128)
        self.bk3_2 = resnet_block(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 43)
        )
    def forward(self, rgb, gray):
        y1_rgb = self.cvrgb(rgb)
        y1_gray = self.cvgray(gray)
        y1 = torch.cat((y1_rgb, y1_gray), dim=1)
        y2 = self.bk1_2(self.bk1(y1))
        y3 = self.bk2(y2)
        y4 = self.bk3_2(self.bk3(y3))
        y5 = self.pool(y4)
        pred = self.FC(y5)
        return pred



#%% Test
if __name__ == "__main__":
    x = torch.rand(32, 3, 32, 32)
    x2 = torch.rand(32, 1, 32, 32)
    F = ResNet18()
    y = F(x, x2)
    print(y.size())
