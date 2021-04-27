#%% Import packages
import torch
import torch.nn as nn

#%% CNN
class CNN_01(nn.Module):
    def __init__(self):
        super(CNN_01, self).__init__()
        def conv_bn_relu(in_dim, out_dim):
            F = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
                )
            return F
        self.cv1 = nn.Sequential(
            conv_bn_relu(3, 64),
            conv_bn_relu(64, 64),
            
            nn.MaxPool2d(2),
            
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),           
        )

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 43),
            nn.Softmax(-1)
        )
        self.tap_MP = nn.MaxPool2d(8)
        self.tap_AP = nn.AvgPool2d(8)
        self.MLP = nn.Sequential(
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Linear(4, 64)
            )    
        self.sf = nn.Softmax(-1)        
    def forward(self, x):
        bz = x.size(0)
        y1 = self.cv1(x)
        y2 = self.cv2(y1)
        # print(y2.size())
        y2_MP = self.MLP(self.tap_MP(y2).view(bz, -1))
        y2_AP = self.MLP(self.tap_AP(y2).view(bz, -1))
        ch_atm = self.sf(y2_MP + y2_AP)
        y3 = y2 * ch_atm.view(bz, -1, 1, 1)
        pred = self.FC(y3)
        return pred, ch_atm

#%% Test
if __name__ == "__main__":
    x = torch.rand(32, 3, 32, 32)
    F = CNN_01()
    y, at = F(x)
    print(y.size())
    print(at.size())