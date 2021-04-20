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
            conv_bn_relu(3, 16),
            conv_bn_relu(16, 16),
            
            nn.MaxPool2d(2),
            
            conv_bn_relu(16, 64),
            conv_bn_relu(64, 64),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.MaxPool2d(2), 

            conv_bn_relu(16, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),           
        )
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 43),
        )
        self.sf = nn.Softmax(-1)
    def forward(self, x):
        bz = x.size(0)
        y1 = self.cv1(x)
        y2 = self.cv2(y1)
        pred = self.sf(self.FC(y2))
        # print(y3.size())
        return pred

#%% Test
if __name__ == "__main__":
    x = torch.rand(32, 3, 32, 32)
    F = CNN_01()
    y = F(x)
    print(y)