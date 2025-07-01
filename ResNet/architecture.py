import torch
import torch.nn as nn
import torch.nn.functional as F
# trong torch, conv2d (in_channels, out_channels, kernel_size, stride, padding, bias)
class BasicBlock(nn.Module):
    expansion_rate = 1 # Tỉ lệ thu phóng số kênh (với Basic là không đổi, với bottleneck là 4)
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm(out_channels)
        )
        self.skip_connect = nn.Sequential()
        if stride !=1 or in_channels!=out_channels:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride), #Projection
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.residual(x) + self.skip_connect(x)
        return F.relu(out)
class Bottleneck(nn.Module):
    expansion_rate = 4
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), # Giảm kênh
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, self.expansion_rate * out_channels, 1, 1, 0, bias=False), # Tăng kênh
            nn.BatchNorm2d(self.expansion_rate * out_channels),
        )
        self.skip_connect = nn.Sequential()
        if stride!=1 or in_channels != self.expansion_rate * out_channels:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion_rate * out_channels),
                nn.BatchNorm2d(self.expansion_rate * out_channels)
            )
    def forward(self, x):
        out = self.bottleneck(x) + self.skip_connect(x)
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1) # kernel_size, stride, padding
        )
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion_rate, num_classes)
    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = block.expansion_rate * out_channels
        for _ in range (1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential (*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

if __name__ == "__main__":
    model = resnet18()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape ResNet-18: {y.shape}")
"""
Công thức tính số lớp:
Các model BasicBlock: 1 cnn đầu + 2 cnn layer mỗi block * (tổng số block mỗi layer) + 1 fc
Các model Bottleneck: 1 cnn đầu + 3 cnn layer mỗi block * (tổng số block mỗi layer) + 1 fc
"""