import torch
import torch.nn as nn
import torch.nn.functional as F

#Thông số rho không phải trong architecture, mà là trong preprocessing
class DSCBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super().__init__()
        self.DSC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False), #Depthwise
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), #Pointwise
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.DSC(x)
    
class MobileNetV1(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 alpha=1.0):
        super().__init__()
        num_channels = [32, 64, 128, 256, 512, 1024]
        num_channels = [max(1, int(x * alpha)) for x in num_channels]
        cfg = [
            [64, 1],
            [128, 2],
            [128, 1],
            [256, 2],
            [256, 1],
            [512, 2],
            [512, 1],
            [512, 1],
            [512, 1],
            [512, 1],
            [512, 1],
            [1024, 2],
            [1024, 1]
        ]
        input_channels = max(1, int(32 * alpha))
        layers = []
        layers.extend([
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        ])

        for channels, stride in cfg:
            output_channels = max(1, int(channels * alpha))
            layers.append(DSCBlock(input_channels, output_channels, stride=stride))
            input_channels = output_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.model = nn.Sequential(*layers)
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        return self.cls(x)

class IRBBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion_rate):
        super().__init__()
        hidden_dim = expansion_rate * in_channels
        self.use_skip_connect = stride == 1 and in_channels == out_channels

        layers = []

        if expansion_rate != 1: # Extend if not increasing channels or decreasing tensor size
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), #Depthwise
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, 1, bias=False), #Pointwise
            nn.BatchNorm2d(out_channels)
        ])
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_skip_connect:
            return x + self.block(x)
        return self.block(x)
    
class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 alpha=1.0):
        super().__init__()
        """
        Config parameters: t, c, n, s
        t (expansion factor): tỉ lệ giãn nở
        c (output channels): số kênh output
        n (num of repeats): số lần lặp lại block
        s (stride of first block): stride của block đầu trong vòng lặp, còn các block còn lại = 1
        """
        adj = lambda x: max(1, int(x * alpha))
        self.config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        input_channels = adj(32)
        last_channels = adj(1280)
        layers = [
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        ]
        
        for t, c, n, s in self.config:
            out_channels = adj(c)
            for i in range(n):
                stride = s if i==0 else 1
                layers.append(IRBBlock(input_channels, out_channels, stride, t))
                input_channels = out_channels
        
        layers.extend([
            nn.Conv2d(input_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        ])

        self.model = nn.Sequential(*layers)
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_channels, num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        return self.cls(x)

class HSigmoid(nn.Module):
    def forward (self, x):
        return F.relu6(x + 3) / 6

class HSwish(nn.Module):
    def forward (self, x):
        return x * F.relu6(x + 3) / 6
    
class SEBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            HSigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        attention = self.pool(x)
        attention = self.fc(attention).view(b, c, 1, 1)
        return x * attention

class IRBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, 
                 kernel_size,
                 stride,
                 expansion,
                 use_se,
                 activation):
        super().__init__()

        hidden_dim = int(in_channels * expansion)
        self.use_skip_connect = stride == 1 and in_channels == out_channels
        act_layer = nn.ReLU(inplace=True) if activation == "RE" else HSwish()

        layers = []
        if expansion != 1: # Extend
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act_layer
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False), #depthwise
            nn.BatchNorm2d(hidden_dim),
            act_layer
        ])
        if use_se:
            layers.append(SEBlock(hidden_dim))
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False), #pointwise
            nn.BatchNorm2d(out_channels)
        ])
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_skip_connect:
            return x + self.block(x)
        return self.block(x)
    
class MobileNetV3(nn.Module):
    def __init__(self,
                 num_classes=1000, 
                 version = "large"):
        super().__init__()
        act_layer = HSwish()

        layers = [
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            act_layer
        ]

        self.config = [
            # k, out_c, se, nl, s, exp
            [3, 16, False, "RE", 1, 1],
            [3, 24, False, "RE", 2, 4],
            [3, 24, False, "RE", 1, 3],
            [5, 40, True, "RE", 2, 6],
            [5, 40, True, "RE", 1, 6],
            [5, 40, True, "RE", 1, 6],
            [3, 80, False, "HS", 2, 6],
            [3, 80, False, "HS", 1, 2.57],
            [3, 80, False, "HS", 1, 2.86],
            [3, 112, True, "HS", 1, 6],
            [3, 112, True, "HS", 1, 6],
            [5, 160, True, "HS", 2, 6],
            [5, 160, True, "HS", 1, 6],
            [5, 160, True, "HS", 1, 6]
        ]
        
        self.config_small = [
            [3, 16,  True, "RE", 2, 1],
            [3, 24,  False, "RE", 2, 4.8],
            [3, 24,  False, "RE", 1, 3.67],
            [5, 40,  True, "HS", 2, 4],
            [5, 40,  True, "HS", 1, 6],
            [5, 40,  True, "HS", 1, 6],
            [5, 48,  True, "HS", 1, 3],
            [5, 96,  True, "HS", 2, 3],
            [5, 96,  True, "HS", 1, 3],
            [5, 96,  True, "HS", 1, 6]
        ]
        input_channel = 16 
        if version == "large":
            for kernel_size, out_channel, use_se, activation, stride, expansion in self.config:
                layers.append(IRBlock(input_channel, out_channel, kernel_size, stride, expansion, use_se, activation))
                input_channel = out_channel
            layers.extend([
                nn.Conv2d(input_channel, 960, 1, bias=False),
                nn.BatchNorm2d(960),
                act_layer,
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(960, 1280, 1),
                act_layer
            ])
        else:
            for kernel_size, out_channel, use_se, activation, stride, expansion in self.config_small:
                layers.append(IRBlock(input_channel, out_channel, kernel_size, stride, expansion, use_se, activation))
                input_channel = out_channel
            layers.extend([
                nn.Conv2d(input_channel, 576, 1, bias=False),
                nn.BatchNorm2d(576),
                act_layer,
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(576, 1024, 1),
                act_layer
            ])
        
        self.block = nn.Sequential(*layers)
        if version == "large":
            self.cls = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1280, num_classes)
            )
        else:
            self.cls = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, num_classes)
            )

    def forward(self, x):
        x = self.block(x)
        return self.cls(x)

if __name__ == "__main__":
    model_v1 = MobileNetV1(num_classes=10, alpha=0.5)
    model_v2 = MobileNetV2(num_classes=10, alpha=1.0)
    model_v3 = MobileNetV3(num_classes=10, version="large")
    model_v3_sm = MobileNetV3(num_classes=10, version="small")
    x = torch.randn(1, 3, 224, 224)
    print(model_v1(x).shape)
    print(model_v2(x).shape)
    print(model_v3(x).shape)   
    print(model_v3_sm(x).shape)   
