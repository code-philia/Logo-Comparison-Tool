import torch
import torch.nn as nn
import torch.nn.functional as F

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        # weight standardization
        w = self.weight
        var, mean = torch.var_mean(w, dim=[1,2,3], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, bias=bias)

def conv3x3(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        mid = cout // 4
        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, mid)
        self.gn2 = nn.GroupNorm(32, mid)
        self.conv2 = conv3x3(mid, mid, stride=stride)
        self.gn3 = nn.GroupNorm(32, mid)
        self.conv3 = conv1x1(mid, cout)
        self.relu = nn.ReLU(inplace=True)
        # downsample if channel or spatial size changes
        self.down = conv1x1(cin, cout, stride=stride) if (stride!=1 or cin!=cout) else nn.Identity()

    def forward(self, x):
        out = self.relu(self.gn1(x))
        res = self.down(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))
        return out + res


class ResNetV2(nn.Module):
    def __init__(self, layers, width=1, num_classes=1000, zero_head=False):
        """
        :param layers: 四个 stage 的 block 数量，例如 [3,4,6,3]
        :param width: 每层通道宽度倍数
        """
        super().__init__()
        wf = width
        # stem
        self.stem = nn.Sequential(
            StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, 64*wf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # build 4 stages
        dims = [64, 256, 512, 1024, 2048]
        self.stages = nn.ModuleList()
        for i, n_blocks in enumerate(layers):
            cin = dims[i] * wf
            cout = dims[i+1] * wf
            blocks = []
            for b in range(n_blocks):
                stride = 2 if (b==0 and i>0) else 1
                blocks.append(PreActBottleneck(cin, cout, stride))
                cin = cout
            self.stages.append(nn.Sequential(*blocks))

        # head
        self.norm = nn.GroupNorm(32, dims[-1]*wf)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(dims[-1]*wf, num_classes, kernel_size=1)
        if zero_head:
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.relu(self.norm(x))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)


# —— 工厂函数：与原 BiT-M-R50x1 接口兼容 ——
def BiT_M_R50x1(num_classes=21843, zero_head=True):
    # 对应 layers=[3,4,6,3], width=1
    return ResNetV2(layers=[3,4,6,3], width=1,
                    num_classes=num_classes,
                    zero_head=zero_head)

