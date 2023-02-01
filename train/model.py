import torch
import torch.nn as nn
input_c = 3 # = bf + gf, depend on your dataset

class CNNLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv=nn.Conv2d(in_c,
                      out_c,
                      3,
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      bias=False,
                      padding_mode='zeros')
        self.bn= nn.BatchNorm2d(out_c)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = torch.relu(y)
        return y


class ResnetLayer(nn.Module):
    def __init__(self, inout_c, mid_c):
        super().__init__()
        self.conv_net = nn.Sequential(
            CNNLayer(inout_c, mid_c),
            CNNLayer(mid_c, inout_c)
        )

    def forward(self, x):
        x = self.conv_net(x) + x
        return x

class OutputHead(nn.Module):

    def __init__(self,out_c,head_mid_c):
        super().__init__()
        self.cnn=CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 3)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1)

    def forward(self, h):
        x=self.cnn(h)

        # value head
        value = x.mean((2, 3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = self.policyHeadLinear(x)
        policy = policy.squeeze(1)

        return value, policy



class Model_ResNet(nn.Module):
    def __init__(self,blocks,channels):
        super().__init__()
        self.model_type = "resnet"
        self.model_param=(blocks,channels)

        self.inputhead=CNNLayer(input_c, channels)
        self.trunk=nn.ModuleList()
        for i in range(blocks):
            self.trunk.append(ResnetLayer(channels,channels))
        self.outputhead=OutputHead(channels,channels)

    def forward(self, x):
        h=self.inputhead(x)

        for block in self.trunk:
            h=block(h)

        return self.outputhead(h)


ModelDic = {
    "resnet": Model_ResNet,
}
