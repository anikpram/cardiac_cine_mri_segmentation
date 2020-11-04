import torch
import torch.nn as nn
import numpy as np



def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def conv1x1(in_channels, out_channels):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1
    )




class DsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pooling):
        super(DsBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling
        if pooling:
            self.mp = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        out = self.conv(x)
        out = self.relu(out)
        before_pool = out
        if self.pooling:
            out = self.mp(out)

        return out, before_pool




def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='trilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class UsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UsBlock, self).__init__()

        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)
        self.conv = conv3x3(2*out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, before_pool, x):
        x = self.upconv(x)
        x = torch.cat((x, before_pool), dim=1)
        x = self.conv(x)
        x = self.relu(x)
        return x

class UsBlockRes(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UsBlockRes, self).__init__()

        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)
        self.conv = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, before_pool, x):
        x = self.upconv(x)
        x = x + before_pool
        x = self.conv(x)
        x = self.relu(x)
        return x



class UNet3D(nn.Module):
    '''
    This class implements a 3D Unet with global and local residual connections. The input construction arguments: num_classes, 
    in_channels, depth, start_filter number, upsampling mode.
    '''

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose', res=False):
        super(UNet3D, self).__init__()

        self.down_convs = []
        self.up_convs = []

        # put one conv  at the beginning
        self.conv_start = conv3x3(in_channels, start_filts, stride=1)
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = start_filts * (2 ** i)
            outs = start_filts * (2 ** (i + 1))
            pooling = True if i < depth - 1 else False

            down_conv = DsBlock(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            if res:
                up_conv = UsBlockRes(ins, outs, up_mode=up_mode)
                self.up_convs.append(up_conv)
            else:
                up_conv = UsBlock(ins, outs, up_mode=up_mode)
                self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        x = self.conv_start(x)
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        # print(len(encoder_outs))
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x





if __name__ == "__main__":

    module = UNet3D(num_classes=4, in_channels=1, depth=4, start_filts=1, up_mode="trilinear")
    x = torch.FloatTensor(np.random.random((2,1,144,144,8)))
    y = module(x)
    print(module)
    print(y.size())
    