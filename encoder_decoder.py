from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
arg
s = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.conv2(out)

        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out


class UNet(nn.Module):
    def __init__(self, imsize):
        super(UNet, self).__init__()
        self.imsize = imsize
        self.conv2_drop = nn.Dropout2d()
        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        # self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(1, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        # self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, 2, 1)

    def forward(self, x):

        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        # pool4 = self.pool4(block4)
        block5=self.conv2_drop(block4)
        # block5 = self.conv_block512_1024(pool4)

        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block5, block4)

        up3 = self.up_block256_128(up2, block3)

        up4 = self.up_block128_64(up3, block2)

        return F.log_softmax(self.last(up4))
