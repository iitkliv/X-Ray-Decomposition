
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class UNetDec(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super(UNetDec,self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(UNetEnc,self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]

        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet,self).__init__()

        self.enc1 = UNetEnc(1, 64)
        self.enc2 = UNetEnc(64, 128)
        self.enc3 = UNetEnc(128, 256)

        self.center = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(512, 256, 2),
            # nn.ReLU(inplace=True),
        )

        self.dec3 = UNetDec(512, 256, 128)
        self.dec2 = UNetDec(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 192, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)
        self.recon = nn.Conv2d(num_classes,1,1)
        self.drr=None
        self.reconstruc=None
        self.loss=None

    def forward(self, x,a=None,b=None,c=None):

        enc1 = self.enc1(x)

        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)

        dec3 = self.dec3(torch.cat([
            center, F.upsample_bilinear(enc3, center.size()[2:])], 1))
        dec2 = self.dec2(torch.cat([
            dec3, F.upsample_bilinear(enc2, dec3.size()[2:])], 1))
        dec1 = self.dec1(torch.cat([
            dec2, F.upsample_bilinear(enc1, dec2.size()[2:])], 1))

        self.drr= F.upsample_bilinear(self.final(dec1), x.size()[2:])
        self.reconstruc = self.recon(self.drr)




        if self.training:
            self.loss= self.build_loss(x,a,b,c)
        return self.reconstruc,self.drr
    #
    def build_loss(self,x,a,b,c):

        drr1,drr2,drr3=torch.split(self.drr,1,1)
        l2loss = nn.MSELoss()
        # l1loss = torch.abs(drr1 - a)+torch.abs(drr2 - b)+torch.abs(drr3 - c)
        output_l2_loss = l2loss(drr1, a)+l2loss(drr2, b)+l2loss(drr3, c)
        l1loss= nn.L1Loss()
        output_l1_loss = l1loss(drr1, a) + l1loss(drr2, b) + l1loss(drr3, c)

        loss_decom=output_l1_loss*0.2+output_l2_loss*0.8
        loss_recon=l2loss(self.reconstruc,x)
        loss = loss_decom*0.5+loss_recon*0.5
        return loss

#
#
# if __name__ == "__main__":
#     """
#         testing
#     """
#     model = UNet(3)
#     print model
#     x = Variable(torch.FloatTensor(np.random.random(( 1,1, 320, 320))))
#     recon,drr = model(x,x,x,x)
#     print drr.size()
#     loss = torch.sum(drr)
#     print loss
#     loss.backward()
