import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from tqdm.notebook import tqdm
from PIL import Image

class DeepImagePrior(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 3
        output = 8

        self.down_conv_1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.down_bn_1 = nn.BatchNorm2d(8)
        
        self.down_conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.down_bn_2 = nn.BatchNorm2d(16)

        self.down_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.down_bn_3 = nn.BatchNorm2d(32)
        self.skip_conv_3 = nn.Conv2d(32, 4, 5, stride=1, padding=2)

        self.down_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.down_bn_4 = nn.BatchNorm2d(64)
        self.skip_conv_4 = nn.Conv2d(64, 4, 5, stride=1, padding=2)
        
        self.down_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.down_bn_5 = nn.BatchNorm2d(128)
        self.skip_conv_5 = nn.Conv2d(128, 4, 5, stride=1, padding=2)

        self.down_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.down_bn_6 = nn.BatchNorm2d(256)

        self.up_deconv_5 = nn.ConvTranspose2d(256, (128-4), 4, stride=2, padding=1)
        self.up_bn_5 = nn.BatchNorm2d(128)

        self.up_deconv_4 = nn.ConvTranspose2d(128, (64-4), 4, stride=2, padding=1)
        self.up_bn_4 = nn.BatchNorm2d(64)

        self.up_deconv_3 = nn.ConvTranspose2d(64, (32-4), 4, stride=2, padding=1)
        self.up_bn_3 = nn.BatchNorm2d(32)

        self.up_deconv_2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.up_bn_2 = nn.BatchNorm2d(16)

        self.up_deconv_1 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.up_bn_1 = nn.BatchNorm2d(8)

        self.out_deconv = nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)
        self.out_bn = nn.BatchNorm2d(3)
    def forward(self, noise):
        
        down_1 = self.down_conv_1(noise)
        down_1 = self.down_bn_1(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.down_conv_2(down_1)
        down_2 = self.down_bn_2(down_2)
        down_2 = F.leaky_relu(down_2)
        
        down_3 = self.down_conv_3(down_2)
        down_3 = self.down_bn_3(down_3)
        down_3 = F.leaky_relu(down_3)
        skip_3 = self.skip_conv_3(down_3)
        
        down_4 = self.down_conv_4(down_3)
        down_4 = self.down_bn_4(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.skip_conv_4(down_4)
        
        down_5 = self.down_conv_5(down_4)
        down_5 = self.down_bn_5(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.skip_conv_5(down_5)
        
        down_6 = self.down_conv_6(down_5)
        down_6 = self.down_bn_6(down_6)
        down_6 = F.leaky_relu(down_6)
        
        up_5 = self.up_deconv_5(down_6)
        up_5 = torch.cat([up_5, skip_5],1)
        up_5 = self.up_bn_5(up_5)
        up_5 = F.leaky_relu(up_5)
    
        up_4 = self.up_deconv_4(up_5)
        up_4 = torch.cat([up_4, skip_4],1)
        up_4 = self.up_bn_4(up_4)
        up_4 = F.leaky_relu(up_4)
        
        up_3 = self.up_deconv_3(up_4)
        up_3 = torch.cat([up_3, skip_3],1)
        up_3 = self.up_bn_3(up_3)
        up_3 = F.leaky_relu(up_3)
        
        up_2 = self.up_deconv_2(up_3)
        up_2 = self.up_bn_2(up_2)
        up_2 = F.leaky_relu(up_2)
        
        up_1 = self.up_deconv_1(up_2)
        up_1 = self.up_bn_1(up_1)
        up_1 = F.leaky_relu(up_1)
        
        output = self.out_deconv(up_1)
        output = self.out_bn(output)
        output = F.sigmoid(output)
        
        return output
        
        