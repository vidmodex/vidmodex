from random import lognormvariate
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalShapDiscriminator2D(nn.Module):
    def __init__(self, n_channels, out_channels, n_classes, width_multiplier=1, trilinear=True, prob_out=True, use_ds_conv=False):
        """A simple 2D Unet, adapted from the originally provided 3D version.
        Arguments:
          n_channels: number of input channels; 3 for RGB, 1 for grayscale input
          out_channels: number of output channels
          n_classes: number of classes for conditional batch norm
          width_multiplier: scales the number of channels in each layer of the U-Net
          trilinear: use bilinear interpolation to upsample; if false, 2D convtranspose layers will be used instead
          prob_out: if True, outputs probabilities; otherwise outputs mean and logvar
          use_ds_conv: if True, use depthwise-separable convolutions"""
        super(ConditionalShapDiscriminator2D, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.prob_out = prob_out
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv2d if use_ds_conv else nn.Conv2d
        self.n_classes = n_classes
        
        self.cbn = ConditionalBatchNorm2d(self.channels[0], self.n_classes)
        
        self.inc = DoubleConv2D(2 * n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down2D(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down2D(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down2D(self.channels[2], self.channels[3], conv_type=self.convtype)
        factor = 2 if trilinear else 1
        self.down4 = Down2D(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.up1 = Up2D(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = Up2D(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up2D(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up2D(self.channels[1], self.channels[0], trilinear)
        if prob_out:
            self.outc = OutConvProb2D(self.channels[0], self.channels[0] // 2, out_channels)
        else:
            self.outc = OutConv2D(self.channels[0], out_channels)
        
    def forward(self, x, cls_idx): 
        x_bn = self.cbn(x, cls_idx)
        x = torch.cat((x, x_bn), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.prob_out:
            logits = self.outc(x)
            return logits
        else:
            mu, logvar = self.outc(x)
            return mu, logvar

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, prob_out=True, use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.prob_out = prob_out
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv3D(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down3D(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down3D(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down3D(self.channels[2], self.channels[3], conv_type=self.convtype)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.up1 = Up3D(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = Up3D(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up3D(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up3D(self.channels[1], self.channels[0], trilinear)
        if prob_out:
            self.outc = OutConvProb3D(self.channels[0], self.channels[0]//2, n_classes)
        else:
            self.outc = OutConv3D(self.channels[0], n_classes)
        #self.constant_multiplier = nn.Parameter(torch.tensor([init_constant or 1.0]))
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.prob_out:
            logits = self.outc(x)
            return logits
        else:
            mu, logvar = self.outc(x)
            return mu, logvar
        # out = self.outc(x)
        # #logits = torch.exp( -1*self.constant_multiplier) * x
        # return logits

class ConditionalShapDiscriminator3D(nn.Module):
    def __init__(self, n_channels, out_channels, n_classes, width_multiplier=1, trilinear=True, prob_out=True, use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          out_channels = number of output channels
          n_classes = number of classes for conditional batch norm
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(ConditionalShapDiscriminator3D, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.prob_out = prob_out
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d
        self.n_classes = n_classes
        
        self.cbn = ConditionalBatchNorm3d(self.channels[0], self.n_classes, self.n_channels)
        
        self.inc = DoubleConv3D(2*n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        factor = 2 if trilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear)
        if prob_out:
            self.outc = OutConvProb3D(self.channels[0], self.channels[0]//2, out_channels)
        else:
            self.outc = OutConv3D(self.channels[0], out_channels)
        #self.constant_multiplier = nn.Parameter(torch.tensor([init_constant or 1.0]))
        
    def forward(self, x, cls_idx): 
        x_bn = self.cbn(x, cls_idx)
        x = torch.concat((x, x_bn), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.prob_out:
            logits = self.outc(x)
            return logits
        else:
            mu, logvar = self.outc(x)
            return mu, logvar

class ConditionalBatchNorm3d(nn.Module):
    def __init__(self, num_features, num_classes, num_output_channels=3):
        super().__init__()
        self.num_features = num_features
        self.num_output_channels = num_output_channels
        self.bn = nn.BatchNorm3d(num_output_channels)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1)
        self.embed.weight.data[:, num_features:].zero_()
        self.fc1_gamma = nn.Linear(num_features, num_features//2)
        self.fc1_beta = nn.Linear(num_features, num_features//2)
        self.fc2_gamma = nn.Linear(num_features//2, num_output_channels)
        self.fc2_beta = nn.Linear(num_features//2, num_output_channels)

    def forward(self, x, class_id):
        out = self.bn(x)
        gamma_init, beta_init = self.embed(class_id).chunk(2, 1)
        
        gamma = F.tanh(self.fc1_gamma(gamma_init))
        beta = F.tanh(self.fc1_beta(beta_init))
        gamma = self.fc2_gamma(gamma)
        beta = self.fc2_beta(beta)
        
        out = gamma.view(-1, self.num_output_channels, 1, 1, 1) * out + beta.view(-1, self.num_output_channels, 1, 1, 1)
        return out

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, num_output_channels):
        super().__init__()
        self.num_features = num_features
        self.num_output_channels = num_output_channels
        self.bn = nn.BatchNorm2d(num_output_channels)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1)
        self.embed.weight.data[:, num_features:].zero_()

        self.fc1_gamma = nn.Linear(num_features, num_features // 2)
        self.fc1_beta = nn.Linear(num_features, num_features // 2)
        self.fc2_gamma = nn.Linear(num_features // 2, num_output_channels)
        self.fc2_beta = nn.Linear(num_features // 2, num_output_channels)

    def forward(self, x, class_id):
        out = self.bn(x)
        gamma_init, beta_init = self.embed(class_id).chunk(2, 1)

        gamma = F.tanh(self.fc1_gamma(gamma_init))
        beta = F.tanh(self.fc1_beta(beta_init))
        gamma = self.fc2_gamma(gamma)
        beta = self.fc2_beta(beta)

        out = gamma.view(-1, self.num_output_channels, 1, 1) * out + beta.view(-1, self.num_output_channels, 1, 1)
        return out

class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv2d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv2d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels,  out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class OutConvProb3D(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(OutConvProb3D, self).__init__()
        self.mu1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.mu2 = nn.Conv3d(inter_channels, out_channels, kernel_size=1)
        self.logvar1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.logvar2 = nn.Conv3d(inter_channels, out_channels, kernel_size=1)
        

    def forward(self, x):
        mu1 = F.relu(self.mu1(x))
        mu = self.mu2(mu1)
        logvar1 = F.relu(self.logvar1(x))
        logvar = self.logvar2(logvar1)
        
        return mu, logvar

class OutConvProb2D(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(OutConvProb2D, self).__init__()
        self.mu1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.mu2 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)
        self.logvar1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.logvar2 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)

    def forward(self, x):
        mu1 = F.relu(self.mu1(x))
        mu = self.mu2(mu1)
        logvar1 = F.relu(self.logvar1(x))
        logvar = self.logvar2(logvar1)
        
        return mu, logvar

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    
class VideoAutoencoder(nn.Module):
    def __init__(self):
        super(VideoAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2), # Reducing to half
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2), # Reducing to half again
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2), # Reducing to half, again
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2), # Reducing to half, again
            # This reduces each spatial dimension by 8 times overall
        )

        # Shared Decoder Layers
        self.shared_decoder_layers = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # After these layers, the dimensions are back to the original size
        )

        # Separate Decoder Paths for mu and sigma
        self.decoder_mu = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Ensures output is in the range [-1, 1]
        )

        self.decoder_logvar = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Ensures output is non-negative
        )

    def forward(self, x):
        encoded = self.encoder(x)
        shared_decoded = self.shared_decoder_layers(encoded)
        mu = self.decoder_mu(shared_decoded)
        logvar = self.decoder_logvar(shared_decoded)
        return mu, logvar