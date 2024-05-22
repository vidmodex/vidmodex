import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator28(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator28, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0),  # Output size: (ngf*4) x 3 x 3
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 0),  # Output size: (ngf*2) x 7 x 7
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # Output size: (ngf) x 14 x 14
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),  # Output size: (nc) x 28 x 28
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))

class Generator32(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator32, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0),  # Output size: (ngf*4) x 4 x 4
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # Output size: (ngf*2) x 8 x 8
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # Output size: (ngf) x 16 x 16
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),  # Output size: (nc) x 32 x 32
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))
    
class Generator64(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator64, self).__init__()
        self.nz = 100
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),  # Output size: (ngf*8) x 4 x 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),  # Output size: (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # Output size: (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # Output size: (ngf) x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),  # Output size: (nc) x 64 x 64
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))
    
    
class Generator128(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator128, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),  # Output size: (ngf*8) x 4 x 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),  # Output size: (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # Output size: (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # Output size: (ngf) x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1),  # Output size: (ngf/2) x 64 x 64
            nn.BatchNorm2d(int(ngf // 2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(ngf // 2), nc, 4, 2, 1),  # Output size: (nc) x 128 x 128
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))

class Generator224(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator224, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0),  # Output size: (ngf*16) x 4 x 4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),  # Output size: (ngf*8) x 8 x 8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),  # Output size: (ngf*4) x 16 x 16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # Output size: (ngf*2) x 32 x 32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # Output size: (ngf) x 64 x 64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1),  # Output size: (ngf/2) x 128 x 128
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1),  # Output size: (nc) x 256 x 256
            nn.AdaptiveAvgPool2d((224, 224)),  # Output size: (nc) x 224 x 224
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))


class Generator256(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator256, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0),  # Output size: (ngf*16) x 4 x 4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),  # Output size: (ngf*8) x 8 x 8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),  # Output size: (ngf*4) x 16 x 16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # Output size: (ngf*2) x 32 x 32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),  # Output size: (ngf) x 64 x 64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1),  # Output size: (ngf/2) x 128 x 128
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1),  # Output size: (nc) x 256 x 256
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))


class CondGenerator28(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, num_classes=10):
        super(CondGenerator28, self).__init__()
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.label_emb = nn.Embedding(self.num_classes, self.nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2 * self.nz, self.ngf * 4, 3, 1, 0),  # Output size: (self.ngf*4) x 3 x 3
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2,  3, 2, 0),  # Output size: (self.ngf*2) x 7 x 7
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),  # Output size: (self.ngf) x 14 x 14
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),  # Output size: (nc) x 28 x 28
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = x.view(-1, 2 * self.nz, 1, 1)
        return self.main(x)

class CondGenerator32(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=10):
        super(CondGenerator32, self).__init__()
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.label_emb = nn.Embedding(self.num_classes, self.nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2 * self.nz, self.ngf * 4, 4, 1, 0),  # Output size: (self.ngf*4) x 4 x 4
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),  # Output size: (self.ngf*2) x 8 x 8
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),  # Output size: (self.ngf) x 16 x 16
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),  # Output size: (nc) x 32 x 32
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = x.view(-1, 2 * self.nz, 1, 1)
        return self.main(x)
    
class CondGenerator64(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=100):
        super(CondGenerator64, self).__init__()
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.label_emb = nn.Embedding(self.num_classes, self.nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2 * self.nz, self.ngf * 8, 4, 1, 0),  # Output size: (self.ngf*8) x 4 x 4
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),  # Output size: (self.ngf*4) x 8 x 8
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),  # Output size: (self.ngf*2) x 16 x 16
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),  # Output size: (self.ngf) x 32 x 32
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),  # Output size: (nc) x 64 x 64
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = x.view(-1, 2 * self.nz, 1, 1)
        return self.main(x)
    
    
class CondGenerator128(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=100):
        super(CondGenerator128, self).__init__()
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.label_emb = nn.Embedding(self.num_classes, self.nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2 * self.nz, self.ngf * 8, 4, 1, 0),  # Output size: (self.ngf*8) x 4 x 4
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),  # Output size: (self.ngf*4) x 8 x 8
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),  # Output size: (self.ngf*2) x 16 x 16
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),  # Output size: (self.ngf) x 32 x 32
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.ngf // 2, 4, 2, 1),  # Output size: (self.ngf/2) x 64 x 64
            nn.BatchNorm2d(int(self.ngf // 2)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(self.ngf // 2), self.nc, 4, 2, 1),  # Output size: (nc) x 128 x 128
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = x.view(-1, 2 * self.nz, 1, 1)
        return self.main(x)

class CondGenerator224(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=100):
        super(CondGenerator224, self).__init__()
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.label_emb = nn.Embedding(self.num_classes, self.nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2 * self.nz, self.ngf * 16, 4, 1, 0),  # Output size: (self.ngf*16) x 4 x 4
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1),  # Output size: (self.ngf*8) x 8 x 8
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),  # Output size: (self.ngf*4) x 16 x 16
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),  # Output size: (self.ngf*2) x 32 x 32
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),  # Output size: (self.ngf) x 64 x 64
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.ngf // 2, 4, 2, 1),  # Output size: (self.ngf/2) x 128 x 128
            nn.BatchNorm2d(self.ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf // 2, self.nc, 4, 2, 1),  # Output size: (self.nc) x 256 x 256
            nn.AdaptiveAvgPool2d((224, 224)),  # Output size: (nc) x 224 x 224
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = x.view(-1, 2 * self.nz, 1, 1)
        return self.main(x)


class CondGenerator256(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=1000):
        super(CondGenerator256, self).__init__()
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.label_emb = nn.Embedding(self.num_classes, self.nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2 * self.nz, self.ngf * 16, 4, 1, 0),  # Output size: (self.ngf*16) x 4 x 4
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1),  # Output size: (self.ngf*8) x 8 x 8
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),  # Output size: (self.ngf*4) x 16 x 16
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),  # Output size: (self.ngf*2) x 32 x 32
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),  # Output size: (self.ngf) x 64 x 64
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, self.ngf // 2, 4, 2, 1),  # Output size: (self.ngf/2) x 128 x 128
            nn.BatchNorm2d(self.ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf // 2, self.nc, 4, 2, 1),  # Output size: (nc) x 256 x 256
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = x.view(-1, 2 * self.nz, 1, 1)
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, img_size=64):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.img_size = img_size

        layers = self.calculate_layers(img_size)
        self.main = nn.Sequential(*layers)

    def calculate_layers(self, img_size):
        assert img_size in [64, 128, 224, 256], "Unsupported image size"
        ngf_mult = 8
        layers = [
            # Input is the latent vector Z.
            nn.ConvTranspose2d(self.nz, self.ngf * ngf_mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * ngf_mult),
            nn.ReLU(True),
        ]

        # Additional layers are added depending on image size
        if img_size == 64:
            num_upsamples = 3
        elif img_size == 128:
            num_upsamples = 4
        elif img_size == 224 or img_size == 256:
            num_upsamples = 5

        
        while num_upsamples > 1:
            layers.append(
                nn.ConvTranspose2d(self.ngf * ngf_mult, self.ngf * (ngf_mult // 2), 4, 2, 1, bias=False)
            )
            layers.append(nn.BatchNorm2d(self.ngf * (ngf_mult // 2)))
            layers.append(nn.ReLU(True))
            ngf_mult //= 2
            num_upsamples -= 1

        # Final layer
        layers.append(nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        return layers

    def forward(self, input):
        output = self.main(input.view(-1, self.nz, 1, 1))
        return output

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, img_size=64):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.img_size = img_size

        layers = self.calculate_layers(img_size)
        self.main = nn.Sequential(*layers)

    def calculate_layers(self, img_size):
        assert img_size in [64, 128, 224, 256], "Unsupported image size"
        layers = [
            # Input is image.
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ndf_mult = 1
        while img_size > 4:
            img_size //= 2
            next_mult = min(8, ndf_mult * 2)
            layers.append(nn.Conv2d(self.ndf * ndf_mult, self.ndf * next_mult, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(self.ndf * next_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ndf_mult = next_mult

        # Final layer
        layers.append(nn.Conv2d(self.ndf * ndf_mult, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        return layers

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

if __name__ == "__main__":
    nz = 100
    img_size = 128
    generator = Generator(nz=nz, ngf=64, nc=3, img_size=img_size)
    discriminator = Discriminator(ndf=64, nc=3, img_size=img_size)
    gen_28 = Generator28(nz=nz, nc=1)
    gen_32 = Generator32(nz=nz, nc=3)
    gen_64 = Generator64(nz=nz, nc=3)
    gen_128 = Generator128(nz=nz, nc=3)
    gen_224 = Generator224(nz=nz, nc=3)
    gen_256 = Generator256(nz=nz, nc=3)
    cond_gen_28 = CondGenerator28(nz=nz, nc=1, num_classes=100)
    cond_gen_32 = CondGenerator32(nz=nz, nc=3, num_classes=100)
    cond_gen_64 = CondGenerator64(nz=nz, nc=3, num_classes=100)
    cond_gen_128 = CondGenerator128(nz=nz, nc=3, num_classes=100)
    cond_gen_224 = CondGenerator224(nz=nz, nc=3, num_classes=100)
    cond_gen_256 = CondGenerator256(nz=nz, nc=3, num_classes=100)
    
    bs = 10
    z = torch.rand(10, nz)
    label = torch.randint(0, 100, (bs,))
    
    # assert generator(z).shape == (bs, 3, img_size, img_size)
    # assert discriminator(generator(z)).shape == (bs,)
    assert gen_28(z).shape == (bs, 1, 28, 28)
    assert gen_32(z).shape == (bs, 3, 32, 32)
    assert gen_64(z).shape == (bs, 3, 64, 64)
    assert gen_128(z).shape == (bs, 3, 128, 128)
    assert gen_224(z).shape == (bs, 3, 224, 224)
    assert gen_256(z).shape == (bs, 3, 256, 256)
    assert cond_gen_28(z, label).shape == (bs, 1, 28, 28)
    assert cond_gen_32(z, label).shape == (bs, 3, 32, 32)
    assert cond_gen_64(z, label).shape == (bs, 3, 64, 64)
    assert cond_gen_128(z, label).shape == (bs, 3, 128, 128)
    assert cond_gen_224(z, label).shape == (bs, 3, 224, 224)
    assert cond_gen_256(z, label).shape == (bs, 3, 256, 256)
    print("All tests passed!")
