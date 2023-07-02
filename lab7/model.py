import torch
import torch.nn as nn


class ConvblockRes(nn.Module):
    def __init__(self, in_c, out_c, used_residual=False):
        super().__init__()
        self.same_channels = False
        if in_c == out_c:
            self.same_channels = True    
        self.used_residual = used_residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        if self.used_residual:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class embedding(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(embedding, self).__init__()

        self.in_dim = in_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        return self.fc(x.float()).reshape(-1, self.emb_dim, 1, 1)


class DownConvUnet(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownConvUnet, self).__init__()
        self.upconv = nn.Sequential(ConvblockRes(in_c, out_c), nn.MaxPool2d(2))
    def forward(self, x):
        return self.upconv(x)


class UpConvUnet(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpConvUnet, self).__init__()

        self.downconv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, 2),
            ConvblockRes(out_c, out_c),
            ConvblockRes(out_c, out_c),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.downconv(x)
        return x




class Unet(nn.Module):
    def __init__(self, in_c=3, out_c=3, classes=24):
        super(Unet, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.initial = ConvblockRes(in_c, 256, used_residual=True)

        self.down1 = DownConvUnet(256, 256)
        self.down2 = DownConvUnet(256, 512)

        self.h = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.layert1 = embedding(1, 512)
        self.layert2 = embedding(1, 256)
        self.layerc1 = embedding(classes, 512)
        self.layerc2 = embedding(classes, 256)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 8, 8),
            nn.GroupNorm(8, 512),
            nn.ReLU(),
        )

        self.up1 = UpConvUnet(1024, 256)
        self.up2 = UpConvUnet(512, 256)
        self.out = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, self.out_c, 3, 1, 1),
        )

    def forward(self, x, t, c):  
        label1 = self.layerc1(c)
        time1 = self.layert1(t)
        label2 = self.layerc2(c)
        time2 = self.layert2(t)
        # f = label2 + time2

        x = self.initial(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden = self.h(down2)
      
        conv = self.conv(hidden)
        up1 = self.up1(label1 * conv + time1 , down2)
        up2 = self.up2(label2 * up1 + time2, down1)
        out = self.out(torch.cat((up2, x), 1))
        return out


