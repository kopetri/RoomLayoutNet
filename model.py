import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import functools


ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]


def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )


'''
Encoder
'''
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4


class Densenet(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        assert backbone in ENCODER_DENSENET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        del self.encoder.classifier

    def forward(self, x):
        lst = []
        for m in self.encoder.features.children():
            x = m(x)
            lst.append(x)
        features = [lst[4], lst[6], lst[8], self.final_relu(lst[11])]
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.features.children()]
        block0 = lst[:4]
        block1 = lst[4:6]
        block2 = lst[6:8]
        block3 = lst[8:10]
        block4 = lst[10:]
        return block0, block1, block2, block3, block4


class Conv2dH(nn.Module):
    def __init__(self, in_c, out_c, ks, ring_conv):
        super().__init__()
        self.ring_conv = ring_conv
        assert ks % 2 == 1
        if ring_conv:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=(ks//2, 0))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2)

    def forward(self, x):
        if self.ring_conv:
            x = F.pad(x,(1,1,0,0), mode = 'circular')
        x = self.conv(x)
        return x

'''
Decoder
'''
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ring_conv, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.conv = Conv2dH(in_c, out_c, ks, ring_conv)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c, use_ring_conv):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c,    in_c//2, use_ring_conv),
            ConvCompressH(in_c//2, in_c//2, use_ring_conv),
            ConvCompressH(in_c//2, in_c//4, use_ring_conv),
            ConvCompressH(in_c//4, out_c,   use_ring_conv),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x

class SingleGlobalHeightConv(nn.Module):
    def __init__(self, in_c, use_ring_conv):
        super(SingleGlobalHeightConv, self).__init__()
        self.layer = ConvCompressH(in_c, 1, use_ring_conv)

    def forward(self, x, out_w):
        x = self.layer(x)
        assert out_w % x.shape[3] == 0, "{} % {} == 0".format(out_w, x.shape[3])
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x.squeeze(1).permute(0, 2, 1)

class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, use_ring_conv, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale, use_ring_conv),
            GlobalHeightConv(c2, c2//out_scale, use_ring_conv),
            GlobalHeightConv(c3, c3//out_scale, use_ring_conv),
            GlobalHeightConv(c4, c4//out_scale, use_ring_conv),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature

'''
HorizonNet
'''
class HorizonNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, use_rnn, gan_c, use_ring_conv=False):
        super(HorizonNet, self).__init__()
        self.backbone = backbone
        self.use_rnn = use_rnn
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512

        # Encoder
        if backbone.startswith('res'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        else:
            raise NotImplementedError()

        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.out_scale

        # Convert gan features
        self.reduce_gan = SingleGlobalHeightConv(gan_c, use_ring_conv)
        self.reduce      = nn.MaxPool1d(2, padding=0)
        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, use_ring_conv, self.out_scale)

        # 1D prediction
        if self.use_rnn:
            self.bi_rnn = nn.LSTM(input_size=c_last,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(0.5)
            self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                    out_features=3 * self.step_cols)
            self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear.bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear.bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        else:
            self.linear = nn.Sequential(
                nn.Linear(c_last, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.rnn_hidden_size, 3 * self.step_cols),
            )
            self.linear[-1].bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear[-1].bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear[-1].bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x, feat):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()
        x = self._prepare_x(x)
        conv_list = self.feature_extractor(x)
        feature = self.reduce_height_module(conv_list, x.shape[3]//self.step_cols)
        feature_g = self.reduce_gan(feat, feat.shape[3])

        feature = torch.cat([feature, feature_g], dim=2)
        feature = self.reduce(feature)

        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        return bon, cor

def bn_act_drop(num_channels, dropout):
    if dropout > 0:
        return nn.Sequential(nn.BatchNorm1d(num_channels), nn.ReLU(), nn.Dropout(dropout))
    else:
        return nn.Sequential(nn.BatchNorm1d(num_channels), nn.ReLU())

class MLP(nn.Module):
    def __init__(self, dropout, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features)
        self.norm = bn_act_drop(out_features, dropout)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        return x

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = functools.partial(Conv2dAuto, kernel_size=3, bias=False)  

def conv_bn(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(conv3x3(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

def conv_bn_relu(in_channels, out_channels, downsampling, *args, **kwargs):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride=downsampling, *args, **kwargs), nn.BatchNorm2d(out_channels), nn.ReLU())

class EncoderSimple(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()
        self.block1 = conv_bn_relu(3,                 latent_size // 8,   downsampling=2)
        self.block2 = conv_bn_relu(latent_size // 8,  latent_size // 4,   downsampling=2)
        self.block3 = conv_bn_relu(latent_size // 4,  latent_size // 2,   downsampling=2)
        self.block4 = conv_bn_relu(latent_size // 2,  latent_size,        downsampling=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return x.reshape(self._shape)

class PixelCornerNetwork(nn.Module):
    def __init__(self, img_size, encoder="simple", latent_size=128, dropout=0.5, n_features=[128, 32], activation=None):
        super().__init__()
        if activation == 'tanh':
            last_layer = torch.nn.Tanh()
        elif activation == 'relu':
            last_layer = torch.nn.ReLU()
        elif activation == 'sigmoid':
            last_layer = torch.nn.Sigmoid()
        else:
            last_layer = torch.nn.Identity()

        if encoder == 'simple':
            self.encoder = EncoderSimple(latent_size=latent_size)
        elif encoder in ENCODER_RESNET:
            self.encoder = Resnet(backbone=encoder)
        elif encoder in ENCODER_DENSENET:
            self.encoder = Densenet(backbone=encoder)
        else:
            raise NotImplementedError()
        
        _, latent_size_half, dim0, dim1 = self.encoder(torch.rand((1,3,img_size[1], img_size[0]))).shape

        latent_size = 2 * latent_size_half

        print(latent_size, dim0, dim1)

        self.decoder = nn.Sequential(
            torch.nn.AvgPool2d(dim0),
            Reshape((-1, latent_size)),
            bn_act_drop(latent_size, dropout),
            MLP(dropout=dropout, in_features=latent_size, out_features=n_features[0]),
            MLP(dropout=0.0, in_features=n_features[0],   out_features=n_features[1]),
            nn.Linear(n_features[1], 8 * 2),
            Reshape((-1, 8, 2)),
            last_layer
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    network = PixelCornerNetwork(img_size=(512, 1024), activation='sigmoid').cuda().eval()

    img = torch.rand((1, 3, 512, 1024)).cuda()

    pred = network(img)

    print(pred)
    
