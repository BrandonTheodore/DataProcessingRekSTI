import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SASNet(nn.Module):
    def __init__(self, pretrained=False, args=None, block_size=8):
        super(SASNet, self).__init__()
        vgg = models.vgg16_bn(pretrained=pretrained)

        features = list(vgg.features.children())
        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])
        self.features5 = nn.Sequential(*features[33:43])
        self.de_pred5 = nn.Sequential(
            Conv2d(512, 1024, 3, same_padding=True, NL='relu'),
            Conv2d(1024, 512, 3, same_padding=True, NL='relu'),
        )

        self.de_pred4 = nn.Sequential(
            Conv2d(512 + 512, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 256, 3, same_padding=True, NL='relu'),
        )

        self.de_pred3 = nn.Sequential(
            Conv2d(256 + 256, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 128, 3, same_padding=True, NL='relu'),
        )

        self.de_pred2 = nn.Sequential(
            Conv2d(128 + 128, 128, 3, same_padding=True, NL='relu'),
            Conv2d(128, 64, 3, same_padding=True, NL='relu'),
        )

        self.de_pred1 = nn.Sequential(
            Conv2d(64 + 64, 64, 3, same_padding=True, NL='relu'),
            Conv2d(64, 64, 3, same_padding=True, NL='relu'),
        )
        self.density_head5 = nn.Sequential(
            MultiBranchModule(512),
            Conv2d(2048, 1, 1, same_padding=True)
        )

        self.density_head4 = nn.Sequential(
            MultiBranchModule(256),
            Conv2d(1024, 1, 1, same_padding=True)
        )

        self.density_head3 = nn.Sequential(
            MultiBranchModule(128),
            Conv2d(512, 1, 1, same_padding=True)
        )

        self.density_head2 = nn.Sequential(
            MultiBranchModule(64),
            Conv2d(256, 1, 1, same_padding=True)
        )

        self.density_head1 = nn.Sequential(
            MultiBranchModule(64),
            Conv2d(256, 1, 1, same_padding=True)
        )
        self.confidence_head5 = nn.Sequential(
            Conv2d(512, 256, 1, same_padding=True, NL='relu'),
            Conv2d(256, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head4 = nn.Sequential(
            Conv2d(256, 128, 1, same_padding=True, NL='relu'),
            Conv2d(128, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head3 = nn.Sequential(
            Conv2d(128, 64, 1, same_padding=True, NL='relu'),
            Conv2d(64, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head2 = nn.Sequential(
            Conv2d(64, 32, 1, same_padding=True, NL='relu'),
            Conv2d(32, 1, 1, same_padding=True, NL=None)
        )

        self.confidence_head1 = nn.Sequential(
            Conv2d(64, 32, 1, same_padding=True, NL='relu'),
            Conv2d(32, 1, 1, same_padding=True, NL=None)
        )

        if args is not None and hasattr(args, 'block_size'):
            self.block_size = args.block_size
        else:
            self.block_size = block_size
    
    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        size = x.size()
    
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
    
        x = self.de_pred5(x5); x5_out = x
        x = F.interpolate(x, size=x4.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x4, x], 1)
        x = self.de_pred4(x); x4_out = x
    
        x = F.interpolate(x, size=x3.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x3, x], 1)
        x = self.de_pred3(x); x3_out = x
    
        x = F.interpolate(x, size=x2.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x2, x], 1)
        x = self.de_pred2(x); x2_out = x
    
        x = F.interpolate(x, size=x1.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x1, x], 1)
        x = self.de_pred1(x); x1_out = x
    
        densities = [self.density_head5(x5_out),
                     self.density_head4(x4_out),
                     self.density_head3(x3_out),
                     self.density_head2(x2_out),
                     self.density_head1(x1_out)]
    
        confidences = [F.adaptive_avg_pool2d(out, output_size=(size[-2] // self.block_size, size[-1] // self.block_size))
                       for out in [x5_out, x4_out, x3_out, x2_out, x1_out]]
        confidences = [head(c) for head, c in zip(
            [self.confidence_head5, self.confidence_head4, self.confidence_head3, self.confidence_head2, self.confidence_head1],
            confidences)]
    
        densities = [F.interpolate(d, size=x1.size()[2:], mode="nearest") for d in densities]
        confidences = [F.interpolate(c, size=x1.size()[2:], mode="nearest") for c in confidences]
    
        confidence_map = torch.cat(confidences, 1)
        confidence_map = torch.softmax(confidence_map, dim=1)
    
        density_map = torch.cat(densities, 1)
        density_map = density_map * confidence_map
        density = torch.sum(density_map, 1, keepdim=True)
    
        density = F.interpolate(density, size=(64, 64), mode="bilinear", align_corners=False)
    
        return density


class MultiBranchModule(nn.Module):
    def __init__(self, in_channels, sync=False):
        super(MultiBranchModule, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch1x1_1 = BasicConv2d(in_channels//2, in_channels, kernel_size=1, sync=sync)

        self.branch3x3_1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch3x3_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=(3, 3), padding=(1, 1), sync=sync)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, in_channels//2, kernel_size=1, sync=sync)
        self.branch3x3dbl_2 = BasicConv2d(in_channels // 2, in_channels, kernel_size=5, padding=2, sync=sync)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.branch1x1_1(branch1x1)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        outputs = [branch1x1, branch3x3, branch3x3dbl, x]
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, sync=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        if sync:
            print('use sync inception')
            self.bn = nn.SyncBatchNorm(out_channels, eps=0.001)
        else:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
 

model = SASNet()
model.load_state_dict(torch.load("final_model_sasnet_scratch.pth", map_location="cpu", weights_only=False))
model.eval()


def predict_count(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    return float(output.sum().item())