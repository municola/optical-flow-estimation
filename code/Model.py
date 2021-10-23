import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from spatial_correlation_sampler import spatial_correlation_sample
import numpy as np
import torchvision.models as models
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    'pwc_dc_net', 'pwc_dc_net_old'
    ]

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential( nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True), nn.LeakyReLU(0.1) )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def batch(num_features):
    return nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

def correlate(input1, input2):
    '''
    Collate dimensions 1 and 2 in order to be treated as a regular 4D tensor
    see equation 1 in the following paper for more details about the correlation function
    -> link : https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf

    iter1:  input  = [1, 196, 10, 10]
            output = [1, 81, 10, 10]

    iter2:  input  = [1, 128, 20, 20]
            output = [1, 81, 20, 20]

    iter3:  input  = [1, 96, 40, 40]
            output = [1, 81, 40, 40]

    iter4:  input  = [1, 64, 80, 80]
            output = [1, 81, 80, 80]

    iter5:  input  = [1, 32, 160, 160]
            output = [1, 81, 160, 160]

    '''

    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=9,
                                          stride=1)
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return out_corr


"""
Implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""
class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4, pretrained=False, ours=False):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet,self).__init__()
        self.ours = ours
        self.upsample =  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout(0.2)


        # Constructing the two feature pyramids
        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.corr    = correlate # Correlation(padding=md, kernel_size=1, patch_size=md, stride=1)
        # self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        # md = maximum displacement (default = 4)
        # nd = area of maximum displacement (left, right, up, down + central pixel)

        nd = (2*md+1)**2 # 81
        dd = np.cumsum([128,128,96,64,32]) # [128, 256, 352, 416, 448]

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1) # conv(81,128)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1) # conv(341,128)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1) # conv(337,96)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1) # conv(433,64)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1) # conv(497,32)
        self.predict_flow6 = predict_flow(od+dd[4]) # predict_flow(529)
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) # deconv(529,2)

        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1) # conv(213,128)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1) # conv(469,128)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1) # conv(469,96)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1) # conv(565,64)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1) # conv(629,32)
        self.predict_flow5 = predict_flow(od+dd[4]) # predict_flow(661)
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) # deconv(661,2)

        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        if pretrained:
            self.load_state_dict('models/pwc_net_chairs.pth.tar')

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        if x.is_cuda:
            mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        else:
            mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, x):
        # x = [1, 6, 640, 640]

        im1 = x[:,:3,:,:] # [1, 3, 640, 640]
        im2 = x[:,3:6,:,:] # [1, 3, 640, 640]

        seg2 = x[:,9:,:,:] # only imported for prediction of segmentation mask 1

        # Construct the two feature pyramids
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1))) # [1, 16, 320, 320]
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2))) # [1, 16, 320, 320]
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11))) # [1, 32, 160, 160]
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21))) # [1, 32, 160, 160]
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12))) # [1, 64, 80, 80]
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22))) # [1, 64, 80, 80]
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13))) # [1, 96, 40, 40]
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23))) # [1, 96, 40, 40]
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14))) # [1, 128, 20, 20]
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24))) # [1, 128, 20, 20]
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15))) # [1, 196, 10, 10]
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25))) # [1, 196, 10, 10]

        # Cost Volume for 6th layer
        corr6 = self.corr(c16, c26) # [1, 81, 10, 10]
        corr6 = self.leakyRELU(corr6) # [1, 81, 10, 10]
        corr6 = self.dropout(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6),1) # [1, 209, 10, 10]
        x = torch.cat((self.conv6_1(x), x),1) # [1, 337, 10, 10]
        x = torch.cat((self.conv6_2(x), x),1) # [1, 433, 10, 10]
        x = torch.cat((self.conv6_3(x), x),1) # [1, 497, 10, 10]
        x = torch.cat((self.conv6_4(x), x),1) # [1, 529, 10, 10]

        # Optical Flow Estimator for 6th layer
        flow6 = self.predict_flow6(x) # [1, 2, 10, 10]
        up_flow6 = self.deconv6(flow6) # [1, 2, 20, 20]
        up_feat6 = self.upfeat6(x) # [1, 2, 20, 20]

        # Warping for 5th layer
        warp5 = self.warp(c25, up_flow6*0.625) # [1, 128, 20, 20]

        # Cost Volume for 5th layer
        corr5 = self.corr(c15, warp5) # [1, 81, 20, 20]
        corr5 = self.leakyRELU(corr5) # [1, 81, 20, 20]
        corr5 = self.dropout(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1) # [1, 213, 20, 20]
        x = torch.cat((self.conv5_0(x), x),1) # [1, 341, 20, 20]
        x = torch.cat((self.conv5_1(x), x),1) # [1, 469, 20, 20]
        x = torch.cat((self.conv5_2(x), x),1) # [1, 565, 20, 20]
        x = torch.cat((self.conv5_3(x), x),1) # [1, 629, 20, 20]
        x = torch.cat((self.conv5_4(x), x),1) # [1, 661, 20, 20]

        # Optical Flow Estimator for 5th layer
        flow5 = self.predict_flow5(x) # [1, 2, 20, 20]
        up_flow5 = self.deconv5(flow5) # [1, 2, 40, 40]
        up_feat5 = self.upfeat5(x) # [1, 2, 40, 40]

        # Warping for 4th layer
        warp4 = self.warp(c24, up_flow5*1.25) # [1, 96, 40, 40]

        # Cost Volume for 4th layer
        corr4 = self.corr(c14, warp4) # [1, 81, 40, 40]
        corr4 = self.leakyRELU(corr4) # [1, 81, 40, 40]
        corr4 = self.dropout(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1) # [1, 181, 40, 40]
        x = torch.cat((self.conv4_0(x), x),1) # [1, 309, 40, 40]
        x = torch.cat((self.conv4_1(x), x),1) # [1, 437, 40, 40]
        x = torch.cat((self.conv4_2(x), x),1) # [1, 533, 40, 40]
        x = torch.cat((self.conv4_3(x), x),1) # [1, 597, 40, 40]
        x = torch.cat((self.conv4_4(x), x),1) # [1, 629, 40, 40]

        # Optical Flow Estimator for 4th layer
        flow4 = self.predict_flow4(x) # [1, 2, 40, 40]
        up_flow4 = self.deconv4(flow4) # [1, 2, 80, 80]
        up_feat4 = self.upfeat4(x) # [1, 2, 80, 80]

        # Warping for 3rd layer
        warp3 = self.warp(c23, up_flow4*2.5) # [1, 64, 80, 80]

        # Cost Volume for 3rd layer
        corr3 = self.corr(c13, warp3) # [1, 81, 80, 80]
        corr3 = self.leakyRELU(corr3) # [1, 81, 80, 80]
        corr3 = self.dropout(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1) # [1, 149, 80, 80]
        x = torch.cat((self.conv3_0(x), x),1) # [1, 277, 80, 80]
        x = torch.cat((self.conv3_1(x), x),1) # [1, 405, 80, 80]
        x = torch.cat((self.conv3_2(x), x),1) # [1, 501, 80, 80]
        x = torch.cat((self.conv3_3(x), x),1) # [1, 565, 80, 80]
        x = torch.cat((self.conv3_4(x), x),1) # [1, 597, 80, 80]

        # Optical Flow Estimator for 3rd layer
        flow3 = self.predict_flow3(x) # [1, 2, 80, 80]
        up_flow3 = self.deconv3(flow3) # [1, 2, 160, 160]
        up_feat3 = self.upfeat3(x) # [1, 2, 160, 160]

        # Warp for 2nd layer
        warp2 = self.warp(c22, up_flow3*5.0) # [1, 32, 160, 160]

        # Cost Volume for 2nd layer
        corr2 = self.corr(c12, warp2) # [1, 81, 160, 160]
        corr2 = self.leakyRELU(corr2) # [1, 81, 160, 160]
        corr2 = self.dropout(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1) # [1, 117, 160, 160]
        x = torch.cat((self.conv2_0(x), x),1) # [1, 245, 160, 160]
        x = torch.cat((self.conv2_1(x), x),1) # [1, 373, 160, 160]
        x = torch.cat((self.conv2_2(x), x),1) # [1, 469, 160, 160]
        x = torch.cat((self.conv2_3(x), x),1) # [1, 533, 160, 160]
        x = torch.cat((self.conv2_4(x), x),1) # [1, 565, 160, 160]

        # Optical flow Estimator 2nd layer
        flow2 = self.predict_flow2(x) # [1, 2, 160, 160]
        contextNetwork = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x)))) # [1, 96, 160, 160]
        contextNetwork = self.dropout(contextNetwork)
        contextNetwork = self.dc_conv7(self.dc_conv6(self.dc_conv5(contextNetwork))) # [1, 2, 160, 160]
        flowFinal = flow2 + contextNetwork # [1, 2, 160, 160]

        # flow2 = 20*4*self.upsample(flow2)
        # flow3 = 20*4*self.upsample(flow3)
        # flow4 = 20*4*self.upsample(flow4)
        # flow5 = 20*4*self.upsample(flow5)
        # flow6 = 20*4*self.upsample(flow6)

        upscaled_flow = 20*self.upsample(flowFinal)
        predicted_im1 = self.warp(im2, upscaled_flow)
        predicted_seg1 = self.warp(seg2, upscaled_flow)

        # Stack the features (for our own model)
        if self.ours:
            return flowFinal, flow3, flow4, flow5, flow6

        if self.training:
            return flowFinal, flow3, flow4, flow5, flow6, predicted_im1, predicted_seg1
        else:
            return flowFinal # [1, 2, 160, 160]


"""
Implementation of our SegNet that runs together with the PWC-Net
"""
class SegNet(nn.Module):
    def __init__(self, data, freeze):
        super(SegNet,self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

        self.batch_resnet = batch(512)

        self.deconv1 = deconv(512, 32, kernel_size=7, stride=1, padding=1)
        self.deconv2 = deconv(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = deconv(16, 8, kernel_size=4, stride=2, padding=1)
        self.deconv4 = deconv(8, 4, kernel_size=4, stride=2, padding=1)
        self.deconv5 = deconv(4, 2, kernel_size=4, stride=2, padding=1)

        self.conv6_S = conv(115, 2, kernel_size=3, stride=1)
        self.deconv6_S = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.batch6 = batch(81)

        self.conv5_S = conv(101, 2, kernel_size=3, stride=1)
        self.deconv5_S = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.batch5 = batch(81)

        self.conv4_S = conv(93, 2, kernel_size=3, stride=1)
        self.deconv4_S = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.batch4 = batch(81)

        self.conv3_S = conv(89, 2, kernel_size=3, stride=1)
        self.deconv3_S = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.batch3 = batch(81)

        self.batch2 = batch(81)
        self.predict_finalFlow = predict_flow(87)

        print('Freeze resnet weights')
        for p in self.resnet18.parameters():
            p.requires_grad = False

        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pwcModel = pwc(data, ours=True)
        self.pwcModelWithoutTop = pwcModel

        if freeze:
            print('Freeze pwc-net weights')
            for p in self.pwcModelWithoutTop.parameters():
                p.requires_grad = False

        self.corr = correlate

        self.upsample =  nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout(0.1)
        self.leakyRELU = nn.LeakyReLU(0.1)

    def unfreeze(self):
        print('\nUnfreeze pwc-net weights')
        for p in self.pwcModelWithoutTop.parameters():
            p.requires_grad = True

    def preprocessingIMG(self, img): # PWC expects a BGR input, but ResNet expects RGB
        # imagesPreprocessed = [self.preprocess(x_.cpu()) for x_ in img]
        # xPreprocessed = torch.cat([x_.unsqueeze(dim=0) for x_ in imagesPreprocessed], dim=0)
        xPreprocessed = torch.cat([x_.unsqueeze(dim=0) for x_ in img], dim=0)
        permute = [2,1,0] # b g r -> r g b
        imgRGB = xPreprocessed[:,permute,:,:]

        return imgRGB

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        if x.is_cuda:
            mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        else:
            mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask

    def forward(self,x):
        # Create the features from the pwcNet
        finalFlow, flow3, flow4, flow5, flow6 = self.pwcModelWithoutTop(x)

        # Multiply each image with its respective segmentation mask
        # We focus exclusively on the region of interest (the bodies), and attribute different body parts different colors
        im1 = x[:,:3,:,:] * x[:,6:9,:,:]
        im2 = x[:,3:6,:,:] * x[:,9:,:,:]

        real_im2 = x[:,3:6,:,:]
        seg2 = x[:,9:,:,:] # only imported for prediction of segmentation mask 1

        # Preprocessing. Inception expects RGB normalized
        im1_preprocessed = self.preprocessingIMG(im1).to(device)
        im2_preprocessed = self.preprocessingIMG(im2).to(device)

        '''
        Visualizations
        
        ## to visualize the input (image 1 and 2) and segmentations masks of image 1 and 2
        input1 = self.preprocessingIMG(x[:,:3,:,:]).to(device)
        input2 = self.preprocessingIMG(x[:,3:6,:,:]).to(device)

        input1_mask = im1_preprocessed
        input2_mask = im2_preprocessed

        seg_mask1 = self.preprocessingIMG(x[:,6:9,:,:]).to(device)
        seg_mask2 = self.preprocessingIMG(x[:,9:,:,:]).to(device)

        input1 = np.transpose(input1[0].detach().cpu().numpy(), (1, 2, 0)) # image 1
        input2 = np.transpose(input2[0].detach().cpu().numpy(), (1, 2, 0)) # image 2
        input1_mask = np.transpose(input1_mask[0].detach().cpu().numpy(), (1, 2, 0)) # image 1 masked
        input2_mask = np.transpose(input2_mask[0].detach().cpu().numpy(), (1, 2, 0)) # image 2 masked
        seg_mask1 = np.transpose(seg_mask1[0].detach().cpu().numpy(), (1, 2, 0)) # segmentation 1
        seg_mask2 = np.transpose(seg_mask2[0].detach().cpu().numpy(), (1, 2, 0)) # segmentation 2

        img1 = Image.fromarray((input1 * 255).astype(np.uint8))
        img1_mask = Image.fromarray((input1_mask * 255).astype(np.uint8))
        img2 = Image.fromarray((input2 * 255).astype(np.uint8))
        img2_mask = Image.fromarray((input2_mask * 255).astype(np.uint8))
        seg1 = Image.fromarray((seg_mask1 * 255).astype(np.uint8))
        seg2 = Image.fromarray((seg_mask2 * 255).astype(np.uint8))

        img1.show('image 1')
        img1_mask.show('image 1 masked')
        img2.show('image 2')
        img2_mask.show('image 2 masked')
        seg1.show('segmentation mask 1')
        seg2.show('segmentation mask 2')
        '''

        reshape_dim = flow6.shape[2:]
        # Apply inception-net
        f1 = self.resnet18(im1) # [1, 512, 1, 1]
        f1 = self.batch_resnet(f1)
        # f1 = self.dropout(f1)
        f1 = self.deconv1(f1) # [1, 32, 5, 5]
        f1a = F.interpolate(f1, size=reshape_dim) # [1, 32, 10, 10]
        f1b = self.deconv2(f1a) # [1, 16, 20, 20]
        f1c = self.deconv3(f1b) # [1, 8, 40, 40]
        f1d = self.deconv4(f1c) # [1, 4, 80, 80]
        f1e = self.deconv5(f1d) # [1, 2, 160, 160]

        f2 = self.resnet18(im2) # [1, 512, 1, 1]
        f2 = self.batch_resnet(f2)
        # f2 = self.dropout(f2)
        f2 = self.deconv1(f2) # [1, 32, 5, 5]
        f2a = F.interpolate(f2, size=reshape_dim) # [1, 32, 10, 10]
        f2b = self.deconv2(f2a) # [1, 16, 20, 20]
        f2c = self.deconv3(f2b) # [1, 8, 40, 40]
        f2d = self.deconv4(f2c) # [1, 4, 80, 80]
        f2e = self.deconv5(f2d) # [1, 2, 160, 160]

        # Sub-pixel refinement for 6th layer
        warp6_S = self.warp(f2a, flow6*0.625) # [1, 32, 10, 10]
        corr6_S = self.corr(f1a, warp6_S) # [1, 81, 10, 10]
        corr6_S = self.batch6(corr6_S)
        corr6_S = self.leakyRELU(corr6_S) # [1, 81, 10, 10]
        # corr6_S = self.dropout(corr6_S)
        x = torch.cat((corr6_S, f1a, flow6),1) # [1, 115, 10, 10]
        flow6_S = self.conv6_S(x) # [1, 2, 10, 10]
        up_flow6 = self.deconv6_S(flow6_S) # [1, 2, 20, 20]

        # Sub-pixel refinement for 5th layer
        warp5_S = self.warp(f2b, up_flow6*1.25) # [1, 16, 20, 20]
        corr5_S = self.corr(f1b, warp5_S) # [1, 81, 20, 20]
        corr5_S = self.batch5(corr5_S)
        corr5_S = self.leakyRELU(corr5_S) # [1, 81, 20, 20]
        # corr5_S = self.dropout(corr5_S)
        x = torch.cat((corr5_S, f1b, flow5, up_flow6),1) # [1, 101, 20, 20]
        flow5_S = self.conv5_S(x) # [1, 2, 20, 20]
        up_flow5 = self.deconv5_S(flow5_S) # [1, 2, 40, 40]

        # Sub-pixel refinement for 4th layer
        warp4_S = self.warp(f2c, up_flow5*2.5) # [1, 8, 40, 40]
        corr4_S = self.corr(f1c, warp4_S) # [1, 81, 40, 40]
        corr4_S = self.batch4(corr4_S)
        corr4_S = self.leakyRELU(corr4_S) # [1, 81, 40, 40]
        # corr4_S = self.dropout(corr4_S)
        x = torch.cat((corr4_S, f1c, flow4, up_flow5),1) # [1, 93, 40, 40]
        flow4_S = self.conv4_S(x) # [1, 2, 40, 40]
        up_flow4 = self.deconv4_S(flow4_S) # [1, 2, 80, 80]

        # Sub-pixel refinement for 3rd layer
        warp3_S = self.warp(f2d, up_flow4*5.0) # [1, 4, 80, 80]
        corr3_S = self.corr(f1d, warp3_S) # [1, 81, 80, 80]
        corr3_S = self.batch3(corr3_S)
        corr3_S = self.leakyRELU(corr3_S) # [1, 81, 80, 80]
        # corr3_S = self.dropout(corr3_S)
        x = torch.cat((corr3_S, f1d, flow3, up_flow4),1) # [1, 89, 80, 80]
        flow3_S = self.conv3_S(x) # [1, 2, 80, 80]
        up_flow3 = self.deconv3_S(flow3_S) # [1, 2, 160, 160]

        # Sub-pixel refinement for 2nd layer (stop here, already desired dimensions)
        corr2_S = self.corr(f1e, f2e) # [1, 81, 160, 160]
        corr2_S = self.batch2(corr2_S)
        corr2_S = self.leakyRELU(corr2_S) # [1, 81, 160, 160]
        # corr2_S = self.dropout(corr2_S)
        x = torch.cat((corr2_S, f1e, finalFlow, up_flow3),1) # [1, 87, 160, 160]
        our_finalFlow = self.predict_finalFlow(x) # [1, 2, 160, 160]

        upscaled_flow = 20*self.upsample(our_finalFlow)
        our_predicted_im1 = self.warp(real_im2, upscaled_flow)
        our_predicted_seg1 = self.warp(seg2, upscaled_flow)

        if self.training:
            return our_finalFlow, flow3_S, flow4_S, flow5_S, flow6_S, our_predicted_im1, our_predicted_seg1
        else:
            return our_finalFlow


def pwc(data=None, ours=False):

    model = PWCDCNet(ours=ours)
    if data is not None:
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model

def pwc_dc_net(path=None):

    model = PWCDCNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model


'''
This method creats our own model "SegNet"
The "our_data" field would be the weights if we have pretrained it already.
If freeze is True, then the pwc-net (submodel in our net) gets its weights frozen
'''

def seg_net(data=None, our_data=None, freeze=False):

    # Init our own Net with pwc as a subnet
    model = SegNet(data=data, freeze=freeze)

    # If we have already trained the new weights of our own model and we now want to fine-tune it.
    # We therefore load our own pretrained model
    if our_data is not None:
        if 'state_dict' in our_data.keys():
            model.load_state_dict(our_data['state_dict'])
        else:
            model.load_state_dict(our_data)
    return model
    
