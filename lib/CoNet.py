import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class CIM(nn.Module):
    def __init__(self, channel):
        super(CIM, self).__init__()
        self.conv1h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(channel)
        self.conv2h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(channel)
        self.conv3h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(channel)
        self.conv4h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(channel)

        self.conv1v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(channel)
        self.conv2v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(channel)
        self.conv3v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(channel)
        self.conv4v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(channel)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)


class SK(nn.Module):
    def __init__(self, channel):
        super(SK, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class layer_SK(nn.Module):
    def __init__(self, channel):
        super(layer_SK, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(channel, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        #b0, c0, _, _ = x0.size()
        b1, c1, _, _ = x1.size()
        b2, c2, _, _ = x2.size()
        b3, c3, _, _ = x3.size()
        b4, c4, _, _ = x4.size()
        #y0 = self.avg_pool(x0).view(b0, c0)
        y1 = self.avg_pool(x1).view(b1, c1)
        y2 = self.avg_pool(x2).view(b2, c2)
        y3 = self.avg_pool(x3).view(b3, c3)
        y4 = self.avg_pool(x4).view(b4, c4)
        y = torch.cat([ y1, y2, y3, y4], dim=1)
        y = self.fc(y).view(b1, 4, 1, 1)
        x1_y = x1 * (y[:,0,0,0].resize(b1,1,1,1)).expand_as(x1)
        x2_y = x2 * (y[:, 1, 0, 0].resize(b1, 1, 1, 1)).expand_as(x2)
        x3_y = x3 * (y[:, 2, 0, 0].resize(b1, 1, 1, 1)).expand_as(x3)
        x4_y = x4 * (y[:, 3, 0, 0].resize(b1, 1, 1, 1)).expand_as(x4)
        x2_y = F.interpolate(x2_y, size=x1_y.size()[2:], mode='bilinear')
        x3_y = F.interpolate(x3_y, size=x1_y.size()[2:], mode='bilinear')
        x4_y = F.interpolate(x4_y, size=x1_y.size()[2:], mode='bilinear')
        x_g = torch.cat([x1_y, x2_y, x3_y, x4_y], dim=1)
        #print('xg_________________', x_g.size())
        x_g = self.linear(x_g)
        #print('xg____________', x_g.size())


        return x_g




class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.cim45  = CIM(channel)
        self.cim34  = CIM(channel)
        self.cim23  = CIM(channel)

    def forward(self, out2h, out3h, out4h, out5v,y, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            out4h, out4v = self.cim45(out4h+refine4, out5v)
            out3h, out3v = self.cim34(out3h+refine3, out4v)
            out2h, pred  = self.cim23(out2h+refine2, out3v)
        else:
            y_gate = torch.sigmoid(y)
            out4h, out4v = self.cim45(out4h, out5v)
            out4h = F.interpolate(out4h, scale_factor=16, mode='bilinear')
            out4h_gate = torch.sigmoid(out4h)
            out4v = F.interpolate(out4v, scale_factor=16, mode='bilinear')
            out4v_gate = torch.sigmoid(out4v)
            #print('4hgate_____________', out4h_gate.size())
            #print('4vgate_____________', out4v_gate.size())
            #print('4v_____________', out4v.size())
            #print('yg_____________', y_gate.size())
            #print('y_____________', y.size())

            out4h = (1+out4h_gate)*out4h + (1-out4h_gate)*(out4v_gate*out4v + y_gate*y)

            out3h, out3v = self.cim34(out3h, out4v)
            out3h = F.interpolate(out3h, scale_factor=8, mode='bilinear')
            out3h_gate = torch.sigmoid(out3h)
            out3v = F.interpolate(out3v, scale_factor=8, mode='bilinear')
            out3v_gate = torch.sigmoid(out3v)
            out3h = (1 + out3h_gate) * out3h + (1 - out3h_gate) * (out3v_gate * out3v + y_gate * y)
            #(32, 64, 64)
            out2h, pred  = self.cim23(out2h, out3v)
            out2h = F.interpolate(out2h, scale_factor=4, mode='bilinear')
            out2h_gate = torch.sigmoid(out2h)
            pred = F.interpolate(pred, scale_factor=4, mode='bilinear')
            pred_gate = torch.sigmoid(pred)
            out2h_fuse = (1 + out2h_gate) * out2h + (1 - out2h_gate) * (pred_gate * pred + y_gate * y)
            pred = (1 + pred_gate) * pred + (1 - pred_gate) * (out2h_gate * out2h + y_gate * y)

            #(32, 128, 128)
        #return out4h,out4v,out3h, out3v, out2h, pred
        return out2h_fuse, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)



class CoNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(CoNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb1_1 = RFB_modified(256, channel)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        #self.agg1 = aggregation(channel)

        self.decoder1 = Decoder(channel)
        #self.decoder2 = Decoder(channel)
        self.linearp1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        self.linearr6 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)


        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.sk = SK(2)
        self.sk1 = layer_SK(128)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 128, 128
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 128, 128
        x2 = self.resnet.layer2(x1)     # bs, 512, 64, 64

        x3 = self.resnet.layer3(x2)     # bs, 1024, 32, 32
        x4 = self.resnet.layer4(x3)     # bs, 2048, 16, 16

        x1_rfb = self.rfb1_1(x1)        # channel -> 32   (128, 128)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32   (64, 64)
        x3_rfb = self.rfb3_1(x3)        # channel -> 32   (32, 32)
        x4_rfb = self.rfb4_1(x4)        # channel -> 32    (16, 16)
        y = self.sk1(x1_rfb,x2_rfb,x3_rfb,x4_rfb)
        #print('y1____________', y.size())
        y = F.interpolate(y, scale_factor=4, mode='bilinear')
        #print('y____________', y.size())
        #y_gate = torch.sigmoid(y)

        #ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)


        # out4h,out4v,out3h, out3v, out2h, pred

       # out2h, out3h, out4h, out5v, pred, out3v, out4v
        out2h, out3h, out4h, out5v, ra5_feat= self.decoder1(x1_rfb, x2_rfb, x3_rfb, x4_rfb,y)  	#pred1: (bs, 32, 128, 128)
        #(32, 128, 128), (32, 64, 64), (32, 32, 32), (32, 16, 16), (32, 128, 128)
        #out2h, out3h, out4h, out5v, ra5_feat = self.decoder2(out2h, out3h, out4h, out5v, pred1)    #ra5_feat(bs, 32, 128, 128)



        #print('ra51____________', ra5_feat.size())
        #pred1 = self.linearp1(pred1)
        #pred1 = F.interpolate(pred1, scale_factor=4, mode='bilinear')
        ra5_feat = self.linearp2(ra5_feat)

        out2h = self.linearr2(out2h)
        #out2h = F.interpolate(out2h, scale_factor=4, mode='bilinear')
        out3h = self.linearr3(out3h)
        #out3h = F.interpolate(out3h, scale_factor=8, mode='bilinear')
        out4h = self.linearr4(out4h)
        #out4h = F.interpolate(out4h, scale_factor=16, mode='bilinear')
        out5v = self.linearr5(out5v)
        out5v = F.interpolate(out5v, scale_factor=32, mode='bilinear')


        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=1, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 128, 128) -> (bs, 1, 512, 512)

        # ---- reverse attention branch_4 ----
        #print('ra5__________________', ra5_feat.size())
        #print('x4______________________',x4.size())
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.03125, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)        #(a, 2048, 16, 16)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 16, 16) -> (bs, 1, 512, 512)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 32, 32) -> (bs, 1, 512, 512)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 64, 64) -> (bs, 1, 512, 512)

        fuse = torch.cat([lateral_map_5, lateral_map_2], dim=1)
        fuse = self.sk(fuse)
        fuse = self.linearr6(fuse)


        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, out2h, out3h, out4h, out5v, fuse


if __name__ == '__main__':
    ras = CoNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
