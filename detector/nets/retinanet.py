import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.anchors import Anchors
from nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# Non-local module
import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            #self.inter_channels = 0
            # 进行压缩得到 channel 数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # θ
        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        # φ
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        batch_size = x.size(0)  # 即图中的 T

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)#dilation=2
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)#dilation=2
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.non_local = _NonLocalBlockND(256)
        # self.non_local1 = _NonLocalBlockND(1024)
        # #self.non_local = _NonLocalBlockND(2048)
        # self.non_local2 = _NonLocalBlockND(2048)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        _, _, h4, w4 = C4.size()
        _, _, h3, w3 = C3.size()

        # 75,75,512 -> 75,75,256
        P3_x = self.P3_1(C3)
        P3_x = self.non_local(P3_x)

        # 38,38,1024 -> 38,38,256
        #C4 = non_local1(C4_x)
        P4_x = self.P4_1(C4)
        P4_x = self.non_local(P4_x)
        # 19,19,2048 -> 19,19,256
        #C5_x = non_local2(C5)
        P5_x = self.P5_1(C5)

        # 19,19,256 -> 38,38,256
        P5_upsampled_x = F.interpolate(P5_x, size=(h4, w4))
        # 38,38,256 + 38,38,256 -> 38,38,256
        P4_x = P5_upsampled_x + P4_x
        # 38,38,256 -> 75,75,256
        P4_upsampled_x = F.interpolate(P4_x, size=(h3, w3))
        # 75,75,256 + 75,75,256 -> 75,75,256
        P3_x = P3_x + P4_upsampled_x

        # 75,75,256 -> 75,75,256

        P3_x = self.P3_2(P3_x)
        # 38,38,256 -> 38,38,256
        P4_x = self.P4_2(P4_x)
        # 19,19,256 -> 19,19,256
        P5_x = self.P5_2(P5_x)
        # 19,19,2048 -> 10,10,256
        #C5_x = self.non_local(C5)
        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        # 10,10,256 -> 5,5,256
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=2, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class Resnet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(Resnet, self).__init__()
        self.edition = [resnet18, resnet34, resnet50, resnet101, resnet152]
        model = self.edition[phi](pretrained)
        del model.avgpool
        del model.fc
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feat1 = self.model.layer2(x)
        feat2 = self.model.layer3(feat1)
        feat3 = self.model.layer4(feat2)

        return [feat1,feat2,feat3]

class ASPP(nn.Module):
    def __init__(self, in_channel,depth,out_channel):
        #depth = in_channel
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ECABlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        x1_1 = x * self.sab(x)
        x1_2 = self.cab(x1_1)
        x2_1 = x * self.cab(x)
        x2_2 = self.cab(x2_1)
        out = x1_2 + x2_2
        return out


class retinanet(nn.Module):
    def __init__(self, num_classes, phi, pretrained=False, fp16=False):
        super(retinanet, self).__init__()
        self.pretrained = pretrained
        self.sigmoid = nn.Sigmoid()
        self.spa = SpatialAttentionBlock(256)
        #self.cha = ChannelAttentionBlock(256)
        self.cha = ECABlock(256)

        #-----------------------------------------#
        #   取出三个有效特征层，分别是C3、C4、C5
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        #   C3     75,75,512
        #   C4     38,38,1024
        #   C5     19,19,2048
        #-----------------------------------------#
        self.conv1 = nn.Conv2d(1024, 512, 1, 1)
        self.conv2 = nn.Conv2d(2048, 1024, 1, 1)
        self.backbone_net = Resnet(phi, pretrained)
        #self.backbone_net2 = Resnet(phi, pretrained)
        # self.sp_net1 = ASPP(512,4,256)
        # self.sp_net2 = ASPP(1024,4,512)
        self.conv3 = nn.Conv2d(4096, 2048, 1, 1)
        self.af_attention = AffinityAttention(256)
        self.af_attention1 = AffinityAttention(512)
        self.af_attention2 = AffinityAttention(1024)
        self.af_attention3= AffinityAttention(2048)
        #self.sp_net3 = ASPP(2048,4,1024)
        fpn_sizes = {
            0: [128, 256, 512],
            1: [128, 256, 512],
            2: [512, 1024, 2048],
            3: [512, 1024, 2048],
            4: [512, 1024, 2048],
        }[phi]

        #-----------------------------------------#
        #   经过FPN可以获得5个有效特征层分别是
        #   P3     75,75,256
        #   P4     38,38,256
        #   P5     19,19,256
        #   P6     10,10,256
        #   P7     5,5,256
        #-----------------------------------------#
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        #----------------------------------------------------------#
        #   将获取到的P3, P4, P5, P6, P7传入到
        #   Retinahead里面进行预测，获得回归预测结果和分类预测结果
        #   将所有特征层的预测结果进行堆叠
        #----------------------------------------------------------#
        self.regressionModel        = RegressionModel(256)
        self.classificationModel    = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()
        self._init_weights(fp16)

    def _init_weights(self, fp16):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        if fp16:
            self.classificationModel.output.bias.data.fill_(-2.9)
        else:
            self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

    def forward(self, inputs1,inputs2):
        #-----------------------------------------#
        #   取出三个有效特征层，分别是C3、C4、C5
        #   C3     75,75,512
        #   C4     38,38,1024
        #   C5     19,19,2048
        #-----------------------------------------#
        # if inputs1 == None and inputs2 != None:
        #     p3, p4, p5 = self.backbone_net1(inputs2)
        # if inputs2 == None and inputs1 != None:
        #     p3, p4, p5 = self.backbone_net1(inputs1)
        # if inputs1 != None and inputs2!=None:
        p3_1, p4_1, p5_1 = self.backbone_net(inputs1)
        p3_1 = self.af_attention1(p3_1)
        p4_1 = self.af_attention2(p4_1)
        p5_1 = self.af_attention3(p5_1)
        # p3_1 = self.sp_net1(p3_1)
        # p4_1 = self.sp_net2(p4_1)
        #p5_1 = self.sp_net3(p5_1)
        p3_2, p4_2, p5_2 = self.backbone_net(inputs2)
        p3_2 = self.af_attention1(p3_2)
        p4_2 = self.af_attention2(p4_2)
        p5_2 = self.af_attention3(p5_2)
        # p3_2 = self.sp_net1(p3_2)
        # p4_2 = self.sp_net2(p4_2)
        #p5_2 = self.sp_net3(p5_2)
        p3 = p3_1 + (1 - self.sigmoid(p3_1)) * p3_2
        p4 = p4_1 + (1 - self.sigmoid(p4_1)) * p4_2
        # p3 = torch.cat((p3_1, p3_2),dim=1)
        # p4 = torch.cat((p4_1, p4_2), dim=1)
        p5 = p5_1 + (1 - self.sigmoid(p5_1)) * p5_2
        #print("p5",p5.shape)
        # p3 = torch.cat((p3_1, p3_2),dim=1)
        # p4 = torch.cat((p4_1, p4_2), dim=1)
        # p5 = torch.cat((p5_1, p5_2), dim=1)
        # p5 = self.conv3(p5)
        #-----------------------------------------#
        #   使用1 x 1卷积块降维
        #-----------------------------------------#
        # p3 = self.conv1(p3)
        # p4 = self.conv2(p4)
        # p5 = self.conv3(p5)
        #-----------------------------------------#
        #   经过FPN可以获得5个有效特征层分别是
        #   P3     75,75,256
        #   P4     38,38,256
        #   P5     19,19,256
        #   P6     10,10,256
        #   P7     5,5,256
        #-----------------------------------------#
        features = self.fpn([p3, p4, p5])
        #----------------------------------------------------------#
        #   将获取到的P3, P4, P5, P6, P7传入到
        #   Retinahead里面进行预测，获得回归预测结果和分类预测结果
        #   将所有特征层的预测结果进行堆叠
        #----------------------------------------------------------#
        regression      = torch.cat([self.regressionModel(self.spa(feature)) for feature in features], dim=1)
        classification  = torch.cat([self.classificationModel(self.cha(feature)) for feature in features], dim=1)
        anchors = self.anchors(features)

        return features, regression, classification, anchors
