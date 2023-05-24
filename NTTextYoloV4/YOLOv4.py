import numpy as np
import torch
import torch.nn as nn

from CSPDarknet53 import CSPDarkNet53
from Neck import SpatialPyramidPoolingPANet

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class Yolo_head(nn.Module):
    def __init__(self, num_class, anchors, stride):
        super(Yolo_head, self).__init__()

        self.anchors = anchors 
        self.num_anchor = len(anchors)
        self.num_class = num_class
        self.stride = stride
        self.isTesting = False

    def forward(self, p):
        batch_size, output_size = p.shape[0], p.shape[-1]
        p = p.view(batch_size, self.num_anchor, 5 + self.num_class, output_size, output_size).permute(0, 3, 4, 1, 2)

        p_decoded = self.__decode(p.clone())

        return p, p_decoded

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.stride
        anchors = (1.0 * self.anchors).to(device)

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = \
            torch.split(p, [2, 2, 1, self.num_class], dim=-1)

        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        grid_xy = torch.stack((x, y), dim=2) # [output_size, output_size, 2]
        grid_xy = (
            grid_xy.unsqueeze(0) # [1, output_size, output_size, 2]
            .unsqueeze(3) # [1, output_size, output_size, 1, 2]
            .repeat(batch_size, 1, 1, 3, 1) # [batch_size, output_size, output_size, 3, 2]
            .float()
            .to(device)
        )

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        # pred_bbox = torch.cat([pred_conf, pred_xywh, pred_prob], dim=-1)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return (
            pred_bbox.view(-1, 5 + self.num_class)
            if self.isTesting
            else pred_bbox
        )
        # return pred_bbox

class YOLOV4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # anchors = [
        #     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        #     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        #     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        # ] 
        # self.anchors = torch.FloatTensor([[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
        #     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
        #     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]])
        self.anchors = torch.FloatTensor([
            [[1.5, 2.], [2.375, 4.5], [5., 3.5]],
            [[2.25, 4.6875], [4.75, 3.4375], [4.5, 9.125]],
            [[4.4375, 3.4375], [6., 7.59375], [14.34375, 12.53125]]]
        )
        self.strides = np.array([8, 16, 32])
        self.backbone = CSPDarkNet53()
        self.spatialPyramidPANNet = SpatialPyramidPoolingPANet(num_classes)
        self.large13YOLOHead = Yolo_head(num_classes, self.anchors[2], self.strides[2])
        self.mid26YOLOHead = Yolo_head(num_classes, self.anchors[1], self.strides[1])
        self.small52YOLOHead = Yolo_head(num_classes, self.anchors[0], self.strides[0])
        self.isTesting = False

    def setIsTesting(self, isTesting):
        self.isTesting = isTesting
        self.large13YOLOHead.isTesting = isTesting
        self.mid26YOLOHead.isTesting = isTesting
        self.small52YOLOHead.isTesting = isTesting
        
    def forward(self, x):
        out = []

        x1, x2, x3 = self.backbone(x)
        x1, x2, x3 = self.spatialPyramidPANNet(x1, x2, x3)

        out.append(self.small52YOLOHead(x1))
        out.append(self.mid26YOLOHead(x2))
        out.append(self.large13YOLOHead(x3))

        p, p_d = list(zip(*out))

        return (p, p_d) if not self.isTesting else torch.cat(p_d, dim=0)

if __name__ == "__main__":
    anchors = torch.FloatTensor([[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]])
    
    net = SpatialPyramidPoolingPANet(1)

    large13YOLOHead = Yolo_head(1, anchors[2], 32)
    mid26YOLOHead = Yolo_head(1,anchors[1], 16)
    small52YOLOHead = Yolo_head(1, anchors[0], 8)

    x1, x2, x3 = torch.rand((1, 256, 52, 52)), torch.rand((1, 512, 26, 26)), torch.rand((1, 1024, 13, 13))
    print("Input Shape:", x1.shape, x2.shape, x3.shape)
    # Input Shape: torch.Size([1, 256, 52, 52]) torch.Size([1, 512, 26, 26]) torch.Size([1, 1024, 13, 13])
    x1, x2, x3 = net.forward(x1, x2, x3)
    print("Output Shape:", x1.shape, x2.shape, x3.shape)
    # Output Shape: torch.Size([1, 18, 52, 52]) torch.Size([1, 18, 26, 26]) torch.Size([1, 18, 13, 13])

    x1, x1_d = small52YOLOHead.forward(x1)
    x2, x2_d = mid26YOLOHead.forward(x2)
    x3, x3_d = large13YOLOHead.forward(x3)
    print("Output Shape:", x1.shape, x2.shape, x3.shape)
    # Output Shape: torch.Size([1, 52, 52, 3, 6]) torch.Size([1, 26, 26, 3, 6]) torch.Size([1, 13, 13, 3, 6])
    print("Output Shape:", x1_d.shape, x2_d.shape, x3_d.shape)
    # Output Shape: torch.Size([1, 52, 52, 3, 6]) torch.Size([1, 26, 26, 3, 6]) torch.Size([1, 13, 13, 3, 6])
