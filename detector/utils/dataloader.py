import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class RetinanetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(RetinanetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image1,image2, box  = self.get_random_data(self.annotation_lines[index],self.input_shape, random = self.train)
        image1       = np.transpose(preprocess_input(np.array(image1, dtype=np.float32)),(2,0,1))
        image2       = np.transpose(preprocess_input(np.array(image2, dtype=np.float32)),(2,0,1))
        box          = np.array(box, dtype=np.float32)
        return image1,image2, box
        
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image1   = Image.open(line[0])
        image1   = cvtColor(image1)
        image2 = Image.open(line[1])
        image2 = cvtColor(image2)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image1.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image1       = image1.resize((nw,nh), Image.BICUBIC)
            image2       = image2.resize((nw,nh), Image.BICUBIC)
            new_image1   = Image.new('RGB', (w,h), (128,128,128))
            new_image1.paste(image1, (dx, dy))
            new_image2 = Image.new('RGB', (w, h), (128, 128, 128))
            new_image2.paste(image2, (dx, dy))
            image_data1  = np.array(new_image1, np.float32)
            image_data2  = np.array(new_image2, np.float32)
            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data1,image_data2, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image1 = image1.resize((nw,nh), Image.BICUBIC)
        image2 = image2.resize((nw, nh), Image.BICUBIC)
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image1 = Image.new('RGB', (w,h), (128,128,128))
        new_image1.paste(image1, (dx, dy))
        image1 = new_image1
        new_image2 = Image.new('RGB', (w, h), (128, 128, 128))
        new_image2.paste(image2, (dx, dy))
        image2 = new_image2
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)

        image_data1      = np.array(image1, np.uint8)
        image_data2 = np.array(image2, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue1, sat1, val1   = cv2.split(cv2.cvtColor(image_data1, cv2.COLOR_RGB2HSV))
        dtype1           = image_data1.dtype
        hue2, sat2, val2 = cv2.split(cv2.cvtColor(image_data2, cv2.COLOR_RGB2HSV))
        dtype2 = image_data2.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x * r[0]) % 180).astype(dtype1)
        lut_sat1 = np.clip(x * r[1], 0, 255).astype(dtype1)
        lut_val1 = np.clip(x * r[2], 0, 255).astype(dtype1)
        lut_hue2 = ((x * r[0]) % 180).astype(dtype2)
        lut_sat2 = np.clip(x * r[1], 0, 255).astype(dtype2)
        lut_val2 = np.clip(x * r[2], 0, 255).astype(dtype2)
        image_data1 = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        image_data2 = cv2.merge((cv2.LUT(hue2, lut_hue2), cv2.LUT(sat2, lut_sat2), cv2.LUT(val2, lut_val2)))
        image_data1 = cv2.cvtColor(image_data1, cv2.COLOR_HSV2RGB)
        image_data2 = cv2.cvtColor(image_data2, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data1,image_data2, box

# DataLoader中collate_fn使用
def retinanet_dataset_collate(batch):
    images1 = []
    images2 = []
    bboxes = []
    for img1,img2, box in batch:
        images1.append(img1)
        images2.append(img2)
        bboxes.append(box)
    images1 = torch.from_numpy(np.array(images1)).type(torch.FloatTensor)
    images2 = torch.from_numpy(np.array(images2)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images1, images2,bboxes


