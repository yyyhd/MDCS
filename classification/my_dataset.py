from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path1: list, images_path2: list, images_path3: list, images_path4: list, images_class: list, images_name1: list, images_name2: list, images_name3: list, images_name4: list, transform=None):
        self.images_path1 = images_path1
        self.images_path2 = images_path2
        self.images_path3 = images_path3
        self.images_path4 = images_path4
        self.images_class = images_class
        self.images_name1 = images_name1
        self.images_name2 = images_name2
        self.images_name3 = images_name3
        self.images_name4 = images_name4
        self.transform = transform

    def __len__(self):
        return len(self.images_path1)

    def __getitem__(self, item):
        img1 = Image.open(self.images_path1[item])
        img2 = Image.open(self.images_path2[item])
        img3 = Image.open(self.images_path3[item])
        img4 = Image.open(self.images_path4[item])
        # RGB为彩色图片，L为灰度图片
        if img1.mode != 'RGB':
            raise ValueError("image1: {} isn't RGB mode.".format(self.images_path1[item]))
        label = self.images_class[item]
        image_name1 = self.images_name1[item]
        image_name2 = self.images_name2[item]
        image_name3 = self.images_name3[item]
        image_name4 = self.images_name4[item]
        if image_name1 != image_name2:
            raise ValueError("加载双分支数据不同")
        if image_name1 != image_name3:
            raise ValueError("加载双视图数据不同")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        return img1, img2, img3, img4, label, image_name1

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images1, images2,images3, images4, labels, image_names = tuple(zip(*batch))
        # print(images)
        # print(labels)
        # print(image_names)

        images1 = torch.stack(images1, dim=0)
        images2 = torch.stack(images2, dim=0)
        images3 = torch.stack(images3, dim=0)
        images4 = torch.stack(images4, dim=0)
        labels = torch.as_tensor(labels)
        image_names = image_names

        return images1, images2, images3, images4, labels, image_names



