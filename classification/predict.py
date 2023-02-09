import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import swin_tiny_patch4_window7_224 as create_model
from PIL import Image
import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        return img2, img2, img3, img4, label


    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images1, images2, images3, images4, labels, image_names = tuple(zip(*batch))
        # print(images)
        # print(labels)
        # print(image_names)

        images1 = torch.stack(images1, dim=0)
        images2 = torch.stack(images2, dim=0)
        images3 = torch.stack(images3, dim=0)
        images4 = torch.stack(images4, dim=0)
        labels = torch.as_tensor(labels)
        image_names = image_names

        return images1, images2, images3, images4, labels

def load_net():
    net = torch.load('/data/cyq/moxing/classification/resnet/double/IAs/model-149.pth', map_location=device)
    return net

def read_split_data(root1: str, root2: str, root3: str, root4: str):
    assert os.path.exists(root1), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root1) if os.path.isdir(os.path.join(root1, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path1 = []
    images_path2 = []
    images_path3 = []
    images_path4 = []
    images_name = []# 存储训练集的所有图片路径
    images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []     # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path1 = os.path.join(root1, cla)
        cla_path2 = os.path.join(root2, cla)
        cla_path3 = os.path.join(root3, cla)
        cla_path4 = os.path.join(root4, cla)
        # 遍历获取supported支持的所有文件路径
        images1 = [os.path.join(cla_path1, i) for i in os.listdir(cla_path1)
                  if os.path.splitext(i)[-1] in supported]
        images2 = [os.path.join(cla_path2, i) for i in os.listdir(cla_path2)
                   if os.path.splitext(i)[-1] in supported]
        images3 = [os.path.join(cla_path3, i) for i in os.listdir(cla_path3)
                   if os.path.splitext(i)[-1] in supported]
        images4 = [os.path.join(cla_path4, i) for i in os.listdir(cla_path4)
                   if os.path.splitext(i)[-1] in supported]

        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images1))

        # 按比例随机采样验证样本
        for img_path1 in images1:
            image_name1 = os.path.basename(img_path1)[:-4]
            image_name.append(image_name1)
            images_path1.append(img_path1)
            images_label.append(image_class)
            for img_path2 in images2:
                image_name2 = os.path.basename(img_path2)[:-4]
                if image_name2 == image_name1:
                    images_path2.append(img_path2)
                else:
                    print("出错啦！！！！")
            for img_path3 in images3:
                image_name3 = os.path.basename(img_path3)[:-4]
                if image_name3 == image_name1:
                    images_path3.append(img_path3)
                else:
                    print("出错啦！！！！")
            for img_path4 in images4:
                image_name4 = os.path.basename(img_path4)[:-4]
                if image_name4 == image_name1:
                    images_path4.append(img_path4)
                else:
                    print("出错啦！！！！")


    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(images_path1)))

    plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(flower_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(flower_class)), flower_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()

    return images_path1, images_path2, images_path3, images_path4, images_label, image_name

def predict(data_path):
    # 加载预训练模型
    net = load_net()

    # 加载测试集
    images_path1, images_path2, images_path3, images_path4, images_label = read_split_data(root1, root2, root3, root4)
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor()])
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  ])}
    # # 实例化测试数据集
    # train_dataset = MyDataSet(images_path=train_images_path,
    #                           images_class=train_images_label,
    #                           transform=data_transform["train"])

    # load image
    assert os.path.exists(images_path1), "file: '{}' dose not exist.".format(img_path)
    img1 = Image.open(images_path1)
    img2 = Image.open(images_path2)
    img3 = Image.open(images_path3)
    img4 = Image.open(images_path4)
    # plt.imshow(img)
    # [N, C, H, W]
    img1 = data_transform(img1)
    img2 = data_transform(img2)
    img3 = data_transform(img3)
    img4 = data_transform(img4)
    # expand batch dimension
    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)
    img3 = torch.unsqueeze(img3, dim=0)
    img4 = torch.unsqueeze(img4, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # model = create_model(num_classes=5).to(device)
    # # load model weights
    # model_weight_path = "./weights/model-9.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    with torch.no_grad():
        net.eval()
        for i in range(len(img_path)):
            # predict class
            output = torch.squeeze(net(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()




            

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    data_path = "/data/zhouheng/classification/data/test/images"
    predict(data_path)