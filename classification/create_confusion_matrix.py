import os
import json
import argparse
import sys
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as pltconda
from prettytable import PrettyTable
from utils import read_split_data
from my_dataset import MyDataSet
from model.densenet import densenet121 as create_model
# from model.efficientNet import efficientnet_b0 as create_model
#from model.inception_v4 import inception_v4_resnet_v2 as create_model
#from model.all_resnet1 import resnet101 as create_model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pandas import Series, DataFrame
import pandas as pd

# from model import swin_base_patch4_window12_384_in22k as create_model

# def create_model(num_classes: int = 2, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=96,
#                             depths=(2, 2, 6, 2),
#                             num_heads=(3, 6, 12, 24),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
def read_split_data1(root1: str, root2: str, root3: str, root4: str):
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
    images_name1 = []
    images_name2 = []
    images_name3 = []
    images_name4 = []
    # 存储训练集的所有图片路径
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
            image_name = os.path.basename(img_path1)
            image_name1 = os.path.basename(img_path1)[:-4]
            images_name1.append(image_name1)
            images_path1.append(img_path1)
            images_label.append(image_class)
            print(image_class)
            img_path2 = cla_path2 + '/' + image_name
            img_path3 = cla_path3 + '/' + image_name
            img_path4 = cla_path4 + '/' + image_name
            images_name2.append(image_name1)
            images_name3.append(image_name1)
            images_name4.append(image_name1)
            images_path2.append(img_path2)
            images_path3.append(img_path3)
            images_path4.append(img_path4)

            
            # for img_path2 in images2:
            #     image_name2 = os.path.basename(img_path2)[:-4]
            #     print("2", image_name2)
            #     if image_name2 == image_name1:
            #
            #         images_name2.append(image_name2)
            #         images_path2.append(img_path2)
            #     else:
            #         print("出错啦！！！！")
            # for img_path3 in images3:
            #     image_name3 = os.path.basename(img_path3)[:-4]
            #     if image_name3 == image_name1:
            #         images_name3.append(image_name3)
            #         images_path3.append(img_path3)
            #     else:
            #         print("出错啦！！！！")
            # for img_path4 in images4:
            #     image_name4 = os.path.basename(img_path4)[:-4]
            #     if image_name4 == image_name1:
            #         images_name4.append(image_name4)
            #         images_path4.append(img_path4)
            #     else:
            #         print("出错啦！！！！")


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

    return images_path1, images_path2, images_path3, images_path4, images_label, images_name1, images_name2, images_name3, images_name4

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    讲解； https://blog.csdn.net/weixin_45902056/article/details/123723921
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        # print('zip(preds, labels);', zip(preds, labels))
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        print('self.matrix; ', self.matrix)
        sum_TP = 0
        for i in range(self.num_classes):
            print('self.matrix[i, i];', self.matrix[i, i])
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            print('TP;', TP)
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

class load_data(Dataset):
    def __init__(self, args):
        self.args = args
        image_path = []
        for path in glob.glob(os.path.join(args.data_path, '*.png')):
            image_path.append(path)
        self.image_path = image_path

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        path = self.image_path[idx]
        image = np.load(path, allow_pickle=True)

        patientid = image['patientid'].tolist()
        label = image['label']
        images = image['image']

        images = images.astype(np.float32)
        label = label.astype(np.float32)
        images = images[np.newaxis, ...].copy()

        return {'inputs0': images, 'label': label, 'patientid': patientid}

def main(args):
    # dataset = load_data(args)
    #
    # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    # test_data_loader = DataLoader(dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=False,
    #                               num_workers=nw,
    #                               pin_memory=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    val_images_path1, val_images_path2, val_images_path3, val_images_path4, val_images_label, image_name1, image_name2, image_name3,image_name4 = read_split_data1(args.root1, args.root2, args.root3, args.root4)
    print("1",val_images_label)
    img_size = 224
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size)),
                                   #transforms.RandomHorizontalFlip(),
                                   # transforms.CenterCrop(img_size),
                                   transforms.ToTensor()])}
                                   #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 实例化验证数据集
    val_dataset = MyDataSet(images_path1=val_images_path1,
                            images_path2=val_images_path2,
                            images_path3=val_images_path3,
                            images_path4=val_images_path4,
                            images_class=val_images_label,
                            images_name1=image_name1,
                            images_name2=image_name2,
                            images_name3=image_name3,
                            images_name4=image_name4,
                            transform=data_transform["val"])
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)
    # load pretrain weights
    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    # confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    with torch.no_grad():
        name = []
        scoder1 = []
        scoder0 = []
        csv = []
        for val_data in tqdm(val_loader, file=sys.stdout):
            images1, images2, images3, images4, labels, img_name = val_data
            name.append(img_name)
            outputs1 = model(images1.to(device), images2.to(device))
            outputs2 = model(images3.to(device), images4.to(device))

            # outputs = (outputs1 + outputs2)/2

            predict1 = F.sigmoid(outputs1).squeeze(1).cpu().detach().numpy()
            predict2 = F.sigmoid(outputs2).squeeze(1).cpu().detach().numpy()
            # labels = np.array(labels.cpu()) > 0
            # th = 0.5
            # predict = predict > th
            # print('labels; ', labels)
            # print('predict;', predict)
            # tp = np.sum((predict == True) & (labels == True))
            # tn = np.sum((predict == False) & (labels == False))
            # fp = np.sum((predict == True) & (labels == False))
            # fn = np.sum((predict == False) & (labels == True))
            # print('tp:{0:.4f}\ttn:{1:.4f}\tfp:{2:.4f}\tfn:{3:.4f}'.format(tp, tn, fp, fn))

            predict1.tolist()
            predict2.tolist()
            score0 = 0.5*predict1[0, 0] + 0.5*predict2[0, 0]
            score1 = 0.5*predict1[0, 1] + 0.5*predict2[0, 1]
            # score0 = predict1[0,0]
            # score1 = predict1[0,1]
            # score0 = predict2[0, 0]
            # score1 = predict2[0, 1]



            # predict1.tolist()

            scoder1.append(score1)
            scoder0.append(score0)

            # outputs1 = torch.softmax(outputs1, dim=1)
            # outputs1 = torch.argmax(outputs1, dim=1)
            # confusion.update(outputs.to("cpu").numpy(), labels.to("cpu").numpy())

        name = DataFrame(name)
        name.columns = ['image_id']
        scoder1 = DataFrame(scoder1)
        scoder1.columns = ['1']
        scoder0 = DataFrame(scoder0)
        scoder0.columns = ['0']
        scoder = pd.concat([scoder1, scoder0], axis=1, join='inner')
        scoder = DataFrame(scoder)
        csv = pd.concat([name, scoder], axis=1, join='inner')
        csv.set_index(['image_id'], inplace=True)
        csv.to_csv(os.path.join(args.save_csv, 'con.csv'))

    # confusion.plot()
    # confusion.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--root1', type=str,
                        default="/media/data/cyq/pretrian_data2/internal/cls_image/test/cc/le")
    parser.add_argument('--root2', type=str,
                        default="/media/data/cyq/pretrian_data2/internal/cls_image/test/cc/re")
    parser.add_argument('--root3', type=str,
                        default="/media/data/cyq/pretrian_data2/internal/cls_image/test/mlo/le")
    parser.add_argument('--root4', type=str,
                        default="/media/data/cyq/pretrian_data2/internal/cls_image/test/mlo/re")
    parser.add_argument('--save_csv', type=str,
                        default="/data/cyq/CESM/classification2/result/final")

    # 训练权重路径
    parser.add_argument('--weights', type=str, default='/media/data/cyq/CESM/classification2/in4/model-98.pth',#22
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
