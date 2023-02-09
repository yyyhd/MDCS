import os
import sys
import json
import pickle
import random
import torch.nn.functional as F
import torch
from tqdm import tqdm
from pandas import Series, DataFrame
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn


def read_split_data(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    patient_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    print('patient_class;', patient_class)
    # 排序，保证顺序一致
    patient_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(patient_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    print('json;', json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []   # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    train_name = []
    val_images_path = []  # 存储训练集的所有图片路径
    val_images_label = []  # 存储训练集图片对应索引信息
    val_name = []
    every_class_num1 = []     # 存储每个类别的样本总数
    every_class_num2 = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in patient_class:
        train_cla_path = os.path.join(root, cla, 'train')
        val_cla_path = os.path.join(root, cla, 'val')
        # 遍历获取supported支持的所有文件路径
        train_images = [os.path.join(train_cla_path, i) for i in os.listdir(train_cla_path)
                  if os.path.splitext(i)[-1] in supported]
        val_images = [os.path.join(val_cla_path, j) for j in os.listdir(val_cla_path)
                        if os.path.splitext(j)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num1.append(len(train_images))
        every_class_num2.append(len(val_images))
        # 按比例随机采样验证样本
        for timg_path in train_images:
            timage_name = os.path.basename(timg_path)
            train_name.append(timage_name[:-4])
            train_images_path.append(timg_path)
            train_images_label.append(image_class)

        for vimg_path in val_images:
            vimage_name = os.path.basename(vimg_path)
            val_name.append(vimage_name[:-4])
            val_images_path.append(vimg_path)
            val_images_label.append(image_class)

    print("{} images were found in the train dataset.".format(sum(every_class_num1)))
    print("{} images were found in the val dataset.".format(sum(every_class_num2)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validating.".format(len(val_images_path)))

    # plot_image = False
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

    return train_images_path, train_images_label, train_name, val_images_path, val_images_label, val_name

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=0.20, gamma=1.5, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
            # self.alpha = torch.tensor(alpha)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):

        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        # target_ = torch.zeros(target.size(0),self.class_num)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class FocalLoss1(nn.Module):

    def __init__(self, reduction='mean', gamma=2, alpha = 0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = self.alpha* (1 - p) ** self.gamma * logp
        return loss.mean()

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = FocalLoss(class_num=2)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num1 = torch.zeros(1).to(device)
    accu_num2 = torch.zeros(1).to(device)# 累计预测正确的样本数
    optimizer.zero_grad()

    save_csv = ''
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    print(data_loader)
    name_list = []
    scoder1 = []
    scoder0 = []
    for step, data in enumerate(data_loader):
        images1, images2, images3, images4, labels, img_name = data
        sample_num += images1.shape[0]

        pred1 = model(images1.to(device), images2.to(device))
        pred2 = model(images3.to(device), images4.to(device))
        predict1 = F.sigmoid(pred1).squeeze(1).cpu().detach().numpy()
        predict2 = F.sigmoid(pred2).squeeze(1).cpu().detach().numpy()
        pred_classes1 = torch.max(pred1, dim=1)[1]
        pred_classes2 = torch.max(pred2, dim=1)[1]
        for i in range(0, len(img_name)):
            name_list.append(img_name[i])
            score0 = 0.5 * predict1[i, 0] + 0.5 * predict2[i, 0]
            score1 = 0.5 * predict1[i, 1] + 0.5 * predict2[i, 1]
            scoder0.append(score0)
            scoder1.append(score1)

        accu_num1 += torch.eq(pred_classes1, labels.to(device)).sum()
        accu_num2 += torch.eq(pred_classes2, labels.to(device)).sum()
        accu_num = (accu_num1 + accu_num2) / 2

        loss1 = loss_function(pred1, labels.to(device))
        # print("l1",loss1)
        loss2 = loss_function(pred2, labels.to(device))
        # print("l2",loss2)
        loss = loss1* 0.5 + loss2* 0.5
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    name_list = DataFrame(name_list)
    name_list.columns = ['id']
    scoder1 = DataFrame(scoder1)
    scoder1.columns = ['1']
    scoder0 = DataFrame(scoder0)
    scoder0.columns = ['0']
    scoder = pd.concat([scoder1, scoder0], axis=1, join='inner')
    scoder = DataFrame(scoder)
    csv = pd.concat([name_list, scoder], axis=1, join='inner')
    csv.set_index(['id'], inplace=True)
    csv.to_csv(os.path.join(save_csv, 'train_result4.csv'))


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = FocalLoss(class_num=2)
    accu_num1 = torch.zeros(1).to(device)
    accu_num2 = torch.zeros(1).to(device) # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    save_csv = '/data/cyq/CESM/classification2/result/newmodel/val'
    sample_num = 0
    name_list = []
    scoder1 = []
    scoder0 = []
    data_loader = tqdm(data_loader, file=sys.stdout)
    j=0
    for step, data in enumerate(data_loader):
        images1, images2, images3, images4, labels, img_name = data
        sample_num += images1.shape[0]
        pred1 = model(images1.to(device), images2.to(device))
        pred2 = model(images3.to(device), images4.to(device))
        predict1 = F.sigmoid(pred1).squeeze(1).cpu().detach().numpy()
        predict2 = F.sigmoid(pred2).squeeze(1).cpu().detach().numpy()
        pred_classes1 = torch.max(pred1, dim=1)[1]
        pred_classes2 = torch.max(pred2, dim=1)[1]

        for i in range(0, len(img_name)):
            name_list.append(img_name[i])
            score0 = 0.5 * predict1[i, 0] + 0.5 * predict2[i, 0]
            score1 = 0.5 * predict1[i, 1] + 0.5 * predict2[i, 1]
            # score0 = predict1[i, 0]
            # score1 = predict1[i, 1]
            scoder0.append(score0)
            scoder1.append(score1)

        accu_num1 += torch.eq(pred_classes1, labels.to(device)).sum()
        accu_num2 += torch.eq(pred_classes2, labels.to(device)).sum()
        accu_num = (accu_num1 + accu_num2) / 2
        loss1 = loss_function(pred1, labels.to(device))
        # print('L1',loss1)

        loss2 = loss_function(pred2, labels.to(device))
        # print('L2', loss2)
        loss = loss1 *0.5 + loss2 * 0.5
        accu_loss += loss
        #print(accu_loss)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        # if not torch.isfinite(loss):
        #     print('WARNING: non-finite loss, ending training ', loss)
        #     sys.exit(1)
        
    name_list = DataFrame(name_list)
    name_list.columns = ['id']
    scoder1 = DataFrame(scoder1)
    scoder1.columns = ['1']
    scoder0 = DataFrame(scoder0)
    scoder0.columns = ['0']
    scoder = pd.concat([scoder1, scoder0], axis=1, join='inner')
    scoder = DataFrame(scoder)
    csv = pd.concat([name_list, scoder], axis=1, join='inner')
    csv.set_index(['id'], inplace=True)
    csv.to_csv(os.path.join(save_csv, 'val_result4.csv'))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num