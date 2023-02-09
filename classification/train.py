import os
import argparse
# from torchvision import models
from sklearn.model_selection import KFold
import random
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
#from model.inception_v4 import inception_v4_resnet_v2 as create_model   #resnet3
from model.densenet import densenet121 as create_model
#from model.efficientNet import efficientnet_b0 as create_model
# from model.all_resnet1 import resnet101 as create_model
from utils import train_one_epoch, evaluate
import torch.nn as nn
import numpy as np

def kfold(root1: str, root2: str, root3: str, root4: str, k:int ,batch_size:int , nw:int):
    """ K折交叉验证 """
    random.seed(0)
    assert os.path.exists(root1), "dataset root: {} does not exist.".format(root1)

    # 遍历文件夹，一个文件夹对应一个类别
    patient_class = [cla for cla in os.listdir(root1) if os.path.isdir(os.path.join(root1, cla))]
    print('patient_class', patient_class)

    # 排序，保证顺序一致
    patient_class.sort()

    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(patient_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    print('json;', json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 定义路径和每个类别的样本总数的空列表
    images_path1 = []
    images_path2 = []
    images_path3 = []
    images_path4 = []
    every_class_num = []
    train_iter, valid_iter = [], []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    # 数据转换
    #224 124
    data_transform = {
        "train": transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]),
                                     #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor()])}
                                   #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 遍历每个文件夹下的文件
    for cla in patient_class:
        cla_path1 = os.path.join(root1, cla)
        cla_path2 = os.path.join(root2, cla)
        cla_path3 = os.path.join(root3, cla)
        cla_path4 = os.path.join(root4, cla)
        # 遍历获取supported支持的所有文件路径
        images1 = [os.path.join(root1, cla, i) for i in os.listdir(cla_path1)
                  if os.path.splitext(i)[-1] in supported]
        images2 = [os.path.join(root2, cla, j) for j in os.listdir(cla_path2)
                  if os.path.splitext(j)[-1] in supported]
        images3 = [os.path.join(root3, cla, i) for i in os.listdir(cla_path3)
                   if os.path.splitext(i)[-1] in supported]
        images4 = [os.path.join(root4, cla, j) for j in os.listdir(cla_path4)
                   if os.path.splitext(j)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images1))
        # 获取图像路径
        for img_path1 in images1:
            images_path1.append(img_path1)
        for img_path2 in images2:
            images_path2.append(img_path2)
        for img_path3 in images3:
            images_path3.append(img_path3)
        for img_path4 in images4:
            images_path4.append(img_path4)
    print("共读取到{}张低能图。。。。。".format(len(images_path1)))
    print("共读取到{}张减影图。。。。。".format(len(images_path2)))

    # 按照k折划分数据集
    X = np.arange(len(images_path1))
    KF = KFold(n_splits=k,shuffle=True)
    for train_idxs, valid_idxs in KF.split(X):

        # 定义空列表
        train_images_path1, train_images_path2, train_images_label, train_name1, train_name2 = [], [], [], [], []
        train_images_path3, train_images_path4, train_name3, train_name4 = [], [], [], []
        val_images_path1, val_images_path2, val_images_label, val_name1, val_name2 = [], [], [], [], []
        val_images_path3, val_images_path4, val_name3, val_name4 = [], [], [], []
        train_iter1, valid_iter1 = [] , []
        train_iter2, valid_iter2 = [] , []
        train_iter3, valid_iter3 = [], []
        train_iter4, valid_iter4 = [], []
        # 获取划分好的数据集索引
        for i in train_idxs:
            train_iter1.append(images_path1[i])
            train_iter2.append(images_path2[i])
            train_iter3.append(images_path3[i])
            train_iter4.append(images_path4[i])
        for j in valid_idxs:
            valid_iter1.append(images_path1[j])
            valid_iter2.append(images_path2[j])
            valid_iter3.append(images_path3[j])
            valid_iter4.append(images_path4[j])

        for timg_path1 in train_iter1:
            timage_name1 = os.path.basename(timg_path1)[:-4]
            train_name1.append(timage_name1)
            train_label = timg_path1.split('/')[-2]
            if train_label == 'malignant':
                train_label = 1
            if train_label == 'benign':
                train_label = 0
            train_images_label.append(train_label)
            train_images_path1.append(timg_path1)
            for timg_path2 in train_iter2:
                timage_name2 = os.path.basename(timg_path2)[:-4]
                if timage_name1 == timage_name2:
                    train_name2.append(timage_name2)
                    train_images_path2.append(timg_path2)
            for timg_path3 in train_iter3:
                timage_name3 = os.path.basename(timg_path3)[:-4]
                if timage_name1 == timage_name3:
                    train_name3.append(timage_name3)
                    train_images_path3.append(timg_path3)
            for timg_path4 in train_iter4:
                timage_name4 = os.path.basename(timg_path4)[:-4]
                if timage_name1 == timage_name4:
                    train_name4.append(timage_name4)
                    train_images_path4.append(timg_path4)

        for vimg_path1 in valid_iter1:
            vimage_name1 = os.path.basename(vimg_path1)[:-4]
            val_name1.append(vimage_name1)
            val_label = vimg_path1.split('/')[-2]
            if val_label == 'malignant':
                val_label = 1
            if val_label == 'benign':
                val_label = 0
            val_images_label.append(val_label)
            val_images_path1.append(vimg_path1)
            for vimg_path2 in valid_iter2:
                vimage_name2 = os.path.basename(vimg_path2)[:-4]
                if vimage_name1 == vimage_name2:
                    val_name2.append(vimage_name2)
                    val_images_path2.append(vimg_path2)
            for vimg_path3 in valid_iter3:
                vimage_name3 = os.path.basename(vimg_path3)[:-4]
                if vimage_name1 == vimage_name3:
                    val_name3.append(vimage_name3)
                    val_images_path3.append(vimg_path3)
            for vimg_path4 in valid_iter4:
                vimage_name4 = os.path.basename(vimg_path4)[:-4]
                if vimage_name1 == vimage_name4:
                    val_name4.append(vimage_name4)
                    val_images_path4.append(vimg_path4)

        # 实例化训练数据集
        train_dataset = MyDataSet(images_path1=train_images_path1,
                                  images_path2=train_images_path2,
                                  images_path3=train_images_path3,
                                  images_path4=train_images_path4,
                                  images_class=train_images_label,
                                  images_name1=train_name1,
                                  images_name2=train_name2,
                                  images_name3=train_name3,
                                  images_name4=train_name4,
                                  transform=data_transform["train"])
        # 实例化验证数据集
        val_dataset = MyDataSet(images_path1=val_images_path1,
                                images_path2=val_images_path2,
                                images_path3=val_images_path3,
                                images_path4=val_images_path4,
                                images_class=val_images_label,
                                images_name1=val_name1,
                                images_name2=val_name2,
                                images_name3=val_name3,
                                images_name4=val_name4,
                                transform=data_transform["val"])
        # 加载训练集
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)
        # 加载验证集
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=val_dataset.collate_fn)

        yield train_loader, val_loader


def main(args):
    # 设置GPU or GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 存放模型的路径
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 日志记录
    tb_writer = SummaryWriter()

    # 创建模型
    model = create_model(num_classes=args.num_classes).to(device)

    # 加载预训练权重
    # if args.weights != "/data/zhouheng/classification/Swin_classification/check_point/resnet101.pth":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)["model"]
    #     # 删除有关分类类别的权重
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head外，其他权重全部冻结
    #         if "head" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # 开始训练
    lr = args.lr
    k = args.k
    batch_size = args.batch_size
    data_path1 = args.data_path1
    data_path2 = args.data_path2
    data_path3 = args.data_path3
    data_path4 = args.data_path4
    train_iterations, train_loss, val_loss, trian_accuracy, val_accuracy = [], [], [], [], []
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    for epoch in range(args.epochs):
        # train
        #####---自动更新学习率的优化器---#####
        # LEARNING_RATE = max(lr * (0.1 ** (epoch // 100)), 1e-5)
        # optimizer = torch.optim.SGD([
        #     {'params': model.parameters()}
        # ], lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

        # 交叉验证
        train_losses = 0.0
        train_accs = 0.0
        val_losses = 0.0
        val_accs = 0.0
        for train_loader, val_loader in kfold(data_path1, data_path2, data_path3, data_path4, k, batch_size, nw):
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

            train_losses += train_loss
            train_accs += train_acc

            val_losses += val_loss
            val_accs += val_acc

            # train_iterations.append(epoch)
            # trian_loss.append((train_losses / k).to('cpu').item())
            # train_accuracy.append((train_accs / k).to('cpu').item())
            # val_loss.append((val_losses / k).to('cpu').item())
            # val_accuracy.append((val_accs / k).to('cpu').item())

        # if epoch % 5 == 0:
        print('train epoch {} trainLoss: {:.3f}, trainAcc: {:.3f} valLoss: {:.3f}, valAcc: {:.3f}'.format(epoch, train_losses/k, train_accs/k , val_losses/k, val_accs/k))

        # if epoch % 5 != 0:
        #     print('train epoch {} trainLoss: {:.3f}, trainAcc: {:.3f} valLoss: {:.3f}, valAcc: {:.3f}'.format(epoch, train_losses/k, train_accs/k))

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path1', type=str,
                        default="/train/cc/le")
    parser.add_argument('--data-path2', type=str,
                        default="/train/cc/re")
    parser.add_argument('--data-path3', type=str,
                        default="/train/mlo/le")
    parser.add_argument('--data-path4', type=str,
                        default="/train/mlo/re")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,
                        default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)

