import torchvision.datasets
import utils
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset, random_split, distributed
import os
import sys
import argparse
import numpy as np
import vgg16
import json
from tqdm import tqdm
import re
import torch.distributed as dist
from PIL import Image

t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
num_attribute = 2
num_category = 6


class UnF(nn.Module):
    def __init__(self):
        super(UnF, self).__init__()
        self.f = vgg16.VGG16()
        self.energy_output = nn.Linear(512, 1)
        self.attribute_output = nn.Linear(512, num_attribute)
        self.category_output = nn.Linear(512, num_category)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def forward_category(self, x):
        penult_z = self.f(x)
        return self.category_output(penult_z).squeeze()

    def forward_attribute(self, x):
        penult_z = self.f(x)
        return self.attribute_output(penult_z).squeeze()


class MyDataset(Dataset):
    def __init__(self, label_list, transform):
        super(MyDataset, self).__init__()
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.label_list[index][0]).convert("RGB")
        return self.transform(img), self.label_list[index][1], self.label_list[index][2]

    def __len__(self):
        return len(self.label_list)



# model = UnF
# f = model()
# # 训练category
# f.train()
# # 保存参数
# t.save(f, "load_path")
# # 读参数
# ckpt_dict = t.load("load_path")
# f.load_state_dict(ckpt_dict["model_state_dict"])
# replay_buffer = ckpt_dict["replay_buffer"]
# freeze = ['f', 'category_output']
# no_freeze = ['energy_output', 'attribute_output']
# # optim_param = []
# for name, param in f.named_parameters():
#     for x in freeze:
#         if x in name:
#             param.requires_grad = False
#     for x in no_freeze:
#         if x in name:
#             param.requires_grad = True
#             # optim_param.append(param)
# # optimizer.SGD(optim_param, lr=1e-3)
# t.optim.SGD([
#     {"params": f.f.parameters(), "lr": 1e-3},
#     {"params": f.category_output.parameters(), "lr": 1e-3}, ],
#     lr=1e-3,
# )
#
# t.optim.SGD([
#     {"params": f.energy_output.parameters(), "lr": 1e-3},
#     {"params": f.attribute_output.parameters(), "lr": 1e-3}, ],
#     lr=1e-3,
# )

# 训练attribute energy
# optim.zero_grad()
# L.backward()
# optim.step()

load_path = "../data/train"
split = 8

def get_data_pairs_deepfake(path_nor, split=8):
    temp_train_list, temp_test_list = [], []  # 临时存放数据集位置及类别
    category_path_arr = os.listdir(os.path.join(path_nor))
    for c_ids, folder in enumerate(category_path_arr): ## bedromm
        attribute_path_arr = os.listdir(os.path.join(path_nor,folder)) ## positive
        for attri_ids, attribute in enumerate(attribute_path_arr): # positive
            img_arr = os.listdir(os.path.join(path_nor, folder, attribute))
            count = 0  # 记录图片的数量，便于划分数据集
            for img in img_arr:
                img_path_nor = os.path.join(path_nor, folder, attribute, img)
                if count % 10 >= split:  # 按照3:7的比例划分数据集
                    temp_test_list.append([img_path_nor, c_ids, attri_ids])
                else:
                    temp_train_list.append([img_path_nor, c_ids, attri_ids])
                count += 1

    return temp_train_list, temp_test_list

train, test = get_data_pairs_deepfake(load_path)
print(train)
print(len(train))
print(len(test))
train, test = np.array(train), np.array(test)
np.save("train_data.npy", train)
np.save("test_data.npy", test)
a = np.load("train_data.npy")
print(a)




# temp_train_list, temp_test_list = [], []
# category_path_arr = os.listdir(os.path.join(load_path))
# print(category_path_arr)
# for c_ids, folder in enumerate(category_path_arr):
#     attribute_path_arr = os.listdir(os.path.join(load_path, folder))
#     print(attribute_path_arr)
#     for attri_ids, attribute in enumerate(attribute_path_arr):
#         print(attri_ids)
#         print(attribute)
#         img_arr = os.listdir(os.path.join(load_path, folder, attribute))
#         print(len(img_arr))
#         count = 0
#         for img in img_arr:
#             img_path_nor = os.path.join(load_path, folder, attribute, img)
#             print(img_path_nor)
#             if count % split == 0:
#                 temp_test_list.append([img_path_nor, c_ids, attri_ids])
#             else:
#                 temp_train_list.append([img_path_nor, c_ids, attri_ids])
#             count += 1
# print(len(temp_train_list))
# print(len(temp_test_list))
