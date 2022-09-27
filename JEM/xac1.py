# mydataset
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as tr
import cv2
from torch.utils.data import DataLoader, Dataset
import re
import h5py


class MyDataset(Dataset):
    def __init__(self, basedata):
        self.basedata = basedata
        self.data = []
        self.category_label = []
        self.attribute_label = []
        for i in basedata:
            self.data.append(i[0])
            self.category_label.append(i[1])
        # for i in range(len(basedata.imgs)):
        #     matchobj = re.match(r'(.*)/train/(.*)/(.*)/(.*)', basedata.imgs[i][0], re.M | re.I)
        #     if matchobj.group(3) == "negative":
        #         self.attribute_label.append(0)
        #     else:
        #         self.attribute_label.append(1)
        for i in range(len(basedata.imgs)):
            attribute = basedata.imgs[i][0].split("/")[4]
            if attribute == "negative":
                self.attribute_label.append(0)
            else:
                self.attribute_label.append(1)


    def __len__(self):
        return len(self.basedata.imgs)

    def __getitem__(self, index):
        dict_data = {
            'img': self.data[index],
            'category_label': self.category_label[index],
            'attribute_label': self.attribute_label[index]
        }
        return dict_data


# line = '../data/train/bedroom/negative/000000.jpg'
# matchobj = re.match(r'(.*)/train/(.*)/(.*)/(.*)', line, re.M|re.I)
# if matchobj:
#     print(matchobj.group(2))
#     print(matchobj.group(3))

path_train = "../data/train/"

dataset = torchvision.datasets.ImageFolder(path_train, transform=tr.Compose([
    tr.ToTensor(),
    tr.Normalize((.5, .5, .5), (.5, .5, .5))]))

print(dataset.classes)
print(dataset.class_to_idx)
print(dataset.imgs[0])
print(dataset.imgs[5000])


for i in dataset:
    print(i[0]) # data
    print(i[1]) # category

    break;
for i in range(len(dataset.imgs)):
    attribute = dataset.imgs[i][0].split("/")[4]
    print(attribute)
    break;

# for i in range(len(dataset.imgs)):
#     matchobj = re.match(r'(.*)/train/(.*)/(.*)/(.*)', dataset.imgs[i][0], re.M|re.I)
#     if matchobj.group(3)=="negative":
#         print("0")
#     else:
#         print("1") # attribute
#     break;


dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
# for index, batch_data in enumerate(dataloader):
#     print(index, batch_data[0].shape, batch_data[1].shape)

for index, (batch_data, x) in enumerate(dataloader):
    print(index, batch_data.shape, batch_data[0].shape)
    print(x)
    break;

mydataset = MyDataset(dataset)
print(mydataset.__len__())
mydataloader = DataLoader(mydataset, batch_size=64, shuffle=False, drop_last=True)

for index, batch_data in enumerate(mydataloader):
    print(index, batch_data["img"].shape, batch_data['category_label'], batch_data["attribute_label"])
    break;







