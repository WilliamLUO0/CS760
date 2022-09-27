import torchvision.transforms as transform
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from tqdm import tqdm
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

# seed=123

# img0=Image.open('shark.jpg')
# img1=transform.Pad([105,105],fill=(0,125,5),padding_mode='constant')(img0)
# img2=transform.Pad([105,105],fill=(0,125,5),padding_mode='edge')(img0)
# img3=transform.Pad([105,105],fill=(0,125,5),padding_mode='reflect')(img0)
# img4=transform.Pad([105,105],fill=(0,125,5),padding_mode='symmetric')(img0)
# img5=transform.Pad(4, padding_mode="reflect")(img0)
# axs = plt.figure().subplots(1, 6)
# axs[0].imshow(img0);axs[0].set_title('src');axs[0].axis('off')
# axs[1].imshow(img1);axs[1].set_title('constant');axs[1].axis('off')
# axs[2].imshow(img2);axs[2].set_title('edge');axs[2].axis('off')
# axs[3].imshow(img3);axs[3].set_title('reflect');axs[3].axis('off')
# axs[4].imshow(img4);axs[4].set_title('symmetric');axs[4].axis('off')
# axs[5].imshow(img5);axs[5].set_title('xxx');axs[5].axis('off')
# plt.show()

# all_inds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# valids_inds, train_inds = all_inds[:2], all_inds[2:]
# print(type(train_inds))
# train_inds = np.array(train_inds)
# print(type(train_inds))

# t.manual_seed(seed)
# if t.cuda.is_available():
#     t.cuda.manual_seed_all(seed)
#
# bs = 2
# a = t.FloatTensor(bs, 3, 32, 32)
# b = a.uniform_(-1, 1)
# print(a)
# print(type(a))
# print(a.shape)
#
# print(b)
# print(type(b))
# print(b.shape)

# print(t.__version__)
# x = t.Tensor([[1,2,3],[3,4,5]])
# x = t.autograd.Variable(x, requires_grad=True)
# # x = t.autograd.Variable(t.ones(2,2),requires_grad=True)
# y1 = x + 3
# y2 = x - 3
# y3 = x * 3
# y4 = t.div(x, 3, rounding_mode='trunc')
# y5 = t.cos(x)
# y6 = y3 * 3
#
# print(x)
# print(y6)
#
# y6.backward(t.ones_like(x), retain_graph=True)
# y3.backward(t.ones_like(x), retain_graph=True)
# # y3.sum().backward()
# # 多次叠加
# print(x.grad)
# print(3e-2)

# x = t.Tensor([[[10,0,0],[9,0,0],[8,0,0]],[[7,0,0],[6,0,0],[5,0,0]]])
# y = x.logsumexp(2)
# print(x)
# print(y)
# print(x.dim())

# input=t.arange(15).view(3,5)
# print("input:\n",input)
#
# index1=t.tensor([
#     [1,0,0,0,0],
#     [0,0,1,2,1],
#     [1,2,0,0,0]])
# print("index:\n",index1)
#
# print("dim=0时:\n",t.gather(input,dim=0,index=index1))

# input=t.tensor([[
#         [1,2,3],
#         [4,5,6],
#         [7,8,9]]
# ])
# index1=t.tensor([[
#         [0,0],
#         [0,0],
#         [0,0]
# ]])
# print("input:\n",input)
# print("index:\n",index1)
# print("dim=3时:\n",t.gather(input,dim=2,index=index1))

# a = t.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[7,8,9],[10,11,12],[13,14,15]]])
# b = a[1]
# print(a)
# print(a.dim())
# print(a.shape)
# print(b)
# print(b.dim())
# print(b.shape)

# print(1e8)

# a = t.Tensor([[1,2,3],[4,5,6]])
# print(a)
# print(a.max(1))
# print(a.max(1)[1])

# a = t.FloatTensor(10000, 3, 32, 32)
# b = t.FloatTensor(10000, 3, 32, 32).uniform_(-1, 2)
# # print(a)
# print(a.shape)
# # print(b)
# print(b.shape)
# print(b.dim())

# a = (t.rand(64) < 0.05).float()
# print(a)
# print(a.shape)
# b = (t.rand(64) < 0.05).float()[:, None, None, None]
# print(b)
# print(b.shape)
# y = t.ones(64)
# print(y)


# def x():
#     def x1(b):
#         c = b+10
#         return c
#
#     def x2(a, b):
#         d = x1(b)
#         e = a+d
#         b = e+d
#         return e
#     return x2
#
#
# y = x()
# p = 10
# q = 30
# z = y(p, q)
# print(z)
# print(p)
# print(q)
# import wideresnet
#
# f = wideresnet.Wide_ResNet(depth=28, widen_factor=10, norm=None, dropout_rate=0.0)
# summary(model=f, input_size=[(3,256,256)], batch_size=2)
# print(summary())
import cv2
import matplotlib.pyplot as plt
import torchvision as tv, torchvision.transforms as tr
import numpy as np
import skimage.io as io
load_path="D:\\Desktop\\000000.jpg"
image = io.imread(load_path)
plt.imshow(image)
plt.show()
# image =transform.resize(image, (208, 208))
# # 将图片的取值范围改成（0~255）。
# img = image * 255
# img = img.astype(np.uint8)
# plt.imshow(image)
# plt.show()
image = tr.Resize((128, 128))
plt.imshow(image)
plt.show()


# tr.Resize((128, 128)),
#         # tr.RandomHorizontalFlip(),
#         tr.ToTensor(),
#         tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#         lambda x: x + sigma * t.randn_like(x)
