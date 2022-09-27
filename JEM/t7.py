# print summary
from torchsummary import summary
from vgg16 import VGG16
import torch as t
import torch.nn as nn
import numpy as np
import wideresnet

num_attribute = 2
num_category = 6

class UnF(nn.Module):
    def __init__(self, depth=28, width=10, norm=None, dropout_rate=0.0):
        super(UnF, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)


    # def forward(self, x):
    #     penult_z = self.f(x)
    #     return penult_z

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return penult_z




class CnF(nn.Module):
    def __init__(self, depth=28, width=10, norm=None, dropout_rate=0.0):
        super(CnF, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.attribute_output = nn.Linear(self.f.last_dim, num_attribute)
        self.category_output = nn.Linear(self.f.last_dim, num_category)

    # def forward(self, x):
    #     penult_z = self.f(x)
    #     return penult_z

    def forward(self, x, y=None):
        logits = self.forward_attribute(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])

    def forward_category(self, x):
        penult_z = self.f(x)
        return self.category_output(penult_z).squeeze()

    def forward_attribute(self, x):
        penult_z = self.f(x)
        return self.attribute_output(penult_z).squeeze()


# f = VGG16()
# print(summary(f, (3, 32, 32)))
depth = 28
width = 10
norm = None  # [None, "norm", "batch", "instance", "layer", "act"]
dropout_rate = 0.0
model = UnF
f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
print(summary(f, (3, 128, 128)))
# print(f)
#
# ckpt_dict = t.load("./wrxac1/best_attribute_ckpt.pt")
# # print(ckpt_dict.keys())
# # print(ckpt_dict["model_state_dict"].keys())
#
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in ckpt_dict["model_state_dict"].items():
#     name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
#     new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
#
# f.load_state_dict(new_state_dict)
# replay_buffer = ckpt_dict["replay_buffer"]
#
# print(new_state_dict.keys())
# print(new_state_dict["energy_output.weight"].shape)
# print(new_state_dict["energy_output.bias"].shape)
#
# x = t.load("../data/ne_il_bedroon.npy")
# print(x.shape)

# print(f.attribute_output.weight)
# print(f.f.weight)
# print(f.category_output.weight)