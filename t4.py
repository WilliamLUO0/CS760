import numpy as np
import torch

# a=np.array([[1,2,3],[2,3,4]])
# b=np.array([[1,2]])
# b=b.reshape(2,1)
# a = a*b
# # print(c)
# print(a)

a = np.load("./predictors/pretrain/W_sceneattribute_wideresnet18.npy")
print(a)

# a = np.zeros((1,4,512,1,1), dtype=np.float)
# a = a.squeeze(axis=(0,3,4))
# print(a.shape)`

# a = torch.zeros([4,512,1,1], dtype=torch.int)
# a = a.view(a.size(0), -1)
# print(a.shape)

