# test f(x,y) and f(x) and application
import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import numpy as np
import wideresnet
import json
# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        print(self.f.last_dim)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:,None])


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def dataset_fn(train, transform):
    if dataset == "cifar10":
        return tv.datasets.CIFAR10(root=data_root, transform=transform,
                                   download=True, train=train)
    elif dataset == "cifar100":
        return tv.datasets.CIFAR100(root=data_root, transform=transform,
                                    download=True, train=train)
    else:
        return tv.datasets.SVHN(root=data_root, transform=transform,
                                download=True, split="train" if train else "test")


def cycle(loader):
    while True:
        for data in loader:
            yield data



def get_data():
    if dataset == 'svhn':
        transform_train = tr.Compose([
            tr.Pad(4, padding_mode="reflect"),
            tr.RandomCrop(im_sz),
            tr.ToTensor(),
            tr.Normalize((.5, .5, .5), (.5, .5, .5)),
            lambda x: x + sigma * t.randn_like(x)
        ])
    else:
        transform_train = tr.Compose([
            tr.Pad(4, padding_mode="reflect"),
            tr.RandomCrop(im_sz),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize((.5, .5, .5), (.5, .5, .5)),
            lambda x: x + sigma * t.randn_like(x)
        ])
    transform_test = tr.Compose([
        tr.ToTensor(),  # ToTensor()把灰度范围从0-255变换到0-1之间
        tr.Normalize((.5, .5, .5), (.5, .5, .5)),  # Normalize()把0-1变换到-1-1,image=(image-mean)/std
        lambda x: x + sigma * t.randn_like(x)
    ])
    # get all imgs
    full_train =dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # 打乱
    np.random.shuffle(all_inds)
    # 划分验证集训练集
    if n_valid is not None:
        valid_inds, train_inds = all_inds[:n_valid], all_inds[n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    # print(type(train_inds))
    train_inds = np.array(train_inds)
    # print(type(train_inds))
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    if labels_per_class>0:
        print("a")
        for i in range(n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:labels_per_class])
            other_inds.extend(train_inds[train_labels == i][labels_per_class:])
    else:
        print("b")
        train_labeled_inds = train_inds
    # print(train_labels)
    # print(train_labels.size)
    # print(train_labeled_inds)
    # print(type(train_labeled_inds))
    # # print(full_train[31704][1])
    # # print(full_train[27439][1])
    # print(train_labeled_inds.size)
    dset_train = DataSubset(dataset_fn(True, transform_train), inds=train_inds)
    dset_train_labeled = DataSubset(dataset_fn(True, transform_train), inds=train_labeled_inds)
    dset_valid = DataSubset(dataset_fn(True, transform_test), inds=valid_inds)
    dset_test = dataset_fn(False, transform_test)
    # print(train_inds)
    # print(type(train_inds))
    # print(train_inds.size)
    # print(len(valid_inds))
    print(train_inds)
    print(train_labeled_inds)
    print(train_inds is train_labeled_inds)
    print((train_inds == train_labeled_inds).all())
    dload_train = DataLoader(dset_train, batch_size=batch_size,
                             shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=batch_size,
                                     shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False,
                             num_workers=4, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False,
                            num_workers=4, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid, dload_test


def init_random(bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_sample_q():
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(bs), []
        print(len(replay_buffer))
        print(len(replay_buffer) // n_classes)
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
        print(buffer_size)
        inds = t.randint(0, buffer_size, (bs,))
        print(inds)
        print(inds.shape)
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            print(inds)
            print(inds.shape)
            assert not uncond
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(bs)
        choose_random = (t.rand(bs) < reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=n_steps):
        f.eval()
        # get batch size
        bs = batch_size if y is None else y.size(0)
        print(y.size(0))
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += sgld_lr * f_prime + sgld_std * t.randn_like(x_k)
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q


def get_model_and_buffer(sample_q):
    model_cls = F if uncond else CCF
    infor = "F" if uncond else "CCF"
    print(infor)
    f = model_cls(depth, width, norm, dropout_rate=dropout_rate, n_classes=n_classes)
    if not uncond:
        assert buffer_size % n_classes == 0, "Buffer size must be divisible by n_classes"
    if load_path is None:
        replay_buffer = init_random(buffer_size)
    else:
        print(f"loading model from {load_path}")
        ckpt_dict = t.load(load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]
    f = f.to(device)
    return f, replay_buffer


save_dir = "./t6"
data_root = "../data"
seed = 2
im_sz = 256
n_ch = 3
sigma = 3e-2
n_valid = 5000
labels_per_class = -1
batch_size = 64
uncond = False
reinit_freq = 0.05
n_steps = 20
sgld_lr = 1.0
sgld_std = 1e-2
depth = 28
width = 10
norm = None  # [None, "norm", "batch", "instance", "layer", "act"]
dropout_rate = 0.0
buffer_size = 10000
load_path = "CIFAR10_MODEL.pt"
clf_only = False
optimizer = "adam"  # ["adam", "sgd"]
lr = 1e-4
weight_decay = 0.0
n_epochs = 20
decay_epochs = [16, 18]
decay_rate = .3
warmup_iters = 1000
class_cond_p_x_sample = False
print_every = 100
# loss weight
p_x_weight = 1.
p_y_given_x_weight = 1.
p_x_y_weight = 0.
plot_uncond = True
plot_cond = False
ckpt_every = 10
eval_every = 1
print_to_log = False
score_fn = "py"


makedirs(save_dir)
if print_to_log:
    sys.stdout = open(f'{save_dir}/log.txt', 'w')
t.manual_seed(seed)
if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)


# dataset "cifar10", "svhn", "cifar100"
dataset = "cifar10"
n_classes = 100 if dataset == "cifar100" else 10
# dload_train, dload_train_labeled, dload_valid, dload_test = get_data()

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

sample_q = get_sample_q()
f, replay_buffer = get_model_and_buffer(sample_q)

y_lab = t.randint(0, 9, (100,))
x_lab = init_random(2)
print(y_lab)
print(x_lab.shape)

re = f.classify(x_lab)
print(re.shape)

re1 = f(x_lab)
print(re1)
# re2 = f(x_lab, y_lab)
# print(re2)
# re3 = f.classify(x_lab)
# print(re3)
# print(re3.shape)

# corrects, losses, pys, preds = [], [], [], []
# logits = f.classify(x_lab)
# # print(logits)
# print(logits.shape)
# py = nn.Softmax()(f.classify(x_lab)).max(1)[0].detach().cpu().numpy()
# pys.extend(py)
# print(py.shape)
# loss = nn.CrossEntropyLoss(reduce=False)(logits, y_lab).cpu().detach().numpy()
# losses.extend(loss)
# correct = (logits.max(1)[1] == y_lab).float().cpu().numpy()
# corrects.extend(correct)
# preds.extend(logits.max(1)[1].cpu().numpy())


def grad_norm(x):
    x_k = t.autograd.Variable(x, requires_grad=True)
    f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
    grad = f_prime.view(x.size(0), -1)
    return grad.norm(p=2, dim=1)


def score_fn(x):
    if score_fn == "px":
        return f(x).detach().cpu()
    elif score_fn == "py":
        return nn.Softmax()(f.classify(x)).max(1)[0].detach().cpu()
    else:
        return -grad_norm(x).detach().cpu()

# real_scores = []
# x_lab = x_lab.to(device)
# scores = score_fn(x_lab)
# real_scores.append(scores.numpy())
# print(scores.mean())
# fake_scores = []
# x_fake = init_random(100)
# x_fake = x_fake.to(device)
# scores = score_fn(x_fake)
# fake_scores.append(scores.numpy())
# print(scores.mean())
# real_scores = np.concatenate(real_scores)
# fake_scores = np.concatenate(fake_scores)
# real_labels = np.ones_like(real_scores)
# fake_labels = np.zeros_like(fake_scores)
# import sklearn.metrics
# scores = np.concatenate([real_scores, fake_scores])
# labels = np.concatenate([real_labels, fake_labels])
# score = sklearn.metrics.roc_auc_score(labels, scores)
# print(score)

# n_it = replay_buffer.size(0)//100
# print(n_it)
# all_y = []
# for i in range(n_it):
#     x = replay_buffer[i * 100: (i + 1) * 100].to(device)
# print(replay_buffer.shape)
# print(x.shape)