import torchvision.datasets
import utils
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset, random_split
import os
import sys
import argparse
import numpy as np
import wideresnet
import json
from tqdm import tqdm
import re

t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
num_attribute = 2
num_category = 4


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=4):
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


class UnF(nn.Module):
    def __init__(self, depth=28, width=10, norm=None, dropout_rate=0.0):
        super(UnF, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.attribute_output = nn.Linear(self.f.last_dim, num_attribute)
        self.category_output = nn.Linear(self.f.last_dim, num_category)

    def forward(self, x):
        penult_z = self.f(x)
        return penult_z

    def forward_energy(self, x):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def forward_category(self, x):
        penult_z = self.f(x)
        return self.category_output(penult_z).squeeze()

    def forward_attribute(self, x):
        penult_z = self.f(x)
        return self.attribute_output(penult_z).squeeze()


class CnF(nn.Module):
    def __init__(self, depth=28, width=10, norm=None, dropout_rate=0.0):
        super(CnF, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.attribute_output = nn.Linear(self.f.last_dim, num_attribute)
        self.category_output = nn.Linear(self.f.last_dim, num_category)

    def forward(self, x):
        penult_z = self.f(x)
        return penult_z

    def forward_energy(self, x, y=None):
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


def cycle(loader):
    while True:
        for data in loader:
            yield data


def get_data():
    dataset = torchvision.datasets.ImageFolder(data_root, transform=tr.Compose([
        # tr.Pad(4, padding_mode="reflect"),
        # tr.RandomCrop(im_sz),
        tr.Resize((128, 128)),
        # tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # lambda x: x + sigma * t.randn_like(x)

    ]))
    mydataset = MyDataset(dataset)
    all_inds = list(range(len(mydataset)))
    np.random.seed(1234)
    np.random.shuffle(all_inds)
    train_inds, test_inds = all_inds[:int(len(mydataset)*0.8)], all_inds[int(len(mydataset)*0.8):]
    train_inds = np.array(train_inds)
    train_labels = np.array([dataset[ind][1] for ind in train_inds])
    train_labeled_inds = train_inds
    dset_train = DataSubset(dataset, inds=train_inds)
    dset_train_labeled = DataSubset(dataset, inds=train_labeled_inds)
    dset_test = DataSubset(dataset, inds=test_inds)
    dload_train = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dload_test = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_train_labeled, dload_test


def init_random(bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_model_and_buffer(sample_q):
    model_cls = F if uncond else CnF
    infor = "F" if uncond else "CnF"
    print(infor)
    f = model_cls(depth, width, norm, dropout_rate=dropout_rate)
    if not uncond:
        assert buffer_size % num_category ==0, "Buffer size must be divisible by args.n_classes"
    if load_path is None:
        replay_buffer = init_random(buffer_size)
    else:
        print(f"loading model from {load_path}")
        ckpt_dict = t.load(load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]
    f = f.to(device)
    return f, replay_buffer


def get_sample_q():
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // num_category
        index = t.randint(0, buffer_size, (bs,))
        if y is not None:
            index = y.cpu() * buffer_size + index
            assert not uncond
        buffer_samples = replay_buffer[index]
        random_samples = init_random(bs)
        choose_random = (t.rand(bs) < reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), index

    def sample_q(f, replay_buffer, y=None, n_steps=n_steps):
        f.eval()
        bs = batch_size if y is None else y.size(0)
        init_sample, buffer_index = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += sgld_lr * f_prime + sgld_std * t.randn_like(x_k)
        f.train()
        final_samples = x_k.detach()
        if len(replay_buffer) > 0:
            replay_buffer[buffer_index] = final_samples.cpu()
        return final_samples
    return sample_q


def checkpoint(f, buffer, tag):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(save_dir, tag))
    f.to(device)


def eval_classification(f, dload):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduction='none')(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


data_root = "../data/train/"
save_dir = "./xac7"
im_sz = 128
n_ch = 3
# tr
sigma = 3e-2
batch_size = 32
# model
uncond = True
depth = 28
width = 10
norm = None  # [None, "norm", "batch", "instance", "layer", "act"]
dropout_rate = 0.0
buffer_size = 10000
load_path = None
# sgld
reinit_freq = 0.05
n_steps = 20
sgld_lr = 1.0
sgld_std = 1e-2
# optimizer
optimizer = "sgd"  # ["adam", "sgd"]
lr = 0.01
weight_decay = 0.0
# training
n_epochs = 200
decay_epochs = [160, 180]
decay_rate = .3
warmup_iters = 1000
p_x_weight = 1.
p_a_given_x_weight = 1.
p_x_a_weight = 0.
p_c_given_x_a_weight = 1.
p_x_c_a_weight = 0.
class_cond_p_x_sample = False
# print
print_every = 100
ckpt_every = 10
eval_every = 1
print_to_log = True
# gpu
device_ids = [0, 1, 2, 3]


utils.makedirs(save_dir)
if print_to_log:
    sys.stdout = open(f'{save_dir}/log.txt', 'w')
t.manual_seed(seed)
if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)

# parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', default=-1, type=int,
#                     help='node rank for distributed training')
# args = parser.parse_args()
#
# dist.init_process_group(backend='nccl')
# t.cuda.set_device(args.local_rank)

dload_train, dload_train_labeled, dload_test = get_data()

for index, batch_data in enumerate(dload_train):
    print(index, batch_data[0].shape, batch_data[1].shape)
    break;

x_lab, y_lab = dload_train_labeled.__next__()
print(x_lab.shape)
print(y_lab)


# x = dload_train_labeled.__next__()
# print(x['img'].shape)
#
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
#
sample_q = get_sample_q()
f, replay_buffer = get_model_and_buffer(sample_q)

sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

params = f.parameters()
if optimizer == "adam":
    optim = t.optim.Adam(params, lr=lr, betas=[.9, .999], weight_decay=weight_decay)
else:
    optim = t.optim.SGD(params, lr=lr, momentum=.9, weight_decay=weight_decay)
#
best_test_acc = 0.0
cur_iter = 0
for epoch in range(n_epochs):
    if epoch in decay_epochs:
        for param_group in optim.param_groups:
            new_lr = param_group['lr'] * decay_rate
            param_group['lr'] = new_lr
        print("Decaying lr to {}".format(new_lr))
    for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
        # if cur_iter <= warmup_iters:
        #     lr = lr * cur_iter / float(warmup_iters)
        #     for param_group in optim.param_groups:
        #         param_group['lr'] = lr

        x_p_d = x_p_d.to(device)
        x_lab, y_lab = dload_train_labeled.__next__()
        x_lab, y_lab = x_lab.to(device), y_lab.to(device)

        L = 0.

        logits = f.classify(x_lab)
        loss = nn.CrossEntropyLoss()(logits, y_lab)
        L += loss
        if cur_iter % print_every == 0:
            acc = (logits.max(1)[1] == y_lab).float().mean()
            print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch, cur_iter, loss.item(), acc.item()))

        if L.abs().item() > 1e8:
            print("BAD BOIIIIIIIIII")
            1 / 0

        optim.zero_grad()
        L.backward()
        optim.step()
        cur_iter += 1

    if epoch % ckpt_every == 0:
        checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt')

    if epoch % eval_every == 0:
        f.eval()
        with t.no_grad():
            correct, loss = eval_classification(f, dload_test)
            print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
            if correct > best_test_acc:
                best_valid_acc = correct
                print("Best Valid!: {}".format(correct))
                checkpoint(f, replay_buffer, "best_valid_ckpt.pt")
        f.train()
    checkpoint(f, replay_buffer, "last_ckpt.pt")