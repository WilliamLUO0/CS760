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

t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
num_attribute = 2
num_category = 6

data_root = "../data/train/"
save_dir = "./vgg_xac"
im_sz = 32
n_ch = 3
# # tr
sigma = 3e-2
batch_size = 64
# # model
uncond = True
# depth = 28
# width = 10
# norm = None  # [None, "norm", "batch", "instance", "layer", "act"]
# dropout_rate = 0.0
buffer_size = 10000
load_path = "./vgg_xac/best_attribute_ckpt1.pt"
# # sgld
reinit_freq = 0.05
n_steps = 20
sgld_lr = 1
sgld_std = 1e-2
# # optimizer
optimizer = "adam"  # ["adam", "sgd"]
lr = 1e-4
weight_decay = 0.0
# # training
n_epochs = 800
decay_epochs = [760, 780]
decay_rate = .3
warmup_iters = 10
p_x_weight = 0.3
p_a_given_x_weight = 0.3
p_x_a_weight = 0.
p_c_given_x_a_weight = 1.
p_x_c_a_weight = 0.
class_cond_p_x_sample = False
# # print
print_every = 100
ckpt_every = 10
eval_every = 1
print_to_log = True
# # gpu
device_ids = [0, 1, 2, 3]


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


class CnF(nn.Module):
    def __init__(self):
        super(CnF, self).__init__()
        self.f = vgg16.VGG16()
        self.attribute_output = nn.Linear(512, num_attribute)
        self.category_output = nn.Linear(512, num_category)

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


class MyDataset(Dataset):
    def __init__(self, basedata, inds=None):
        self.basedata = basedata
        self.data = []
        self.category_label = []
        self.attribute_label = []
        self.inds = inds
        for i in basedata:
            self.data.append(i[0])
            self.category_label.append(i[1])
        for i in range(len(basedata.imgs)):
            attribute = basedata.imgs[i][0].split("/")[4]
            if attribute == "negative":
                self.attribute_label.append(0)
            else:
                self.attribute_label.append(1)

    def __len__(self):
        return len(self.basedata.imgs)

    def __getitem__(self, index):

        return self.data[index], self.category_label[index], self.attribute_label[index]


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
        tr.Resize((32, 32)),
        # tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        lambda x: x + sigma * t.randn_like(x)
    ]))
    mydataset = MyDataset(dataset)
    all_inds = list(range(len(mydataset)))
    np.random.seed(1234)
    np.random.shuffle(all_inds)
    train_inds, test_inds = all_inds[:int(len(mydataset) * 0.8)], all_inds[int(len(mydataset) * 0.8):]
    train_inds = np.array(train_inds)
    dset_train = DataSubset(mydataset, inds=train_inds)
    dset_test = DataSubset(mydataset, inds=test_inds)
    train_sample = distributed.DistributedSampler(dset_train)
    test_sample = distributed.DistributedSampler(dset_test)
    dload_train = DataLoader(dset_train, batch_size=batch_size, shuffle=False, num_workers=4,
                             drop_last=True, sampler=train_sample)
    dload_test = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4,
                            drop_last=False, sampler=test_sample)
    mydataset = MyDataset(dataset)
    all_inds = list(range(len(mydataset)))
    np.random.seed(123)
    np.random.shuffle(all_inds)
    train_inds, test_inds = all_inds[:int(len(mydataset) * 0.8)], all_inds[int(len(mydataset) * 0.8):]
    train_labeled_inds = np.array(train_inds)
    dset_train_labeled = DataSubset(mydataset, inds=train_labeled_inds)
    train_sample_labeled = distributed.DistributedSampler(dset_train_labeled)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=batch_size, shuffle=False, num_workers=4,
                                     drop_last=True, sampler=train_sample_labeled)
    dload_train_labeled = cycle(dload_train_labeled)
    return dload_train, dload_train_labeled, dload_test


def init_random(bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_model_and_buffer(sample_q):
    model_cls = UnF if uncond else CnF
    infor = "UnF" if uncond else "CnF"
    print(infor)
    f = model_cls()
    f = nn.parallel.DistributedDataParallel(f.cuda(), device_ids=[args.local_rank])
    if not uncond:
        assert buffer_size % num_category == 0, "Buffer size must be divisible by args.n_classes"
    if load_path is None:
        replay_buffer = init_random(buffer_size)
    else:
        print(f"loading model from {load_path}")
        ckpt_dict = t.load(load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]
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
        return samples.cuda(non_blocking=True), index

    def sample_q(f, replay_buffer, y=None, n_steps=n_steps):
        f.eval()
        bs = batch_size if y is None else y.size(0)
        init_sample, buffer_index = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f.module(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
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
    f.cuda()


def eval_classification(f, dload):
    corrects_a, corrects_c, losses_a, losses_c, losses = [], [], [], [], []
    for batch_data in dload:
        x_p_d, a_p_d, c_p_d = batch_data[0], batch_data[2], batch_data[1]
        x_p_d, a_p_d, c_p_d = x_p_d.cuda(non_blocking=True), a_p_d.cuda(non_blocking=True), c_p_d.cuda(non_blocking=True)
        logits_a = f.module.forward_attribute(x_p_d)
        loss_a = nn.CrossEntropyLoss(reduction='none')(logits_a, a_p_d).cpu().numpy()
        logits_c = f.module.forward_category(x_p_d)
        loss_c = nn.CrossEntropyLoss(reduction='none')(logits_c, c_p_d).cpu().numpy()
        loss = loss_a + loss_c
        losses.extend(loss)
        losses_a.extend(loss_a)
        losses_c.extend(loss_c)
        correct_a = (logits_a.max(1)[1] == a_p_d).float().cpu().numpy()
        corrects_a.extend(correct_a)
        correct_c = (logits_c.max(1)[1] == c_p_d).float().cpu().numpy()
        corrects_c.extend(correct_c)
    loss_a = np.mean(losses_a)
    loss_c = np.mean(losses_c)
    loss = np.mean(losses)
    c_a = np.mean(corrects_a)
    c_c = np.mean(corrects_c)
    return c_a, c_c, loss_a, loss_c, loss



utils.makedirs(save_dir)
if print_to_log:
    sys.stdout = open(f'{save_dir}/log.txt', 'w')
t.manual_seed(seed)
if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
t.cuda.set_device(args.local_rank)

print(lr)
print(p_x_weight)
print(p_a_given_x_weight)
print(p_c_given_x_a_weight)

dload_train, dload_train_labeled, dload_test = get_data()
for index, batch_data in enumerate(dload_train):
    print(index, batch_data[0].shape, batch_data[1], batch_data[2])
    break;

for index, batch_data in enumerate(dload_train_labeled):
    print(index, batch_data[0].shape, batch_data[1], batch_data[2])
    break;

for index, batch_data in enumerate(dload_test):
    print(index, batch_data[0].shape, batch_data[1], batch_data[2])
    break;

x = dload_train_labeled.__next__()
print(x[0].shape)

sample_q = get_sample_q()
f, replay_buffer = get_model_and_buffer(sample_q)

sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

params = f.parameters()
if optimizer == "adam":
    optim = t.optim.Adam(params, lr=lr, betas=[.9, .999], weight_decay=weight_decay)
else:
    optim = t.optim.SGD(params, lr=lr, momentum=.9, weight_decay=weight_decay)

best_category_acc = 0.0
best_attribute_acc = 0.0
cur_iter = 1
for epoch in range(n_epochs):
    if epoch in decay_epochs:
        for param_group in optim.param_groups:
            new_lr = param_group['lr'] * decay_rate
            param_group['lr'] = new_lr
        print("Decaying lr to {}".format(new_lr))
    for i, batch_data_p in tqdm(enumerate(dload_train)):
        # if cur_iter <= warmup_iters:
        #    lr = lr * cur_iter / float(warmup_iters)
        #    print("current iter:{}, lr:{}".format(cur_iter, lr))
        #    for param_group in optim.param_groups:
        #        param_group['lr'] = lr

        x_p_d = batch_data_p[0]
        x_p_d = x_p_d.cuda(non_blocking=True)
        batch_data_lab = dload_train_labeled.__next__()
        x_lab, c_lab, a_lab = batch_data_lab[0], batch_data_lab[1], batch_data_lab[2]
        x_lab, c_lab, a_lab = x_lab.cuda(non_blocking=True), c_lab.cuda(non_blocking=True), a_lab.cuda(non_blocking=True)

        L = 0.
        if p_x_weight > 0:  # max log p(x)
            if class_cond_p_x_sample:
                assert not uncond, "can only draw class-conditional samples if EBM is class-cond"
                y_q = t.randint(0, num_category, (batch_size,)).cuda(non_blocking=True)
                x_q = sample_q(f, replay_buffer, y=y_q)
            else:
                x_q = sample_q(f, replay_buffer)

            fp_all = f.module(x_p_d)
            fq_all = f.module(x_q)
            fp = fp_all.mean()
            fq = fq_all.mean()

            l_p_x = -(fp - fq)
            L += p_x_weight * l_p_x
            if cur_iter % print_every ==0:
                print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq, fp-fq))

        if p_a_given_x_weight > 0:  # max log(a|x)
            logits_a = f.module.forward_attribute(x_lab)
            l_p_a_given_x = nn.CrossEntropyLoss()(logits_a, a_lab)
            L += p_a_given_x_weight * l_p_a_given_x

            if cur_iter % print_every == 0:
                acc = (logits_a.max(1)[1] == a_lab).float().mean()
                print('P(a|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch, cur_iter, l_p_a_given_x.item(), acc.item()))

        if p_c_given_x_a_weight > 0:  # max log(c|x,a)
            logits_c = f.module.forward_category(x_lab)
            l_p_c_given_x_a = nn.CrossEntropyLoss()(logits_c, c_lab)
            L += p_c_given_x_a_weight * l_p_c_given_x_a

            if cur_iter % print_every == 0:
                acc = (logits_c.max(1)[1] == c_lab).float().mean()
                print('P(c|x,a) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch, cur_iter, l_p_c_given_x_a.item(), acc.item()))

        if L.abs().item() > 1e8:
            print("BAD BOI")
            print(L)
            1/0

        optim.zero_grad()
        L.backward()
        optim.step()
        cur_iter += 1

    if epoch % ckpt_every == 0:
        checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt')

    if epoch % eval_every == 0:
        f.eval()
        with t.no_grad():
            correct_a, correct_c, loss_a, loss_c, loss = eval_classification(f, dload_test)
            print("Epoch {}: Test Loss {}, Test Attribute Loss {}, Test Category Loss {}, Test Attribute Acc {}, Test Category Acc {}".format(epoch, loss, loss_a, loss_c, correct_a, correct_c))
            correct_a_train, correct_c_train, loss_train_a, loss_train_c, loss_train = eval_classification(f, dload_train)
            print("Epoch {}: Train Loss {}, Train Attribute Loss {}, Train Category Loss {}, Train Attribute Acc {}, Train Category Acc {}".format(epoch, loss_train, loss_train_a, loss_train_c, correct_a_train, correct_c_train))
            if correct_a > best_attribute_acc:
                best_attribute_acc = correct_a
                print("Best Attribute Acc! : {}".format(correct_a))
                checkpoint(f, replay_buffer, "best_attribute_ckpt.pt")
            if correct_c > best_category_acc:
                best_category_acc = correct_c
                print("Best Category Acc! : {}".format(correct_c))
                checkpoint(f, replay_buffer, "best_category_ckpt.pt")
        f.train()
    checkpoint(f, replay_buffer, "last_ckpt.pt")
