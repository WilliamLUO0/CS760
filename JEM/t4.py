# test
import torch as t
import torch.nn as nn
import wideresnet
import os
import logging
import torchvision as tv
import torchvision.transforms as tr
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

    def forward1(self, x):
        penult_z = self.f(x)
        return penult_z


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
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
        for i in range(n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:labels_per_class])
            other_inds.extend(train_inds[train_labels == i][labels_per_class:])
    else:
        train_labeled_inds = train_inds
    dset_train = DataSubset(dataset_fn(True, transform_train), inds=train_inds)
    dset_train_labeled = DataSubset(dataset_fn(True, transform_train), inds=train_labeled_inds)
    dset_valid = DataSubset(dataset_fn(True, transform_test), inds=valid_inds)
    dset_test = dataset_fn(False, transform_test)
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
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
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
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


save_dir = "./t1"
data_root = "../data"
seed = 1
im_sz = 32
n_ch = 3
sigma = 3e-2
n_valid = 5000
labels_per_class = -1
batch_size = 64
uncond = True
reinit_freq = 0.05
n_steps = 20
sgld_lr = 1.0
sgld_std = 1e-2
depth = 28
width = 10
norm = None  # [None, "norm", "batch", "instance", "layer", "act"]
dropout_rate = 0.0
buffer_size = 10000
load_path = None
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

makedirs(save_dir)
t.manual_seed(seed)
if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)

# dataset "cifar10", "svhn", "cifar100"
dataset = "cifar10"
n_classes = 100 if dataset == "cifar100" else 10
dload_train, dload_train_labeled, dload_valid, dload_test = get_data()

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

sample_q = get_sample_q()
f, replay_buffer = get_model_and_buffer(sample_q)
sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

params = f.class_output.parameters() if clf_only else f.parameters()
if optimizer == "adam":
    optim = t.optim.Adam(params, lr=lr, betas=[.9, .999], weight_decay=weight_decay)
else:
    optim = t.optim.SGD(params, lr=lr, momentum=.9, weight_decay=weight_decay)

best_valid_acc = 0.0
cur_iter = 0
for epoch in range(n_epochs):
    print("Epoch:{}".format(epoch))
    if epoch in decay_epochs:
        for param_group in optim.param_groups:
            new_lr = param_group['lr'] * decay_rate
            param_group['lr'] = new_lr
        print("Decaying lr to {}".format(new_lr))
    for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
        if cur_iter <= warmup_iters:
            lr = lr * cur_iter / float(warmup_iters)
            for param_group in optim.param_groups:
                param_group['lr'] = lr

        x_p_d = x_p_d.to(device)
        # print(i)
        # print(x_p_d.shape)  # torch.Size([64, 3, 32, 32])
        x_lab, y_lab = dload_train_labeled.__next__()
        x_lab, y_lab = x_lab.to(device), y_lab.to(device)
        print("x")
        print(x_lab.shape)
        print("y")
        print(y_lab)
        print(y_lab.shape)
        # print(x_lab.shape)  # torch.Size([64, 3, 32, 32])
        # print(y_lab.shape)  # torch.Size([64])

        L = 0
        if p_x_weight > 0:  # maximize log p(x)
            if class_cond_p_x_sample:
                assert not uncond, "can only draw class-conditional samples if EBM is class-cond"
                y_q = t.randint(0, n_classes, (batch_size,)).to(device)
                x_q = sample_q(f, replay_buffer, y=y_q)
            else:
                x_q = sample_q(f, replay_buffer)  # sample from log-sumexp
                print(x_q.shape)  # torch.Size([64, 3, 32, 32])

            fp_all = f(x_p_d)
            print(fp_all.shape)  # torch.Size([64])
            fq_all = f(x_q)
            print(fq_all.shape)  # torch.Size([64])
            fp = fp_all.mean()
            fq = fq_all.mean()
            l_p_x = -(fp - fq)
            L += p_x_weight * l_p_x

            if cur_iter % print_every == 0:
                print('p(x) | {}:{} f(x_p_d)={} f(x_q)={} d={}'.format(epoch, i, fp, fq, fp - fq))
                print('L += p_x_weight * l_p_x, L:{}, p_x_weight:{}, l_p_x:{}'.format(L, p_x_weight, l_p_x))

        if p_y_given_x_weight > 0:  # maximize log p(y | x)
            logits = f.classify(x_lab)
            l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
            L += p_y_given_x_weight * l_p_y_given_x
            if cur_iter % print_every == 0:
                acc = (logits.max(1)[1] == y_lab).float().mean()
                print('P(y|x) {}:{} loss={}, acc={}'.format(epoch, cur_iter, l_p_y_given_x.item(), acc.item()))
                print('L += p_y_given_x_weight * l_p_y_given_x, L:{}, p_y_given_x_weight:{}, l_p_y_given_x:{}'.format(L, p_y_given_x_weight, l_p_y_given_x))

        if p_x_y_weight > 0:  # maximize log p(x, y)
            assert not uncond, "this objective can only be trained for class-conditional EBM"
            x_q_lab = sample_q(f, replay_buffer, y=y_lab)
            fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
            l_p_x_y = -(fp - fq)

            L += p_x_y_weight * l_p_x_y

            if cur_iter % print_every == 0:
                print('P(x, y) | {}:{} f(x_p_d)={} f(x_q)={} d={}'.format(epoch, i, fp, fq, fp-fq))
                print('L += p_x_y_weight * l_p_x_y, L:{}, p_x_y_weight:{}, l_p_x_y:{}'.format(L, p_x_y_weight, l_p_x_y))

        if L.abs().item() > 1e8:
            print("BAD BOI")
            print(L)
            1 / 0

        optim.zero_grad()
        L.backward()
        optim.step()
        cur_iter += 1

        if cur_iter % 100 == 0:
            if plot_uncond:
                if class_cond_p_x_sample:
                    assert not uncond, "can only draw class-conditional sample if EMB is class-cond"
                    y_q = t.randint(0, n_classes, (batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer)
                plot('{}/x_q_{}_{:>06d}.png'.format(save_dir, epoch, i), x_q)
            if plot_cond:
                y = t.arange(0, n_classes)[None].repeat(n_classes, 1).transpose(1, 0).contiguous().view(
                    -1).to(device)
                x_q_y = sample_q(f, replay_buffer, y=y)
                plot('{}/x_q_y{}_{:>06d}.png'.format(save_dir, epoch, i), x_q_y)
        break

    if epoch % ckpt_every == 0:
        checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt')

    if epoch % eval_every == 0 and (p_y_given_x_weight > 0 or p_x_y_weight > 0):
        f.eval()
        print("f.eval()")
        with t.no_grad():
            correct, loss = eval_classification(f, dload_valid)
            print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
            if correct > best_valid_acc:
                best_valid_acc = correct
                print("Best Valid!: {}".format(correct))
                checkpoint(f, replay_buffer, "best_valid_ckpt.pt")
            correct, loss = eval_classification(f, dload_test)
            print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
        f.train()
        print("f.train()")
    checkpoint(f, replay_buffer, "last_ckpt.pt")
    break

