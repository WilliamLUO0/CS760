# sgld
import torch as t, torch.nn as nn
import torchvision as tv, torchvision.transforms as tr
import numpy as np
import wideresnet


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
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def init_random(bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


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


seed = 1
uncond = False
depth = 28
width = 10
norm = None
dropout_rate = 0.0
dataset = "cifar10"
n_classes = 100 if dataset == "cifar100" else 10
buffer_size = 10000
load_path = None
n_ch = 3
im_sz = 32
batch_size = 64
reinit_freq = 0.05
n_steps = 20
sgld_lr = 1.0
sgld_std = 1e-2

t.manual_seed(seed)
if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

model_cls = F if uncond else CCF
infor = "F" if uncond else "CCF"
print(infor)
f = model_cls(depth, width, norm, dropout_rate=dropout_rate, n_classes=n_classes)
if load_path is None:
    replay_buffer = init_random(buffer_size)
print("replay_buffer")
print(replay_buffer.shape)
f = f.to(device)
# sample_q = get_sample_q()
# x_q = sample_q(f, replay_buffer)
y = None
# sample_q
f.eval()
bs = batch_size if y is None else y.size(0)
# sample_p_0
buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
inds = t.randint(0, buffer_size, (bs,))
print("inds")
print(inds)
print(inds.shape)
if y is not None:
    inds = y.cpu() * buffer_size + inds
    assert not uncond
buffer_samples = replay_buffer[inds]
print("buffer_samples")
print(buffer_samples.shape)
random_samples = init_random(bs)
print("random_samples")
print(random_samples.shape)
choose_random = (t.rand(bs) < reinit_freq).float()[:, None, None, None]
samples = choose_random * random_samples + (1-choose_random) * buffer_samples
samples.to(device)
# sample_q
init_sample = samples
print("innit_samples")
print(init_sample.shape)
buffer_inds = inds
print("buffer_inds")
print(buffer_inds)
print(buffer_inds.shape)
x_k = t.autograd.Variable(init_sample, requires_grad=True)
# y_k1 = f(x_k)
# print(y_k1)
# print(y_k1.shape)
# print(y_k1.sum())
# y_k2 = f(x_k, y=y)
# print(y_k2)
# print(y_k2.shape)
# print(y_k2.sum())
# y = t.ones(64)
# y_k3 = f(x_k, y=y)
# print(y_k3)
# print(y_k3.shape)
# sgld
for k in range(n_steps):
    f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
    x_k.data += sgld_lr * f_prime + sgld_std * t.randn_like(x_k)
f.train()
final_samples = x_k.detach()
if len(replay_buffer) > 0:
    replay_buffer[buffer_inds] = final_samples.cpu()

