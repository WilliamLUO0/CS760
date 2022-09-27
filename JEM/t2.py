import torch
import torch.nn as nn
import wideresnet
import torchvision.transforms as tr


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, 10)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z)


def test_clf(f):
    transform_test= tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + torch.randn_like(x) * sigma]
    )
    
    return 0


depth = 28
width = 10
norm = None
sigma = 3e-2
n_steps = 0

model_cls = F
f = model_cls(depth, width, norm)
ckpt_dict = torch.load("CIFAR10_MODEL.pt")
f.load_state_dict(ckpt_dict['model_state_dict'])
replay_buffer = ckpt_dict["replay_buffer"]

test_clf(f)


