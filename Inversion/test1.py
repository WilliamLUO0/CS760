import torch


def _strip_state_dict_prefix(state_dict, prefix='module.'):
    """Removes the name prefix in checkpoint.

    Basically, when the model is deployed in parallel, the prefix `module.` will
    be added to the saved checkpoint. This function is used to remove the
    prefix, which is friendly to checkpoint loading.

    Args:
        state_dict: The state dict where the variable names are processed.
        prefix: The prefix to remove. (default: `module.`)
    """
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict

    stripped_state_dict = dict()
    for key in state_dict:
        stripped_state_dict[key.replace(prefix, '')] = state_dict[key]
    return stripped_state_dict


pthfile=r'checkpoint_iter025000.pth'
gfile = r'models/pretrain/styleganinv_ffhq256_generator.pth'
efile = r'models/pretrain/styleganinv_ffhq256_encoder.pth'
checkpoint = torch.load(pthfile, map_location=torch.device('cpu'))
generator = torch.load(gfile, map_location=torch.device('cpu'))
encoder = torch.load(efile, map_location=torch.device('cpu'))
if 'models' not in checkpoint:
    checkpoint = {'models': checkpoint}
print(checkpoint.keys())
models = checkpoint['models']
print(models.keys())
cp_g = models['generator']
cp_e = models['encoder']
cp_gs = models['generator_smooth']
print(cp_g.keys())
print(cp_gs.keys())
print(generator.keys())
print("\n")
print(cp_e.keys())
print(encoder.keys())
#torch.save(cp_g, "styleganinv_ffhq256_generator_1.pth")
#torch.save(cp_gs, "styleganinv_ffhq256_generator_s.pth")
#torch.save(cp_e, "styleganinv_ffhq256_encoder_1.pth")