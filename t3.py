import numpy as np
from tqdm import tqdm
import os.path
from utils.visualizer import save_image
from models.helper import build_generator
from utils.editor import parse_boundary_list, get_layerwise_manipulation_strength, parse_indices

# boundary1 = np.load("./a/indoor_lighting_boundary.npy")
# print(boundary1)
# boundary2 = np.load("./boundaries/stylegan_bedroom/indoor_lighting_boundary.npy", allow_pickle=True)[()]
# print(isinstance(boundary2['meta_data']['manipulate_layers'], str))


def manipulate(latent_codes,
               boundary,
               start_distance=-5.0,
               end_distance=5.0,
               step=21,
               layerwise_manipulation=False,
               num_layers=1,
               manipulate_layers=None,
               is_code_layerwise=False,
               is_boundary_layerwise=False,
               layerwise_manipulation_strength=1.0):
    layer_indices = parse_indices(manipulate_layers, min_val=0, max_val=num_layers - 1)
    print(layer_indices)
    x = latent_codes
    print(x.shape)

    if not is_boundary_layerwise:
        b = boundary
        b = np.tile(b, [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
        print(b.shape)
    else:
        b = boundary[0]
        if b.shape[0] != num_layers:
            raise ValueError(f'Boundary should be with shape [num_layers, '
                             f'*code_shape], where `num_layers` equals to '
                             f'{num_layers}, but {b.shape} is received!')
    print(layerwise_manipulation_strength)
    if isinstance(layerwise_manipulation_strength, (int, float)):
        s = [float(layerwise_manipulation_strength) for _ in range(num_layers)]
    elif isinstance(layerwise_manipulation_strength, (list, tuple)):
        s = layerwise_manipulation_strength
        if len(s) != num_layers:
            raise ValueError(f'Shape of layer-wise manipulation strength `{len(s)}` '
                             f'mismatches number of layers `{num_layers}`!')
        print(s)
    elif isinstance(layerwise_manipulation_strength, np.ndarray):
        s = layerwise_manipulation_strength
        if s.size != num_layers:
            raise ValueError(f'Shape of layer-wise manipulation strength `{s.size}` '
                             f'mismatches number of layers `{num_layers}`!')
    else:
        raise ValueError(f'Unsupported type of `layerwise_manipulation_strength`!')
    s = np.array(s).reshape([num_layers if axis == 0 else 1 for axis in range(b.ndim)])
    print(s.shape)
    print(b.shape)
    b = b * s
    print(b.shape)
    print(x.shape[1:])

    num = x.shape[0]
    print(num)
    code_shape = x.shape[2:]
    print(code_shape)
    x = x[:, np.newaxis]
    print(x.shape)
    b = b[np.newaxis, np.newaxis, :]
    print(b.shape)
    l = np.linspace(start_distance, end_distance, step).reshape(
        [step if axis == 1 else 1 for axis in range(x.ndim)]
    )
    print(l)
    print(l.shape)
    results = np.tile(x, [step if axis == 1 else 1 for axis in range(x.ndim)])
    print(results.shape)
    is_manipulatable = np.zeros(results.shape, dtype=bool)
    print(is_manipulatable.shape)
    print(layer_indices)
    is_manipulatable[:, :, layer_indices] = True
    print(is_manipulatable.shape)
    results = np.where(is_manipulatable, x + l * b, results)
    assert results.shape == (num, step, num_layers, *code_shape)

    return results if layerwise_manipulation else results[:, :, 0]


work_dir = "./a"
step = 7
layerwise_manipulation = True
latent_space_type = 'w'
disable_manipulation_truncation  = False
start_distance = -3.0
end_distance = 3.0

model = build_generator("stylegan_bedroom")

latent_codes = np.load("./a/w.npy")
latent_codes = latent_codes.reshape(-1, 512).astype(np.float)
print(latent_codes.shape)

total_num = latent_codes.shape[0]
latent_codes = model.easy_synthesize(latent_codes=latent_codes,
                                     latent_space_type=latent_space_type, generate_style=False, generate_image=False)
print(latent_codes['wp'].shape)

for key, val in latent_codes.items():
    print(key)
    print(val.shape)

boundaries = parse_boundary_list("b/boundary_list.txt")

for boundary_info, boundary_path in boundaries.items():
    boundary_name, space_type = boundary_info
    prefix = f'{boundary_name}_{space_type}'
    print(prefix)
    try:
        boundary_file = np.load(boundary_path, allow_pickle=True).item()
        boundary = boundary_file['boundary']
        manipulate_layers = boundary_file['meta_data']['manipulate_layers']
    except ValueError:
        boundary = np.load(boundary_path)
        manipulate_layers = "6-11"

    if layerwise_manipulation and space_type != 'z':
        layerwise_manipulation = True
        is_code_layerwise = True
        is_boundary_layerwise = (space_type == 'wp')
        if(not disable_manipulation_truncation) and space_type == 'w':
            strength = get_layerwise_manipulation_strength(
            model.num_layers, model.truncation_psi, model.truncation_layers)
            print(strength)
        else:
            strength = 1.0
            print("strength")
        space_type = 'wp'
    else:
        if layerwise_manipulation:
            print(f'  Skip layer-wise manipulation for boundary '
                           f'`{boundary_name}` from Z space. Traditional '
                           f'manipulation is used instead.')
        layerwise_manipulation = False
        is_code_layerwise = False
        is_boundary_layerwise = False
        strength = 1.0

    print(space_type)
    print(latent_codes[space_type].shape)
    print(boundary.shape)
    print(start_distance)
    print(end_distance)
    print(step)
    print(layerwise_manipulation)
    print(model.num_layers)
    print(manipulate_layers)
    print(is_code_layerwise)
    print(is_boundary_layerwise)
    print(strength)

    codes = manipulate(latent_codes=latent_codes[space_type],
                       boundary=boundary,
                       start_distance=start_distance,
                       end_distance=end_distance,
                       step=step,
                       layerwise_manipulation=layerwise_manipulation,
                       num_layers=model.num_layers,
                       manipulate_layers=manipulate_layers,
                       is_code_layerwise=is_code_layerwise,
                       is_boundary_layerwise=is_boundary_layerwise,
                       layerwise_manipulation_strength=strength)

    print(codes.shape)
    # for s in range(step):
    #     images = model.easy_synthesize(latent_codes=)
    print(space_type)
    for s in tqdm(range(step), leave=False):
        images = model.easy_synthesize(latent_codes=codes[:, s],
                                       latent_space_type=space_type)['image']
        for n, image in enumerate(images):
            save_image(os.path.join(work_dir, f'{prefix}_{n:05d}_{s:03d}.jpg'), image)

