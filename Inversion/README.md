### Inversion

```bash
MODEL_PATH='styleganinv_face_256.pkl'
IMAGE_LIST='examples/test.list'
python invert.py $MODEL_PATH $IMAGE_LIST
```

### Semantic Diffusion

```bash
MODEL_PATH='styleganinv_face_256.pkl'
TARGET_LIST='examples/target.list'
CONTEXT_LIST='examples/context.list'
python diffuse.py $MODEL_PATH $TARGET_LIST $CONTEXT_LIST
```

### Interpolation

```bash
SRC_DIR='results/inversion/test'
DST_DIR='results/inversion/test'
python interpolate.py $MODEL_PATH $SRC_DIR $DST_DIR
```

### Manipulation

```bash
IMAGE_DIR='results/inversion/test'
BOUNDARY='boundaries/expression.npy'
python manipulate.py $MODEL_PATH $IMAGE_DIR $BOUNDARY
```

### Style Mixing

```bash
STYLE_DIR='results/inversion/test'
CONTENT_DIR='results/inversion/test'
python mix_style.py $MODEL_PATH $STYLE_DIR $CONTENT_DIR
```
```bibtex
@inproceedings{zhu2020domain,
  title={In-domain gan inversion for real image editing},
  author={Zhu, Jiapeng and Shen, Yujun and Zhao, Deli and Zhou, Bolei},
  booktitle={European conference on computer vision},
  pages={592--608},
  year={2020},
  organization={Springer}
}
```
