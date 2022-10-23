### Inversion

```bash
MODEL_PATH='styleganinv_face_256.pkl'
IMAGE_LIST='examples/test.list'
python invert.py $MODEL_PATH $IMAGE_LIST
```

NOTE: We find that 100 iterations are good enough for inverting an image, which takes about 8s (on P40). But users can always use more iterations (much slower) for a more precise reconstruction.

### Semantic Diffusion

```bash
MODEL_PATH='styleganinv_face_256.pkl'
TARGET_LIST='examples/target.list'
CONTEXT_LIST='examples/context.list'
python diffuse.py $MODEL_PATH $TARGET_LIST $CONTEXT_LIST
```

NOTE: The diffusion process is highly similar to image inversion. The main difference is that only the target patch is used to compute loss for **masked** optimization.

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

NOTE: Boundaries are obtained using [InterFaceGAN](https://github.com/genforce/interfacegan).

### Style Mixing

```bash
STYLE_DIR='results/inversion/test'
CONTENT_DIR='results/inversion/test'
python mix_style.py $MODEL_PATH $STYLE_DIR $CONTENT_DIR
```
