### Step-1: Synthesize images and get semantic prediction

```bash
MODEL_NAME=stylegan_bedroom
OUTPUT_DIR=stylegan_bedroom
python synthesize.py $MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --num=500000 \
    --generate_prediction \
    --logfile_name=synthesis.log
```

### Step-2: Boundary search for potential candidates (repeat)

```bash
BOUNDARY_NAME=indoor_lighting
python train_boundary.py $OUTPUT_DIR/w.npy $OUTPUT_DIR/attribute.npy \
    --score_name=$BOUNDARY_NAME \
    --output_dir=$OUTPUT_DIR \
    --logfile_name=${BOUNDARY_NAME}_training.log
```

### Step-3: Rescore to identity the most relevant semantics

Use following command to conduct the layer-wise analaysis and identify relevant semantics:

```bash
BOUNDARY_LIST=stylegan_bedroom/boundary_list.txt
python rescore.py $MODEL_NAME $BOUNDARY_LIST \
    --output_dir $OUTPUT_DIR \
    --layerwise_rescoring \
    --logfile_name=rescore.log
```

## BibTeX

```bibtex
@article{yang2019semantic,
  title   = {Semantic hierarchy emerges in deep generative representations for scene synthesis},
  author  = {Yang, Ceyuan and Shen, Yujun and Zhou, Bolei},
  journal = {IJCV},
  year    = {2020}
}
```
