# GMAN: Generative Meta-Adversarial Network for Unseen Object Navigation

## Setup
- Clone the repository `git clone https://github.com/sx-zhang/GMAN.git` and move into the top-level directory `cd GMAN`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate ng`
- We provide pre-trained model of our GMAN in the `trained_models` directory.
- Download our [dataset](https://drive.google.com/file/d/1cyUH6hMrfiJD5VC2FFCYZY0hGJR78lQ5/view?usp=sharing). 
- Then extract files in `data` by
`tar -zxvf ai2thor_GMAN.tar.gz`
- The `data` folder should look like this
```python
  data/ 
    └── ai2thor//
        ├── FloorPlan1/
        │   ├── resnet18_featuremap.hdf5
        │   ├── graph.json
        │   ├── visible_object_map.json
        │   ├── att_in_view_v2.hdf5
        │   ├── grid.json
        ├── FloorPlan2/
        └── ...
```
## Training and Evaluation
### Train the baseline model 
`python main.py --title Basemodel --model BaseModel --workers 12 -–gpu-ids 0`
### Train our GMAN model 
`python main.py --title GMAN --model GMAN --workers 12 -–gpu-ids 0 --num_steps 20`
### Evaluate the GMAN model for seen objects 
```shell
python full_eval.py \
    --title GMAN \
    --model GMAN \
    --results-json GMAN_seen.json \
    --gpu-ids 0 \
    --num_steps 20 \
    --seen seen
```
### Evaluate the GMAN model for unseen objects 
```shell
python full_eval.py \
    --title GMAN \
    --model GMAN \
    --results-json GMAN_unseen.json \
    --gpu-ids 0 \
    --num_steps 20 \
    --seen unseen
```