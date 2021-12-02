# GMAN: Generative Meta-Adversarial Network for Unseen Object Navigation

## Setup
- Clone the repository `git clone https://github.com/sx-zhang/GMAN.git` and move into the top-level directory `cd GMAN`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate ng`
- We provide pre-trained model of our GMAN in the `trained_models` directory.
- Download the [dataset](). The offline data is discretized from [AI2THOR](https://ai2thor.allenai.org/) simulator.  
The `data` folder should look like this
```python
  data/ 
    └── New_ai2thor_40/
        ├── FloorPlan1/
        │   ├── resnet18_featuremap.hdf5
        │   ├── graph.json
        │   ├── visible_object_map.json
        │   ├── att_in_view_v2.hdf5
        │   ├── grid.json
        │   ├── object_poses.json
        ├── FloorPlan2/
        └── ...
```