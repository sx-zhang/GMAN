# GMAN: Generative Meta-Adversarial Network for Unseen Object Navigation

## Setup
- Clone the repository `git clone https://github.com/sx-zhang/GMAN.git` and move into the top-level directory `cd GMAN`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate ng`
- We provide pre-trained model of [hoz](https://drive.google.com/file/d/11L-ejoWgLHPBe_F-gQ7dJ5gQZB0dzNjr/view?usp=sharing) and [hoztpn](https://drive.google.com/file/d/1hoqBLO6Oaty-TKT7a2slnhVx0wYi7LsC/view?usp=sharing). For evaluation and fine-tuning training, you can download them to the `trained_models` directory.
- Download the [dataset](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view), which refers to [ECCV-VN](https://github.com/xiaobaishu0097/ECCV-VN). The offline data is discretized from [AI2THOR](https://ai2thor.allenai.org/) simulator.  
The `data` folder should look like this
```python
  data/ 
    └── Scene_Data/
        ├── FloorPlan1/
        │   ├── resnet18_featuremap.hdf5
        │   ├── graph.json
        │   ├── visible_object_map_1.5.json
        │   ├── det_feature_categories.hdf5
        │   ├── grid.json
        │   └── optimal_action.json
        ├── FloorPlan2/
        └── ...
```