# Data Augmented Deep Reinforcement Learning for Online 3D Bin Packing Problems

This repo is a Rainbow version of PCT. The off-policy RL algorithm is combined with a data augmentation trick, which increases sampling efficiency and reduces resources required by vectorized environment. Better performance is reached on benchmark problems. 

![](./../csc-lab/CCC论文/imgs/main.png)


## Paper

This work has been accepted by Chinese Conrol Conference 2024 as a poster paper.


## Dependencies
```bash
conda env create -f environment.yml
```
## Quick start

For training with the default arguments:
```bash
python main.py 
```
The training data is generated on the fly. The training logs (tensorboard) are saved in './logs/runs'. Related file backups are saved in './logs/experiment'.

## Usage

### Data description

Describe your 3D container size and 3D item size in 'givenData.py'
```
container_size: A vector of length 3 describing the size of the container in the x, y, z dimension.
item_size_set:  A list records the size of each item. The size of each item is also described by a vector of length 3.
```
### Dataset
We use the same dataset as PCT. You can download the prepared dataset from [here](https://drive.google.com/drive/folders/1QLaLLnpVySt_nNv0c6YetriHh0Ni-yXY?usp=sharing).
The dataset consists of 3000 randomly generated trajectories, each with 150 items. The item is a vector of length 3 or 4, the first three numbers of the item represent the size of the item, the fourth number (if any) represents the density of the item.

### Training

For training BPP policy on setting 1 (80 internal nodes and 50 leaf nodes) nodes:
```bash
python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50
```
If you want to train a model that works on the **continuous** domain, add '--continuous', don't forget to change your problem in 'givenData.py':
```bash
python main.py --continuous --setting 1 --internal-node-holder 80 --leaf-node-holder 50
```
#### Warm start
You can initialize a run using a pretrained model:
```bash
python main.py --load-model --model-path path/to/your/model
```

### Evaluation
To evaluate a model, you can add the `--evaluate` flag to `evaluation.py`:
```bash
python evaluation.py --evaluate --load-model --model-path path/to/your/model --load-dataset --dataset-path path/to/your/dataset
```
### Heuristic
Running heuristic.py for test heuristic baselines, the source of the heuristic algorithm has been marked in the code:

Running heuristic on setting 1 （discrete） with LASH method:
```
python heuristic.py --setting 1 --heuristic LSAH --load-dataset  --dataset-path setting123_discrete.pt
```

Running heuristic on setting 2 （continuous） with OnlineBPH method:
```
python heuristic.py --continuous --setting 2 --heuristic OnlineBPH --load-dataset  --dataset-path setting2_continuous.pt
```

### Help
```bash
python main.py -h
python evaluation.py -h
python heuristic.py -h
```

### License
```
This source code is released only for academic use. Please do not use it for commercial purpose without authorization of the author.
```
