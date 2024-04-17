# Install
Minimal example to visualize the data.

## Create virutal environment

```
conda create --name opensun3d python=3.8
conda activate opensun3d
pip install pyviz3d numpy pandas opencv-python scipy
```

## Download repository

```
git clone git@github.com:OpenSun3D/OpenSun3D.github.io.git
cd OpenSun3D.github.io
```

## Download example scenes

```
python challenge/download_data_opensun3d.py --data_type=challenge_development_set --download_dir=data
```

## Dataload and visualize scene 

```
python challenge/demo_dataloader_lowres.py
```