# Logo Recognition

## Environment
python=3.9.13
```
pip install pip-tools

cd STR
platform=cu117
make torch-${platform}
pip install -r requirements/core.${platform}.txt -e .[train,test]
pip install pip-tools
make clean-reqs reqs
cd ..

cd ChineseCLIP
pip install -e .
cd ..

pip install pytorch-metric-learning

export PYTHONPATH=STR/
```
## Our pretrained model

## Prepare datasets
download LogoDet-3K to dataset folder

https://drive.google.com/file/d/1YogZRnp93gQaJ_2mEdtL0BXNa-hp6kIu/view?usp=sharing
```
cd dataset
unzip LogoDet-3K
cd ..
python dataset/crop_logodet3k.py
python dataset/split_logodet3k.py
```

## Train
```
python train.py
```
## Evaluate
```
python evaluate.py
```
## Predict
