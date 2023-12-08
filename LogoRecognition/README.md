# Logo Recognition

## Environment
python=3.9.13
```
pip install torch
pip install pip-tools

cd STR
platform=cu117
make torch-${platform}
pip install -r requirements/core.${platform}.txt -e .[train,test]
make clean-reqs reqs
cd ..

cd ChineseCLIP
pip install -e .
cd ..

pip install pytorch-metric-learning
pip install lightning

export PYTHONPATH=STR/
```
## Our pretrained model
### Best
https://drive.google.com/file/d/1hvE7E-XeIwG0c9IIGk9NOpDp7tDbb5I2/view?usp=sharing
### OCR Best
https://drive.google.com/file/d/13otW5eUo7-Mthms0vdy5XOhk3Q7f8U3t/view?usp=sharing
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
```
python predict.py
```

