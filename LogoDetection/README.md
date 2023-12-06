# Logo Detection
detector: yolov8

## Environment
python=3.9.13
```
pip install Pillow
pip install ultralytics
pip uninstall opencv-python
pip install opencv-python-headless
```

## Our pretrained model
https://drive.google.com/file/d/1FDYOmZ5K712Zxvr0PMTuZ2flcTtdpaID/view?usp=sharing

## Prepare datasets
### OSLD
https://drive.google.com/file/d/1znu_IJs8k2FdxqYma4KDbP4bQIGlPSSu/view?usp=drive_link
```
unzip osld.zip
python osld2yolo.py
```
change path in osld.yaml

## Train
```
yolo detect train model=yolov8x.pt data=osld.yaml device=0,1
```
## Evaluate
```
yolo detect val model=best.pt data=osld.yaml device=0,1
```
## Predict
```
yolo detect predict model=best.pt source=test_images device=0,1 conf=0.001 iou=0.1
```