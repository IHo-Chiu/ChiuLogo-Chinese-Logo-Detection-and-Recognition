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

## Prepare datasets
### OSLD
unzip osld.zip
python osld2yolo.py

change path in osld.yaml

## Train
yolo detect train model=yolov8x.pt data=osld.yaml device=0,1

## Evaluate
yolo detect val model=runs/detect/train/weights/best.pt data=osld.yaml device=0,1

## Predict
yolo detect predict model=runs/detect/train/weights/best.pt source=test_images device=0,1 conf=0.001 iou=0.1