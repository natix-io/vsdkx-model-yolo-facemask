# Face mask detection

This repo supports face mask detection utilizing YoloV5. Ideally this repo could be deprecated and the face mask detection model could be integrated to be supported by the `vsdkx-model-yolo-torch` repo.

### Model Settings
```yaml
'conf_thresh': 0.5, # Float class confidence threshold
'iou_thresh': 0.4 # Float Intersection of Union threshold
```
### Model Config
The model is currently configured to support a custom trained facemask detection on YOLOv5 as TFlite. The model configuration can be found in `config/models.yaml`.
 ```yaml
model_path: 'wight.tflite'
input_shape: [320, 320]
classes_len: 2
anchors: [ [ 5.98438, 7.60156, 12.57031, 15.35938, 21.73438, 27.95312 ],
             [ 32.90625, 41.90625, 47.37500, 61.37500, 70.06250, 90.50000 ],
             [ 94.75000, 123.12500, 125.25000, 161.12500, 192.00000, 198.25000 ] ]
anchor_len: 3
max_width_height: 4096
mask_on: 1
mask_off: 0
mask_on_threshold: 0.60
```
