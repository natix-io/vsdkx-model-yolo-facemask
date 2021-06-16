# -*- coding:utf-8 -*-

import time
import cv2

import numpy as np
import tensorflow as tf
from datetime import datetime

from vsdkx.core.interfaces import ModelDriver
from vsdkx.core.structs import Inference
from vsdkx.core.util.model import load_tflite


class YoloFacemaskDriver(ModelDriver):
    """
    Class for object detection
    """
    def __init__(self, model_settings: dict, model_config: dict,
                 drawing_config: dict):
        """
        Args:
            model_config (dict): Config dictionary with the following keys:
                'model_path' (str): Path to the tflite models
                'input_shape' (tuple): Shape of input(inference) image
                'classes_len' (int): Length of classes
                'anchors' (np.array): Anchors array
                'max_width_height' (int): Max pixel width and height
                'mask_on' (int): Mask on class ID
                'mask_off' (int): Mask off class ID
                'mask_on_threshold' (int): Minimum confidence threshold
                for 'mask'
            model_settings (dict): Model settings config with the
            following keys:
                'target_shape' (tuple): Image target shape
                'iou_thresh' (float): Threshold for Intersection of Union
                'conf_threshold' (float): Confidence threshold
        """
        super().__init__(model_settings, model_config, drawing_config)
        self._input_shape = model_config['input_shape']
        self._classes_len = model_config['classes_len']
        self._anchors = self._get_anchors(model_config['anchors'])
        self._anchor_len = model_config['anchor_len']
        self._max_width_height = model_config['max_width_height']
        self._mask_on = model_config['mask_on']
        self._mask_off = model_config['mask_off']
        self._mask_on_threshold = model_config['mask_on_threshold']
        self._target_shape = model_settings['target_shape']
        self._conf_thresh = model_settings['conf_thresh']
        self._iou_thresh = model_settings['iou_thresh']
        self._interpreter, self._input_details, self._output_details = \
            load_tflite(model_config['model_path'])

    def inference(
            self,
            image
    ) -> Inference:
        """
        Driver function for object detection inference

        Args:
            image (np.array): 3D image array
            debug (bool): Debug mode flag

        Returns:
            (array): Array of bounding boxes
        """
        # Resize the original image for inference
        resized_image = self._resize_img(image, self._input_shape)
        # Run the inference
        self._interpreter.set_tensor(
            self._input_details[0]['index'], resized_image)
        self._interpreter.invoke()
        # Get the inference result
        x = [self._interpreter.get_tensor(self._output_details[i]['index'])
             for i in range(1, self._anchor_len + 1)]

        # Process the inference result
        y = self._get_pred(x, self._input_shape)

        # Run the NMS to get the boxes with the highest confidence
        y = self._process_pred(y)
        boxes, scores, classes = [], [], []
        if y[0] is not None:
            y = np.squeeze(y, axis=0)
            boxes, scores, classes = y[:, :4], y[:, 4:5], y[:, 5:6]
            classes = self._skew_no_mask_bias(boxes, scores, classes)
            boxes = self._scale_boxes(boxes, self._input_shape, self._target_shape)

        return Inference(boxes, classes, scores)

    def _scale_boxes(self, boxes, input_shape, target_shape):
        """
        Scales the boxes to the size of the target image

        Args:
            boxes (np.array): Array containing the bounding boxes
            input_shape (tuple): The shape of the resized image
            target_shape (tuple): The shape of the target image

        Returns:
            (np.array): np.array with the scaled bounding boxes
        """
        gain = min(input_shape[0] / target_shape[0],
                   input_shape[1] / target_shape[1])
        pad = (input_shape[1] - target_shape[1] * gain) / 2, \
              (input_shape[0] - target_shape[0] * gain) / 2
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :] /= gain

        return boxes

    def _skew_no_mask_bias(self, boxes, scores, classes):
        """
        Skews the prediction results to the 'no_mask' class based on a minimum
        threshold for the 'mask' class. When the class confidence is lower or
        equal to that threshold, the class is updated to 'no_mask'. This bias
        allows to eliminate any false positive predictions.

        Args:
            boxes (np.array): Array of detected bounding boxes
            scores (np.array): Array of confidence scores
            classes (np.array): Array with class IDs

        Returns:
            classes (np.array): Updates classes array
        """

        for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            if self._mask_on_threshold >= score and class_id == self._mask_on:
                classes[i] = self._mask_off

        return classes

    def _resize_img(self, image, input_shape):
        """
        Resize input image to the expected input shape

        Args:
            image (np.array): 3D numpy array of input image
            input_shape (tuple): The shape of the input image

        Returns:
            (array): Resized image
        """
        image_resized = self._letterbox(image,
                                        new_shape=input_shape)[0]

        image_np = image_resized / 255.0
        image_np = np.expand_dims(image_np, axis=0)
        image_np = tf.cast(image_np, dtype=tf.float32)

        return image_np

    def _decode_box(self, box):
        """
        Decodes boxes from [box, y, w, h] to [x1, y1, x2, y2]
        where xy1=top-left, xy2=bottom-right

        Args:
            box (np.array): Array with box coordinates

        Returns:
            (np.array): np.array with new box coordinates
        """
        y = np.zeros_like(box)
        y[:, 0] = box[:, 0] - box[:, 2] / 2  # top left box
        y[:, 1] = box[:, 1] - box[:, 3] / 2  # top left y
        y[:, 2] = box[:, 0] + box[:, 2] / 2  # bottom right box
        y[:, 3] = box[:, 1] + box[:, 3] / 2  # bottom right y
        return y

    def _get_anchors(self, anchors):
        """
        Creates an anchor grid to replicate the output
        of the last convolution layers.

        Returns:
            (np.array): np.array with the anchor grid
        """
        anchor_reshape = np.array(anchors).reshape(3, -1, 2)
        anchor_grid = np.copy(anchor_reshape).reshape((3, 1, -1, 1, 2))
        return anchor_grid

    def _make_grid(self, x, y):
        """
        Builds a meshgrid from the shapes of x and y.

        Args:
            x (int): Horizontal shape of the i-th yolo head
            y (int): Vertical shape of the i-th yolo head

        Returns:
            (np.array): np.array with the meshgrid
        """
        xv, yv = np.meshgrid(np.arange(x), np.arange(y))
        return np.stack((xv, yv), axis=2).reshape((1, 1, y * x, 2))

    def _get_pred(self, predicted, image_size):
        """
        Performs post-processing on the output of the inference.

        Args:
            predicted (list): Inference output
            image_size (tuple): Image size

        Returns:
            (np.array): np.array of the processed predictions
        """
        z = []
        grid = [np.zeros(1)] * self._anchor_len
        num_outputs = self._classes_len + 5

        for i in range(self._anchor_len):
            _, _, y_x, _ = predicted[i].shape
            # Calculate the ratio
            ratio = image_size[1] / image_size[0]
            # Calculate the x and y shapes of the predicted bounding box
            y = int(np.sqrt(y_x / ratio))
            x = int(ratio * y)
            # Create the meshgrid for x and y
            grid[i] = self._make_grid(x, y)
            # Calculate the stride
            stride = image_size[0] // y
            # Pass the output through a sigmoid function
            out = tf.math.sigmoid(predicted[i]).numpy()
            # Get the center coordinates xy, and
            # height of the bounding boxes
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid[i]) * stride  # xy
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * self._anchors[i]  # wh
            z.append(out.reshape((-1, num_outputs)))

        pred = np.concatenate((z[0], z[1], z[2]), axis=0)
        pred = np.expand_dims(pred, axis=0)
        return pred

    def _process_pred(self, prediction):
        """
        Processes the prediction results and passes them through NMS

        Args:
            prediction (np.array): Array with the post-processed
            inference predictions

        Returns:
             (np.array): np.array detections with shape
              nx6 (x1, y1, x2, y2, conf, cls)
        """
        # Get candidates with confidence higher than the threshold
        xc = prediction[..., 4] > self._conf_thresh  # candidates

        output = [None]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center box, center y, width, height) to (x1, y1, x2, y2)
            box = self._decode_box(x[:, :4])
            i, j = np.nonzero(x[:, 5:] > self._conf_thresh)
            i, j = np.transpose(i), np.transpose(j)
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), axis=1)
            # Batched NMS
            classes = x[:, 5:6] * self._max_width_height  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + classes, x[:, 4]
            nms = np.array(tf.image.non_max_suppression(boxes, scores,
                                                        100, self._iou_thresh,
                                                        self._conf_thresh))

            if len(nms) > 0:
                output[xi] = x[nms]

        return np.asarray(output)

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Resize image in letterbox fashion.

        Args:
            img (np.array): 3D numpy array of input image
            new_shape (tuple): Array with the new image height and width
            color (tuple): Color array

        Returns:
            (np.array): np.array with the resized image
            (tuple): The height and width ratios
            (tuple): The width and height paddings
        """
        # Resize image to a 32-pixel-multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
                 new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color)  # add border

        return img, ratio, (dw, dh)
