# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to work with detection models."""

import collections
import platform

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.

    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    @property
    def width(self):
        """Returns bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """Returns bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """Returns bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Returns whether bounding box is valid or not.

        Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
        width >= 0 and height >= 0.
        """
        return self.width >= 0 and self.height >= 0

    @property
    def centroid(self):
        return ((self.xmax + self.xmin) / 2, (self.ymax + self.ymin) / 2)

    @property
    def np_arr(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)

    def translate(self, dx, dy):
        """Returns translated bounding box."""
        return BBox(xmin=dx + self.xmin,
                    ymin=dy + self.ymin,
                    xmax=dx + self.xmax,
                    ymax=dy + self.ymax)

    def map(self, f):
        """Returns bounding box modified by applying f for each coordinate."""
        return BBox(xmin=f(self.xmin),
                    ymin=f(self.ymin),
                    xmax=f(self.xmax),
                    ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        """Returns the intersection of two bounding boxes (may be invalid)."""
        return BBox(xmin=max(a.xmin, b.xmin),
                    ymin=max(a.ymin, b.ymin),
                    xmax=min(a.xmax, b.xmax),
                    ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        """Returns the union of two bounding boxes (always valid)."""
        return BBox(xmin=min(a.xmin, b.xmin),
                    ymin=min(a.ymin, b.ymin),
                    xmax=max(a.xmax, b.xmax),
                    ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        """Returns intersection-over-union value."""
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)

    def similar_to(self, other_arr, thresh=20):
        dist = np.linalg.norm(self.np_arr - other_arr)
        if dist > thresh:
            return False

        return True

def create_obj(bbox, class_id, score):
    return Object(
        id=class_id,
        score=round(score, 2),
        bbox=BBox(
            xmin=bbox[0],
            ymin=bbox[1],
            xmax=bbox[2],
            ymax=bbox[3]))


def get_distance(bbox1, bbox2):
    np_bbox1 = np.array(bbox1)
    np_bbox2 = np.array(bbox2)
    return np.linalg.norm(np_bbox1 - np_bbox2)


def input_size(interpreter):
    """Returns input image size as (width, height) tuple."""
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return width, height


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def set_input(interpreter, size, resize):
    """Copies a resized and properly zero-padded image to the input tensor.

    Args:
      interpreter: Interpreter object.
      size: original image size as (width, height) tuple.
      resize: a function that takes a (width, height) tuple, and returns an RGB
        image resized to those dimensions.
    Returns:
      Actual resize ratio, which should be passed to `get_output` function.
    """
    width, height = input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    tensor[:h, :w] = np.reshape(resize((w, h)), (h, w, channel))
    return scale, scale


def output_tensor(interpreter, i):
    """Returns output tensor view."""
    tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
    return np.squeeze(tensor)


def get_res_off_interp(interpreter):
    """Returns list of detected objects."""
    boxes = output_tensor(interpreter, 0)
    class_ids = output_tensor(interpreter, 1)
    scores = output_tensor(interpreter, 2)
    # count = int(output_tensor(interpreter, 3))
    width, height = input_size(interpreter)

    return boxes, scores, class_ids, width, height


def scale_box(box, sx, sy):
    ymin, xmin, ymax, xmax = box
    return (
        int(sx * xmin),
        int(sy * ymin),
        int(sx * xmax - sx * xmin),
        int(sy * ymax - sy * ymin)
    )


def parse_raw_output(boxes, scores, class_ids, score_threshold, width, height, img_scale=(1.0, 1.0)):
    image_scale_x, image_scale_y = img_scale
    sx, sy = width / image_scale_x, height / image_scale_y

    count = len(boxes)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=float(scores[i]),
            bbox=BBox(xmin=xmin,
                      ymin=ymin,
                      xmax=xmax,
                      ymax=ymax).scale(sx, sy).map(int))

    return [make(i) for i in range(count) if scores[i] >= score_threshold]


EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

    Args:
      path: path to label file.
      encoding: label file encoding.
    Returns:
      Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                 {'device': device[0]} if device else {})
        ])


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def append_objs_to_img(cv2_im, objs, labels):
    for obj in objs.values():
        x0, y0, x1, y1 = obj.bbox
        percent = int(100 * obj.confidence)
        label = '{}% {} - {}'.format(percent,
                                     labels.get(obj.class_id, obj.class_id),
                                     obj.id[:4])

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im
