r"""
Object detection algorithm trained on MS-COCO dataset.
An eCAL layer has been integrated over it to seamlessly allow the passage of input and output
The exchange of messages happen over eCAL through protobuf
input protobuf message is defined in imageservice.proto
output protobuf message is defined in AlgoInterface.proto

https://github.com/karolmajek/Mask_RCNN
https://github.com/cocodataset/cocoapi
The pre-trained model weights are in releases section of this repository
The latest model can be downloaded from repo and replaced as it is in the directory.

object_class_filter.json will have the list of objects that needs to be detected and published over eCAL
 - One can refer to object_classes.txt file for list of object classes the model has been pre-trained for

Breakdown of topics.json:
    - image_request: topic name on which the image data is subscribed
    - image_response: topic name on which the output data is published(Bounding box coordinates, class names and image numpy array)
    - algo_end_response & algo_begin_response: These topics are to be defined if callbacks are used to notify when the algo has begun
            and to terminate the eCAL process. The detection API for this algo does not return any response when callback is used
    - visualization: If set to 'True' then an image would pop-up with segmentations over the image and
            the respective objects being identified by their class names
    - full_efficiency: If set to 'True' then the model would operate at maximum efficiency. It will take more time to generate output
            with more objects detected. Else it would eliminate objects with smaller masks but it would generate output in half the time

Compiling protobuf files is done by the following commands using protoc 3.5
protoc -I=.\ --python_out=.\ AlgoInterface.proto
eCAL 4.9.0, 4.9.3 and 4.9.6 has been tested with label tool 5G

Creating an executable is done by Pyinstaller module:
C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller --onefile rcnn_detection_service.py --add-data _ecal_py_3_5_x64.pyd;.
    - Once .exe is created, copy the model weights file into the directory
    - Copy topics.json, object_classes.txt and object_class_filter.json
    - Using the '--onefile' attribute greatly reduces the size of .exe by half of it eliminating the inclusion of unnecessary DLLs

GPU usage:
This model and algo is fully capable of using the GPU resources. It has been tested over Tesla K80 GPU accelerator(12 GB) over an AWS instance.
tensorflow-gpu to be installed along with CUDA and pycocotools.
Time taken for off-the-shelf inference on CPU:
    Full model efficiency - 19 seconds for 13 objects detected in image
    Half efficiency - 8 seconds for 13 objects detected in image
Time taken for off-the-shelf inference on GPU:
    Full model efficiency - 0.7 seconds for 13 objects detected in image
    Half efficiency - 0.3 seconds for 13 objects detected in image
Note: If a latest GPU is used Nvidia Titan X dual core 12 GB, the FPS will increase by a good number
"""

import ecal
import AlgoInterface_pb2
import imageservice_pb2
import sys
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
import json
import cv2
from datetime import datetime
import coco
import model as modellib

# if getattr(sys, 'frozen', False):
#     os.chdir(sys._MEIPASS)

# if getattr(sys, 'frozen', False):
#     os.chdir(sys.executable)

FROZEN_MODEL = "mask_rcnn_coco.h5"
TOPICS_JSON = 'topics.json'

with open(TOPICS_JSON) as data_file:
    json_data = json.load(data_file)
    # print(json_data)

request = str(json_data['image_request'])
response = str(json_data['image_response'])
vis_flag = True if str(json_data['visualization']) == "True" else False
full_eff_flag = True if str(json_data['full_efficiency']) == "True" else False

ecal.initialize(sys.argv, "Mask RCNN MS COCO detector")
ld_req_obj = imageservice_pb2.ImageResponse()
subscriber_obj = ecal.subscriber(topic_name=request)

lbl_response_obj = AlgoInterface_pb2.LabelResponse()
publisher_roi_obj = ecal.publisher(topic_name=response)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    if not full_eff_flag:
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512

config = InferenceConfig()
# config.print()

ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load model weights trained on MS-COCO dataset
model.load_weights(FROZEN_MODEL, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def show_detected_img(img_np, rois, class_ids, masks):
    rois_tupl_conv_lst = [tuple(ech_roi) for ech_roi in rois.tolist()]
    detected_dict = dict(zip(rois_tupl_conv_lst, class_ids.tolist()))
    # print("detected_dict :: ", detected_dict)

    for class_coordinates_dict, class_id in detected_dict.items():
        # for each_ordinate_set in class_coordinates_dict:
        image_rgb = Image.fromarray(np.uint8(img_np)).convert('RGB')
        draw = ImageDraw.Draw(image_rgb)
        # class_name_str = list(class_coordinates_dict.keys())[0]
        # print("class_name_str >> ", class_name_str)
        # print("coordinates :: ", each_ordinate_set)
        # y1, x1, y2, x2
        (top, left, bottom, right) = class_coordinates_dict
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=2, fill='red')
        class_name = class_names[class_id]
        draw.text((int(left), int(top)), str(class_name), font=None)

        np.copyto(img_np, np.array(image_rgb))
        # print("str(track_id) >> ", str(track_id))

    N = rois.shape[0]
    # print("N :: ", N)
    for i in range(N):
        mask = masks[:, :, i]
        # print("mask >> ", type(mask))
        # img_np = apply_mask(img_np, mask, color)
        alpha = 0.5
        color = (1.0, 1.0, 0.0)
        for c in range(3):
            img_np[:, :, c] = np.where(mask == 1,
                                       img_np[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                       img_np[:, :, c])
    # Display the decoded image to verify if it has been decoded
    cv2.imwrite('color_img.jpg', img_np)
    cv2.imshow('Color image', img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_objs_detected_by_cls_names(roi_dict):
    with open('object_class_filter.json') as outfile:
        obj_class_json = json.load(outfile)

    filter_obj_class_dict = {}
    for class_coordinates_tupl, class_id_mask_tupl in roi_dict.items():
        class_name = class_names[class_id_mask_tupl[0]]
        if class_name in obj_class_json:
            filter_obj_class_dict[class_coordinates_tupl] = class_id_mask_tupl
    return filter_obj_class_dict


def publish_rois(roi_dict):
    """

    :param roi_dict: It is a dictionary with key as coordinate tuple and value is a tuple
    having class id and its corresponding mask array
    :return:
    """
    if bool(roi_dict):
        # print("Total ROIs :: ", roi_dict)
        track_id = 100
        roi_dict = filter_objs_detected_by_cls_names(roi_dict)
        lbl_response_obj = AlgoInterface_pb2.LabelResponse()
        # Unpack the values
        for class_coordinates_tupl, class_id_mask_tupl in roi_dict.items():
            if class_coordinates_tupl:
                # for class_coordinates_tupl in ordinates:
                nextattr_obj = lbl_response_obj.NextAttr.add()
                # print("class_id_mask_tupl :>> ", class_id_mask_tupl[1])
                class_name = class_names[class_id_mask_tupl[0]]
                nextattr_obj.type.object_class = class_name
                print("class_name :: ", class_name)
                nextattr_obj.trackID = track_id
                # print("track id :: ", track_id)
                # Encode numpy arrays to strings
                mask = class_id_mask_tupl[1]
                # print("mask_layer >>", type(mask_layer))
                # Colour the image a little gray else it would just appear plain black
                pil_mask = Image.fromarray(np.uint8(255.0 * 0.4 * mask)).convert('L')
                mask_layer = np.array(pil_mask.convert('RGB'))

                # cv2.imwrite('color_img.jpg', mask_layer)
                # cv2.imshow('Color image', mask_layer)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                mask_layer_string = cv2.imencode('.png', mask_layer)[1].tostring()
                nextattr_obj.mask = mask_layer_string

                track_id += 1
                # x1, x2 , y1, y2 ---->> y1, x1, y2, x2
                # Create ROI object for Xmin, Ymin
                # print("class_coordinates_tupl ::", class_coordinates_tupl)
                roi_min_obj1 = nextattr_obj.ROI.add()
                roi_min_obj1.X = int(class_coordinates_tupl[1])
                roi_min_obj1.Y = int(class_coordinates_tupl[0])

                roi_min_obj2 = nextattr_obj.ROI.add()
                roi_min_obj2.X = int(class_coordinates_tupl[3])
                roi_min_obj2.Y = int(class_coordinates_tupl[0])

                # Create ROI object for Xmax, Ymax
                roi_max_obj3 = nextattr_obj.ROI.add()
                roi_max_obj3.X = int(class_coordinates_tupl[3])
                roi_max_obj3.Y = int(class_coordinates_tupl[2])

                roi_max_obj4 = nextattr_obj.ROI.add()
                roi_max_obj4.X = int(class_coordinates_tupl[1])
                roi_max_obj4.Y = int(class_coordinates_tupl[2])
        publisher_roi_obj.send(lbl_response_obj.SerializeToString())
    else:
        print("No ROIs detected..")
        lbl_response_obj = AlgoInterface_pb2.LabelResponse()
        nextattr_obj = lbl_response_obj.NextAttr.add()
        nextattr_obj.type.object_class = 'car'
        nextattr_obj.trackID = 100
        # Create ROI object for Xmin, Ymin
        roi_min_obj1 = nextattr_obj.ROI.add()
        roi_min_obj1.X = -1
        roi_min_obj1.Y = -1

        roi_min_obj2 = nextattr_obj.ROI.add()
        roi_min_obj2.X = -1
        roi_min_obj2.Y = -1

        # Create ROI object for Xmax, Ymax
        roi_max_obj3 = nextattr_obj.ROI.add()
        roi_max_obj3.X = -1
        roi_max_obj3.Y = -1

        roi_max_obj4 = nextattr_obj.ROI.add()
        roi_max_obj4.X = -1
        roi_max_obj4.Y = -1
        publisher_roi_obj.send(lbl_response_obj.SerializeToString())

#eCAL Subscriber begins here
while ecal.ok():
  # Subscribe for the message which is an image encoded through protobuf
  ret, msg, time = subscriber_obj.receive(500)
  # print("---:: ", ret, msg, time, type(msg))
  if msg is not None:
      ld_req_obj.ParseFromString(msg)
      print("received image",datetime.now())
      # print("ld_req_obj :: ", ld_req_obj)
      img_data_obj = ld_req_obj.base_image
      print("recieved_timestamp :: ", ld_req_obj.recieved_timestamp)
      img_data_str = img_data_obj
      nparr = np.fromstring(img_data_str, np.uint8)
      # print("nparr :: ", nparr)
      re_img_np_ary = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      # print("re_img_np_ary :: ", re_img_np_ary)
      img_shape = re_img_np_ary.shape
      print("img_shape ::{ ", img_shape)

      st_time = datetime.now()
      # print("model :: ", dir(model))
      results = model.detect([re_img_np_ary], verbose=1)
      r = results[0]
      ed_time = datetime.now()
      duration = ed_time - st_time
      print("detection done in ... ", duration)
      # If visualization flag is set to True in topics.json file it would pop an
      # image with masks, object classes
      if vis_flag:
        show_detected_img(re_img_np_ary, r['rois'], r['class_ids'], r['masks'])

      # Arrange the detected output in a dictionary
      # key: tuple having co-ordinates of BBox
      # value: tuple, 1st value is class id, 2nd value is a mask of numpy array
      rois_tupl_conv_lst = [tuple(ech_roi) for ech_roi in r['rois'].tolist()]
      N = r['rois'].shape[0]
      # print("N :: ", N)
      mask_obj_lst = []
      for i in range(N):
          mask = r['masks'][:, :, i]
          # print("mask >> ", type(mask))
          mask_obj_lst.append(mask)
      # detected_dict = dict(zip(rois_tupl_conv_lst, r['class_ids'].tolist()))
      clsid_mask_tupl = zip(r['class_ids'].tolist(), mask_obj_lst)
      detected_dict = dict(zip(rois_tupl_conv_lst, clsid_mask_tupl))
      # Publish the class name, bounding box coordinates and the mask numpy array
      # over prototobuf layer
      publish_rois(detected_dict)
      print("published response", datetime.now())

ecal.finalize()


