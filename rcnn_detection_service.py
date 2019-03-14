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

D:\Work\2018\code\Tensorflow_code\object_detection\virtual_environments\py35_detection_rcnn\Lib\site-packages
Have kers 2.1.5

"""

# import ecal
# import AlgoInterface_pb2
# import imageservice_pb2
import sys
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
import json
import cv2
import copy
from datetime import datetime
import coco
import model as modellib
import tensorflow as tf

# if getattr(sys, 'frozen', False):
#     os.chdir(sys._MEIPASS)

# if getattr(sys, 'frozen', False):
#     os.chdir(sys.executable)



FROZEN_MODEL = "mask_rcnn_coco.h5"
TOPICS_JSON = 'topics.json'

TICKET_PATH = r'C:\Users\uidr8549\Desktop\label_tool_latest\Batch_Ticket'

# Read the label schema
label_schema_file = os.path.join(TICKET_PATH, 'ToolConfiguration', 'LabelSchema.json')
ticket_folder_name = os.path.basename(TICKET_PATH)
labeled_data_json = os.path.join(TICKET_PATH, 'LabeledData', ticket_folder_name+'_LabelData.json')
# labeled_data_json = os.path.join(TICKET_PATH, 'LabeledData')

with open(r'sample_labeledData.json') as data_file:
    sample_labeled_data_json = json.load(data_file)

# Create the labeled data json
with open(labeled_data_json, "w") as f:
    f.close()

sample_labeled_data_cpy_json = copy.deepcopy(sample_labeled_data_json)
sample_anno = sample_labeled_data_json['Sequence'][0]['DeviceData'][0]['ChannelData'][0]['AnnotatedElements'][0]
# Assign frame anno elem list to empty list

sample_labeled_data_cpy_json['Sequence'][0]['DeviceData'][0]['ChannelData'][0]['AnnotatedElements'] = []

# GPU memory allocation
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

with open(TOPICS_JSON) as data_file:
    json_data = json.load(data_file)
    # print(json_data)

request = str(json_data['image_request'])
response = str(json_data['image_response'])
vis_flag = True if str(json_data['visualization']) == "True" else False
full_eff_flag = True if str(json_data['full_efficiency']) == "True" else False

# ecal.initialize(sys.argv, "Mask RCNN MS COCO detector")
# ld_req_obj = imageservice_pb2.ImageResponse()
# subscriber_obj = ecal.subscriber(topic_name=request)
#
# lbl_response_obj = AlgoInterface_pb2.LabelResponse()
# publisher_roi_obj = ecal.publisher(topic_name=response)

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
    """
    USage : roi_dict = filter_objs_detected_by_cls_names(roi_dict)
    :param roi_dict:
    :return:
    """
    with open('object_class_filter.json') as outfile:
        obj_class_json = json.load(outfile)

    filter_obj_class_dict = {}
    for class_coordinates_tupl, class_id_mask_tupl in roi_dict.items():
        class_name = class_names[class_id_mask_tupl[0]]
        if class_name in obj_class_json:
            filter_obj_class_dict[class_coordinates_tupl] = class_id_mask_tupl
    return filter_obj_class_dict

def save_lbld_data_json(detected_dict, frame_number, time_stamp):

    if detected_dict:
        track_id = 100
        sample_frm_anno = sample_anno["FrameAnnoElements"][0]

        sample_anno_cpy = copy.deepcopy(sample_anno)
        sample_anno_cpy["FrameNumber"] = frame_number
        sample_anno_cpy["TimeStamp"] = time_stamp
        print("TimeStamp>>", time_stamp, frame_number)
        for class_coordinates_tupl, class_id_mask_tupl in detected_dict.items():
            if class_coordinates_tupl:
                class_name = class_names[class_id_mask_tupl[0]]
                # y1, x1, y2, x2
                xmin = class_coordinates_tupl[1]
                ymin = class_coordinates_tupl[0]
                xmax = class_coordinates_tupl[3]
                ymax = class_coordinates_tupl[2]

                track_id += 1

                # No mask data to be processed, to be in H5 file later
                sample_frm_anno = copy.deepcopy(sample_frm_anno)
                sample_frm_anno["category"] = class_name
                sample_frm_anno["Trackid"] = track_id
                sample_frm_anno["shape"]["x"] = [xmin, xmax, xmax, xmin]
                sample_frm_anno["shape"]["y"] = [ymin, ymax, ymin, ymax]

                sample_frm_anno["height"] = xmax - xmin
                sample_frm_anno["width"] = ymax - ymin

                sample_anno_cpy["FrameAnnoElements"].append(sample_frm_anno)
        # print("sample_anno_cpy::", sample_anno_cpy)
        sample_labeled_data_cpy_json['Sequence'][0]['DeviceData'][0]['ChannelData'][0]['AnnotatedElements'].append(sample_anno_cpy)

    with open(labeled_data_json, 'w+') as outfile:
        json.dump(sample_labeled_data_cpy_json, outfile, indent=4)

#eCAL Subscriber begins here
def perform_detection():

    img_fl_pth_lst = os.listdir(os.path.join(TICKET_PATH, 'Images'))
    frame_number = 0
    for img_name in img_fl_pth_lst:
        frame_number += 1
        img = cv2.imread(os.path.join(os.path.join(TICKET_PATH, 'Images'), img_name), cv2.IMREAD_COLOR)
        # print("img_name :: ", img_name)
        splt_content = img_name.split('_')
        channel_name = splt_content[0]
        # Assign the channel name here
        sample_labeled_data_cpy_json['Sequence'][0]['DeviceData'][0]['ChannelData'][0]["ChannelName"] = channel_name

        tm_stamp = os.path.splitext(splt_content[1])[0]
        img_encoded = cv2.imencode('.png', img)[1].tostring()
        nparr = np.fromstring(img_encoded, np.uint8)
        # print("nparr :: ", nparr)
        re_img_np_ary = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # re_img_np_ary = np.asarray(img_encoded, dtype=np.int8)
        # print("re_img_np_ary :: ", re_img_np_ary)
        img_shape = re_img_np_ary.shape
        print("img_shape ::{ ", img_shape)

        st_time = datetime.now()
        # print("model :: ", dir(model))
        results = model.detect([re_img_np_ary], verbose=1)
        # print("results::", results)
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
        # print("detected_dict::", detected_dict)
        # Publish the class name, bounding box coordinates and the mask numpy array
        save_lbld_data_json(detected_dict, frame_number, tm_stamp)

if __name__ == '__main__':
    perform_detection()

