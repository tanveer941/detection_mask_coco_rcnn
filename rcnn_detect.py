

import ecal
import AlgoInterface_pb2
import imageservice_pb2
import sys
import os
import numpy as np
import time
from PIL import Image
from PIL import ImageDraw
import json
import cv2
from datetime import datetime

import coco
import model as modellib
import visualize

BEXIT = True
ALGO_READINESS = True
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)


FROZEN_MODEL = "mask_rcnn_coco.h5"
IMAGE_SIZE = (12, 8)
NUMBER_OF_DETECTIONS = 20
TOPICS_JSON = 'topics.json'

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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


# class InferenceConfig(coco.CocoConfig):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# config = InferenceConfig()
# # config.print()
#
# ROOT_DIR = os.getcwd()
# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# # Create model object in inference mode.
# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# # Load weights trained on MS-COCO
# model.load_weights(FROZEN_MODEL, by_name=True)


class DetectionMasksRCNN(object):

    def __init__(self):
        # Initialize eCAL
        ecal.initialize(sys.argv, "Python Mask RCNN Detector")
        # Read the JSON files
        with open(TOPICS_JSON) as data_file:
            self.json_data = json.load(data_file)
        self.vis_flag = True if str(self.json_data['visualization']) == "True" else False
        # Load the detection model
        self.load_model_detection()
        # Define the callbacks for publisher subscriber
        self.initialize_subscr_topics()
        self.initialize_publsr_topics()
        # Inform the tool that model is loaded
        # self.inform_model_loaded()
        # The callbacks will redirect to the detection function and publish ROI
        self.define_subscr_callbacks()



    def load_model_detection(self):

        # config = self.InferenceConfig()
        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        # config.print()

        ROOT_DIR = os.getcwd()
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(FROZEN_MODEL, by_name=True)

    def initialize_subscr_topics(self):
        # Initialize all the subscriber topics
        self.lt5_img_subscr_obj = ecal.subscriber(self.json_data['image_request'])
        self.lt5_finl_subscr_obj = ecal.subscriber(self.json_data['algo_end_response'])

    def initialize_publsr_topics(self):
        # Initialize all the publisher topics
        self.lt5_img_publr_obj = ecal.publisher(self.json_data['image_response'])
        self.lt5_algo_publr_obj = ecal.publisher(self.json_data['algo_begin_response'])

    def filter_obj_class(self, obj_details_lst):
        with open('object_class_filter.json') as outfile:
            obj_class_json = json.load(outfile)
        # Get class names which are set to True
        # obj_cls_true_lst = []
        # for cls_name,cls_bool in obj_class_json.items():
        #     if cls_bool == 'True':
        #         obj_cls_true_lst.append(cls_name)
        filter_obj_class_lst = []
        for each_obj_dict in obj_details_lst:
            if each_obj_dict['class'] in obj_class_json:
                filter_obj_class_lst.append(each_obj_dict)
        return filter_obj_class_lst

    def show_detected_img(self, img_np, rois, class_ids, masks):
        rois_tupl_conv_lst = [tuple(ech_roi) for ech_roi in rois.tolist()]
        detected_dict = dict(zip(rois_tupl_conv_lst, class_ids.tolist()))
        print("detected_dict :: ", detected_dict)

        for class_coordinates_dict, class_id in detected_dict.items():
            # for each_ordinate_set in class_coordinates_dict:
            image_rgb = Image.fromarray(np.uint8(img_np)).convert('RGB')
            draw = ImageDraw.Draw(image_rgb)
            # class_name_str = list(class_coordinates_dict.keys())[0]
            # print("class_name_str >> ", class_name_str)

            # print("coordinates :: ", each_ordinate_set)
            # (left, right, top, bottom) = class_coordinates_dict
            # y1, x1, y2, x2
            (top, left, bottom, right) = class_coordinates_dict
            # (right, left, bottom, top, ) = each_ordinate_set
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=2, fill='red')
            class_name = CLASS_NAMES[class_id]
            draw.text((int(left), int(top)), str(class_name), font=None)
            # draw.text((int(left), int(bottom)), class_name_str, font=None)

            np.copyto(img_np, np.array(image_rgb))
            # print("str(track_id) >> ", str(track_id))
            # cv2.putText(img_np, str(track_id), (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), lineType=cv2.LINE_AA)

        N = rois.shape[0]
        for i in range(N):
            mask = masks[:, :, i]
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

    def filter_objs_detected_by_cls_names(self, roi_dict):
        with open('object_class_filter.json') as outfile:
            obj_class_json = json.load(outfile)

        filter_obj_class_dict = {}
        for class_coordinates_tupl, class_id_mask_tupl in roi_dict.items():
            class_name = CLASS_NAMES[class_id_mask_tupl[0]]
            if class_name in obj_class_json:
                filter_obj_class_dict[class_coordinates_tupl] = class_id_mask_tupl
        return filter_obj_class_dict

    def publish_rois(self, roi_dict):
        if bool(roi_dict):
            print("Total ROIs :: ", roi_dict)
            track_id = 100
            roi_dict = self.filter_objs_detected_by_cls_names(roi_dict)
            lbl_response_obj = AlgoInterface_pb2.LabelResponse()
            # Unpack the values
            for class_coordinates_tupl, class_id_mask_tupl in roi_dict.items():
                if class_coordinates_tupl:
                    # for class_coordinates_tupl in ordinates:
                    nextattr_obj = lbl_response_obj.NextAttr.add()
                    # print("class_id_mask_tupl :>> ", class_id_mask_tupl[1])
                    class_name = CLASS_NAMES[class_id_mask_tupl[0]]
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
            self.lt5_img_publr_obj.send(lbl_response_obj.SerializeToString())
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
            self.lt5_img_publr_obj.send(lbl_response_obj.SerializeToString())

    def detect_objects(self, image_array):
        print("detect_objects begin======")
        st_time = datetime.now()
        results = self.model.detect([image_array], verbose=1)
        print("results :>> ", results)

        # cv2.imshow('Color image', image_array)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # results = model.detect([image_array], verbose=1)
        # results = [1,2,3]
        r = results[0]
        ed_time = datetime.now()
        duration = ed_time - st_time
        print("detection done in ... ", duration)

        if self.vis_flag:
            self.show_detected_img(image_array, r['rois'], r['class_ids'], r['masks'])

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

        self.publish_rois(detected_dict)


    def publ_detection_result(self, topic_name, msg, time):
        global ALGO_READINESS
        ALGO_READINESS = False
        ld_req_obj = imageservice_pb2.ImageResponse()
        lbl_response_obj = AlgoInterface_pb2.LabelResponse()
        if msg is not None:
            ld_req_obj.ParseFromString(msg)
            # print("ld_req_obj :: ", ld_req_obj)
            img_data_obj = ld_req_obj.base_image
            timestamp_img = ld_req_obj.recieved_timestamp
            print("recieved_timestamp :: ", timestamp_img)
            img_data_str = img_data_obj
            nparr = np.fromstring(img_data_str, np.uint8)
            # print("nparr :: ", nparr)
            re_img_np_ary = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # print("re_img_np_ary :: ", re_img_np_ary)
            img_shape = re_img_np_ary.shape
            print("img_shape ::{ ", img_shape)

            # print("model :: ", self.model)
            self.detect_objects(re_img_np_ary)


    def abort_algo(self, topic_name, msg, time):

        if topic_name == self.json_data['algo_end_response']:
            global BEXIT
            BEXIT = False
            # ecal.finalize()
            # exit(0)
            # print(">>>>>>>>>>>>>>>>", BEXIT)


    def inform_model_loaded(self):
        # Inform model is loaded
        # time.sleep(2)

        lbl_response_obj = AlgoInterface_pb2.AlgoState()
        lbl_response_obj.isReady = True
        self.lt5_algo_publr_obj.send(lbl_response_obj.SerializeToString())

    def define_subscr_callbacks(self):

        # For Image data
        self.lt5_img_subscr_obj.set_callback(self.publ_detection_result)
        self.lt5_finl_subscr_obj.set_callback(self.abort_algo)
        while ecal.ok() and BEXIT:
            # print("#########################", BEXIT)
            time.sleep(0.5)
            if ALGO_READINESS:
                self.inform_model_loaded()


if __name__ == "__main__":
    DetectionMasksRCNN()
