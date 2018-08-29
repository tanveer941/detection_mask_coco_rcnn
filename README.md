# detection_mask_coco_rcnn
Object detection with segmentation using pre-trained model over MS-COCO dataset
# MS-COCO object detection and segmentation over eCAL
Mask RCNN object detection uses frozen model pre-trained on MSCOCO dataset

# Object detection algorithm trained on MS-COCO dataset.  
  - An eCAL layer has been integrated over it to seamlessly allow the passage of input and output.   
  - The exchange of messages happen over eCAL through protobuf  
  - input protobuf message is defined in imageservice.proto  
  - output protobuf message is defined in AlgoInterface.proto   
  - https://github.com/karolmajek/Mask_RCNN  
  - https://github.com/cocodataset/cocoapi  
  - The pre-trained model weights are in releases section of this repository.   
  - The latest model can be downloaded from repo and replaced as it is in the directory.
  
object_class_filter.json will have the list of objects that needs to be detected and published over eCAL 
  - One can refer to object_classes.txt file for list of object classes the model has been pre-trained for

# Breakdown of topics.json:   
  - image_request: topic name on which the image data is subscribed 
  - image_response: topic name on which the output data is published(Bounding box coordinates, class names and image numpy array) 
  - algo_end_response & algo_begin_response: These topics are to be defined if callbacks are used to notify when the algo has begun 
            and to terminate the eCAL process. The detection API for this algo does not return any response when callback is used 
  - visualization: If set to 'True' then an image would pop-up with segmentations over the image and the respective objects being                 identified by their class names
  - full_efficiency: If set to 'True' then the model would operate at maximum efficiency. It will take more time to generate output 
           with more objects detected. Else it would eliminate objects with smaller masks but it would generate output in half the time 
  
Compiling protobuf files is done by the following commands using protoc 3.5    
  - protoc -I=.\ --python_out=.\ AlgoInterface.proto      
  - eCAL 4.9.0, 4.9.3 and 4.9.6 has been tested with label tool 5G      

# Creating an executable by Pyinstaller module: 
C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller --onefile rcnn_detection_service.py --add-data _ecal_py_3_5_x64.pyd;.  
  - Once .exe is created, copy the model weights file into the directory.   
  - Copy topics.json, object_classes.txt and object_class_filter.json  
  - Using the '--onefile' attribute greatly reduces the size of .exe by half of it eliminating the inclusion of unnecessary DLLs.   
  
# GPU usage:    
  - This model and algo is fully capable of using the GPU resources. It has been tested over Tesla K80 GPU accelerator(12 GB) over an AWS instance.         
  - tensorflow-gpu to be installed along with CUDA and pycocotools.

Time taken for off-the-shelf inference on CPU:   
  - Full model efficiency - 19 seconds for 13 objects detected in image.      
  - Half efficiency - 8 seconds for 13 objects detected in image.  
  
Time taken for off-the-shelf inference on GPU:      
  - Full model efficiency - 0.7 seconds for 13 objects detected in image.    
  - Half efficiency - 0.3 seconds for 13 objects detected in image.    
Note: If a latest GPU is used Nvidia Titan X dual core 12 GB, the FPS will increase by a good number
