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
  - gpu_allocation: It is a positive number which is always less than 1. This determines how much potion of GPU memory is to be utilized            by the detection alorithm            
  - oversample: This is a whole number always more than 1. This determines the number of frames to be skipped to generate the prelabel  
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

# How to use? 
Prerequisites for using the algo to detect the objects
  - Should be in the LT5G ticket folder format.
      a. The folder has a subfolder 'Images' containing the images.
      b. The images are in the format 'MFC4xxShortImageRight_1230761019084708', where MFC4xxShortImageRight
         is the channel name and 1230761019084708 is the timestamp of the image
      c. If images are not in this format, it has to be converted into the same
  - Another sub-folder 'LabeledData' will have the output JSON written into it in the format
     'ticketFolderName_LabelData.json'
  - There is no need for a 'ToolConfiguration' folder containing the schema file as this is independent of it
  - The prelabels can still be visualized in the label tool with the desired configuration along with specific attributes
     The attributes will be automatically copied into the output labeled JSON where the labeler can set each object attribute
     during labeling
  - The same labeled data JSON can be used to derive statistics pertaining to object density from frame level to the
     recording level
     
## Simple Inference  
  - From the code base the algo can be run by: python rcnn_dt_ne.py ticket/folder/path
  - From the exe the algo can be run by: rcnn_dt_ne.exe ticket/folder/path
 
# GPU usage:    
  - This model and algo is fully capable of using the GPU resources. It has been tested over Tesla K80 GPU accelerator(12 GB) over an AWS instance.         
  - tensorflow-gpu to be installed along with CUDA and pycocotools.

## Time taken for off-the-shelf inference on CPU:   
  - Full model efficiency - 19 seconds for 13 objects detected in image.      
  - Half efficiency - 8 seconds for 13 objects detected in image.  
  
## Time taken for off-the-shelf inference on GPU:      
  - Full model efficiency - 0.7 seconds for 13 objects detected in image.    
  - Half efficiency - 0.3 seconds for 13 objects detected in image.    
Note: If a latest GPU is used Nvidia Titan X dual core 12 GB, the FPS will increase by a good number

