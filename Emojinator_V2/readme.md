# Emojinator V2 [![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Emojinator/blob/master/LICENSE.md)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)
This code helps you classify different hand gestures as emojis using Object Detection API.


### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

### Description
Emojis are ideograms and smileys used in electronic messages and web pages. Emoji exist in various genres, including facial expressions, common objects, places and types of weather, and animals. They are much like emoticons, but emoji are actual pictures instead of typographics.


### Functionalities
1) Tensorflow's object detection API for training SSD with MobilnetV1
1) Filters to detect hand.
2) CNN for training the model.


### Python  Implementation

1) Network Used- Convolutional Neural Network

If you face any problem, kindly raise an issue

### Procedure

1) First, generate images using `get_hand_images.py`. Make sure that you take images from different angles.
2) Data will be stored in `/data` folder
3) Annotate the data using the [labelImg program](https://github.com/tzutalin/labelImg) by tzutalin.
4) For training, you can use my uploaded model in `/hand_detection_inference_graph` or you can train your own model using [sentdex turorial](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/)
5) Finally, run `Emojinator_V2.py` for using Emojinator 2.0 via webcam.

### Tensorboard Visualization

For tensorboard visualization, go to the specific log directory and run this command tensorboard --logdir=. You can go to localhost:6006 for visualizing your loss function.

### Contributors

##### 1) [Akshay Bahadur](https://github.com/akshaybahadur21/)
##### 2) [Raghav Patnecha](https://github.com/raghavpatnecha)
 
 
<img src="emo_v2.gif">

### References:
 
 - [Tensorflow Object Detection API Tutorial by sentdex](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/)
 - [Raccoon Detector Dataset by Datitran](https://github.com/datitran/raccoon_dataset) 
 - [Tzuta Lin's LabelImg for dataset annotation](https://github.com/tzutalin/labelImg)
 - This implementation also took a lot of inspiration from the Victor D github repository : https://github.com/victordibia/handtracking  
