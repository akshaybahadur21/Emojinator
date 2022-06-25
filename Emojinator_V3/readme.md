# Emojinator 3.0 

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Emojinator/blob/master/LICENSE.md)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

An enhanced emoji classifier for humans.


### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

### Description
Emojis are ideograms and smileys used in electronic messages and web pages. Emoji exist in various genres, including facial expressions, common objects, places and types of weather, and animals. They are much like emoticons, but emoji are actual pictures instead of typographics.


### Functionalities
1) Mediapipe Hand Detection.
2) CNN for training the model.
3) OpenCV for vision-based modeling.


### Python  Implementation

1) Network Used- Convolutional Neural Network

If you face any problem, kindly raise an issue

### Procedure

1) First, generate images using `CreateGest_V3.py.py`. Make sure that you take images from different angles.
2) Data will be stored in `/gestures` folder
3) Run `TrainEmojinator_V3.py` to train the model on the stored images.
4) Finally, run Emojinator_V3.py for using Emojinator 3.0 via webcam.

### Contributors

##### 1) [Akshay Bahadur](https://github.com/akshaybahadur21/)
##### 2) [Raghav Patnecha](https://github.com/raghavpatnecha)
 
 
<img src="https://github.com/akshaybahadur21/BLOB/blob/master/emo_v3.gif" width=1000>

### References:
 
 - [Tensorflow Object Detection API Tutorial by sentdex](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/)
 - [Raccoon Detector Dataset by Datitran](https://github.com/datitran/raccoon_dataset) 
 - [Tzuta Lin's LabelImg for dataset annotation](https://github.com/tzutalin/labelImg)
 - This implementation also took a lot of inspiration from the Victor D github repository : https://github.com/victordibia/handtracking  
