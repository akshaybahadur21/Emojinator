# EmoGram [![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Emojinator/blob/master/LICENSE.md)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)
This code helps you to play the game of Instagram's Hand Gesture Challenge, with a twist.


### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

### Description
The Gesture Challenge on Instagram is a fun little game where you use your own hands to copy a set of hand emojis that are shown on your screen. It seems easy enough, except you have to imitate the gestures quickly and on-beat to a song. And if you don't get it right the first time, you'll have to try again with a brand new set of randomized hand gestures. 


### Functionalities
1) Filters to detect hand.
2) CNN for training the model.


### Python  Implementation

1) Network Used- Convolutional Neural Network

If you face any problem, kindly raise an issue

### Procedure

1) First, you have to create a gesture database. For that, run `CreateGest.py`. Enter the gesture name and you will get 2 frames displayed. Look at the contour frame and adjust your hand to make sure that you capture the features of your hand. Press 'c' for capturing the images. It will take 1200 images of one gesture. Try moving your hand a little within the frame to make sure that your model doesn't overfit at the time of training.
2) Repeat this for all the features you want.
3) Run `Emojigram_DataLoader.py` for converting the images to pickle files.
4) If you want to train the model, run 'Emojigram_Model.py'
5) Finally, run `Emojigram_App.py` for playing Hand Gesture Challenge via webcam.

### Tensorboard Visualization

For tensorboard visualization, go to the specific log directory and run this command tensorboard --logdir=. You can go to localhost:6006 for visualizing your loss function.

### Contributors

##### 1) [Akshay Bahadur](https://github.com/akshaybahadur21/)
##### 2) [Raghav Patnecha](https://github.com/raghavpatnecha)
 
 
<img src="https://github.com/akshaybahadur21/BLOB/blob/master/EmoGram.gif">





