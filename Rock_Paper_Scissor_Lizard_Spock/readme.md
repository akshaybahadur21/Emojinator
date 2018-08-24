# Rock-Paper-Scissors-Lizard-Spock [![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Emojinator/blob/master/LICENSE.md)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)
This code helps you to play the classic game of Rock-Paper-Scissors, with a twist.

### Sourcerer
[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/0)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/0)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/1)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/1)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/2)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/2)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/3)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/3)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/4)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/4)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/5)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/5)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/6)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/6)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/images/7)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Emojinator/links/7)

### Code Requirements
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

### Description
Rock–paper–scissors (also known as scissors-paper-rock or other variants) is a hand game usually played between two people, in which each player simultaneously forms one of three shapes with an outstretched hand. These shapes are "rock" (a closed fist), "paper" (a flat hand), and "scissors" (a fist with the index finger and middle finger extended, forming a V). "Scissors" is identical to the two-fingered V sign (aka "victory" or "peace sign") except that it is pointed horizontally instead of being held upright in the air. A simultaneous, zero-sum game, it has only two possible outcomes: a draw, or a win for one player and a loss for the other.

### Rules
##### Scissors cuts Paper --> Paper covers Rock --> Rock crushes Lizard --> Lizard poisons Spock --> Spock smashes Scissors --> Scissors decapitates Lizard --> Lizard eats Paper --> Paper disproves Spock --> Spock vaporizes Rock --> (and as it always has) Rock crushes Scissors 


### Functionalities
1) Filters to detect hand.
2) CNN for training the model.


### Python  Implementation

1) Network Used- Convolutional Neural Network

If you face any problem, kindly raise an issue

### Procedure

1) First, you have to create a gesture database. For that, run `CreateGest.py`. Enter the gesture name and you will get 2 frames displayed. Look at the contour frame and adjust your hand to make sure that you capture the features of your hand. Press 'c' for capturing the images. It will take 1200 images of one gesture. Try moving your hand a little within the frame to make sure that your model doesn't overfit at the time of training.
2) Repeat this for all the features you want.
3) Run `CreateCSV.py` for converting the images to a CSV file
4) If you want to train the model, run 'RPS_Model.py'
5) Finally, run `RPS_App.py` for playing Rock-Paper-Scissors-Lizard-Spock via webcam.

### Tensorboard Visualization

For tensorboard visualization, go to the specific log directory and run this command tensorboard --logdir=. You can go to localhost:6006 for visualizing your loss function.

### Contributors

##### 1) [Akshay Bahadur](https://github.com/akshaybahadur21/)
##### 2) [Raghav Patnecha](https://github.com/raghavpatnecha)
 
 
<img src="https://github.com/akshaybahadur21/Emojinator/blob/master/RPS.gif">





