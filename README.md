# Behavioral Cloning Project

<video width="320" height="200" controls preload> 
    <source src="./video.mp4"></source> 
</video>

Overview
---
This repository contains code and a trained Deep Learning Model that tries to mimic my driving behavior on a simulated track. Hence, the name Behavioral Cloning Project.

A Convolutional Neural Network was built using Keras, and it was trained on images collected while I was driving a simulated car on a simulated track (further details are provided below). It was ensured while driving to collect data that the car stayed in the middle of the road, also images were collected while steering the car into the center of the lane from sides of the road. So, the training images represented my driving behaviour. Also recorded along with the images are velocity, braking and steering angle of the vehicle associated with each image, which were also used in training. And, the network was trained to output steering angle. 

The trained network was saved and used in driving the car autonomously around the track (steering angle was controlled by the network).
 
### Simulator
The simulator mentioned above can be downloaded from here. [Linux](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip) [Mac](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip) [Windows](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip)

+ Extract the downloaded zip file.
+ You will find an executable file inside, on launching the file the simulator will be loaded.

### Collecting Training Data
+ To collect training data on the launched simulator click "Training Mode". Then try driving few laps using arrow keys, to get accustomed with the simulator.
+ Once you are ready to collect the data click on the record button, choose the directory where you want to save, and start driving. The data will be automatically collected - frames of images from your driving.  (Below section has further details).

### Details of collected training data:

+ If the above step of collecting training data went well, you can see a folder with name "IMG" and file named "driving_log.csv" in the directory you chose to record the data.
+ The "IMG" folder has all the collected images (the frames of driving around the track).
+ NOTE: The images are clicked by three cameras mounted on the car simultaneously. There is a camera in the middle, left and right of the car. 
+ The "driving_log.csv" file has seven columns, representing path to recorded center camera image, left camera image, right camera image, steering angle, throttle, break and speed respectively. Please note, the images will be present in the "IMG" folder, and other values are collect simultaneously along with the images.

<img src="./images_readme/driving-log-output.png" width="600" height="50" />

NOTE: Collect data in such a way that the network can be trained to work well in different situations - like recovering from the sides to the center of the road, also in opposite direction of the track (as the track is a circular track).

### Included folders and files in the repo:
* video/ (COntains the frames/images collected while driving car in simulator in Autonomous Mode). Details on how to run in Autonomous mode are provided below. 
* model.py (script used to create and train the model. If you are training the model on your own data, change the variables pointing to the directory of the training data)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* Behavioral_Cloning_Writeup.pdf (a pdf file with detailed explaination on the network architecture and  more. Please do read the pdf)
* video.mp4 (a video recording of the vehicle driving autonomously around the track)

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Details About Files In This Directory

### `model.py`

Use `model.py` to train and save the trained model. It saves the model as `model.h5`. 
```sh
python model.py
```

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. 

Once the model has been saved, execute the below command and start the simulator in the Autonomous Mode. The car will drive by itself:

```sh
python drive.py model.h5
```

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 video
```

The fourth argument, `video`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

### `video.py`

```sh
python video.py video
```

Creates a video based on images found in the `video` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `video.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py video --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
