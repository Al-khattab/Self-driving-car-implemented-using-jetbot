<h1>Self driving car using Jetson nano </h1>

This project is a small implementation of a smart car system aims to use low cost embedded systems with integrated GPU to add autonomous features like detection, tracking and following to any system.
This project will also test the efficiency of those algorithms on such a setup.

The implementation will use Jetbot hardware to make a small car with the following features:

# Features overview

### Lane following
Using the input from the CSI camera, the program uses OpenCV to detect the lanes and calculate the best trajectory to follow.


### Obstacles avoidance
This feature uses a CNN model trained to recognize obstacles in the path of the car and stops before collision,
this feature requires a lot of training data to have an effective accuracy and for real world application should be replaced with a sensor instead of a camera.


### Object tracking and following
Just like obstacles avoidance this feature uses a light CNN model trained to detect a specific object and follow it based on the object's position in the frame.
This feature is computationally extensive on this device, therefore it can only be used alone without the rest of the features unless used on a more powerful device.


### Remote operation
Using a WiFi module, it is possible to establish remote connection to the vehicle and it can be used to control it.


# Requirements
ipython==8.4.0  
ipywidgets==7.6.5  
numpy==1.21.2  
opencv_python==4.5.2.54  
torch==1.9.0+cu111  
torchvision==0.10.0+cu111  
traitlets==5.1.1  
jetbot==0.4.3  
jetpack=4.5  


# Usage
To use this project on a Jetbot hardware you need to setup the system as stated <a href="https://github.com/NVIDIA-AI-IOT/jetbot" target="_top">here</a> then  
1) Clone this repository to the system.  
2) Train the models based on your environment as stated in the tutorials.  
3) Place the trained models in the main directory.  
4) Use the masking tool to get the the values for lane detection.  
5) Run the script based on what feature you want to use.
