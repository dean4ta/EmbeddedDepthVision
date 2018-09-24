# EmbeddedDepthVision

This project aims to extract depth information from (1) two cameras and (2) an xbox one kinect, fuse the data, and present information in real time for object avoidance.

Stereo Vision implementation is being prototyped in python with OpenCV

methods in progress:
  - Shi and Tomasi Good Features to Track Corner Detector on one image and sliding window using MSE to find correspondace points
  - SIFT point matching

Future Work:
  - Calibrate cameras
  - Migrate Python prototype code to C++ for implementation on Raspberry Pi
  - use Microsoft's C++ API to interface with Xbox One Kinect
  - Design code architecture for RPi (bare metal vs RTOS)
