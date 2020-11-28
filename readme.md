# EasyNN_ROS

[![license - apache 2.0](https://img.shields.io/:license-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[![support level: community](https://img.shields.io/badge/support%20level-community-lightgray.png)](http://rosindustrial.org/news/2016/10/7/better-supporting-a-growing-ros-industrial-software-platform)

__Requirements__

This python-based package requires torch 1.7.0+, OpenCV 4.4+, PyQt5, and other libraries. The necessary libraries will be installed automatically by pip during the catkin_make operation. See [requirements](requirements.txt) to view full list.

__Installation__

Copy the project files to your catkin workspace `src` folder and run `cankin_make`.

___Training your network___

To train your network use `main` script. Yes, wiser naming is still to come. 

```text
rosrun easynn_ros main.py
```

You will be provided with GUI for creating classes and image annotation with bounding boxes. When the training is over, save --weights file to your project.

___Run under ROS___

To run a neural network use `main` script with the name of an image topic as a parameter.

```text
rosrun easynn_ros imageSubscriver.py --weights (.pt file with weights) --source (your topic to read data from name)
```