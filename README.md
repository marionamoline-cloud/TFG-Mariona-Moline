# Preliminary Measurements to Guide the Development of a Standing CT

### Project Overview
This project analyzes human postural sway and movement patterns on an Arduino-controlled rotating platform to validate a low-cost, standing Walk-Through PET-CT scanner design.
### Requirements & Hardware Architecture
The experimental tracking framework developed in this study consists of:
* **Platform Control:** Motorized rotating platform driven by a stepper motor and controlled via an **Arduino Uno** microcontroller using automated kinematic scripts.
* **Motion Capture (MoCap) System:** An **Orbbec Femto Mega** depth camera operating as a Time-of-Flight (ToF) sensor to track real-time 3D displacements.

### Repository Structure
This repository contains the following core Jupyter Notebooks:
* `Study1.ipynb`: Data processing pipeline, computer vision scripts for infrared marker segmentation, and platform trajectory calibration.
* `Study2.1.ipynb`: Scripts to analyze the tracking error and spatial distortions of the camera using the first (ceiling-mounted) configuration.
* `Study2.2.ipynb`: Scripts to analyze the camera tracking error using the second (platform-mounted) configuration, alongside participant kinematic evaluation plots (MAE, MSE, and 3D Euclidean displacement graphs).

---
*Note: This study complies with the General Ethical Protocol of Ghent University and adheres to European GDPR regulations.*
