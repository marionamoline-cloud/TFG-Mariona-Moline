
# Preliminary Measurements to Guide the Development of a Standing CT
### Code files

The tracking pipeline and geometric evaluations are implemented across three Jupyter Notebook files:
* `Study1.ipynb`: Handles the data processing pipeline, infrared marker segmentation, and trajectory calibration.
* `Study2.1.ipynb`: Analyzes the camera tracking error and spatial distortions using the first (ceiling-mounted) configuration.
* `Study2.2.ipynb`: Analyzes the tracking error using the second (platform-mounted) configuration and plots participant kinematic results.

Additionally, the repository includes three core Python utility scripts used for data acquisition, validation, and real-time handling:
* `groundtruth.py`: Establishes and processes the baseline reference data (ground truth) to evaluate the spatial accuracy of the camera system.
* `playback_fast.py`: Provides high-speed playback and rapid visualization of the recorded depth and spatial data streams.
* `recorder_roob 1.py`: Manages the data logging and multi-channel recording process from the infrared depth camera during the experimental trials.

### Data Access

To run the analysis scripts, the following steps must be taken:

```text
Input Data: Ensure the synchronized multi-channel spatial data arrays (.npy format) are loaded into the local working directory.
Coordinate Output: The computer vision script will automatically extract and compile the 3D metric coordinates into a structured trajectories file (.csv).
