
# Preliminary Measurements to Guide the Development of a Standing CT

This project investigates patient movement patterns and postural sway on a motorized rotating platform. It quantifies upper-body landmark displacements using an infrared depth camera system to prevent image artifacts during vertical CT tomographic acquisitions.

## Requirements

### Code files

The tracking pipeline and geometric evaluations are implemented across three Jupyter Notebook files:
* `Study1.ipynb`: Handles the data processing pipeline, infrared marker segmentation, and trajectory calibration.
* `Study2.1.ipynb`: Analyzes the camera tracking error and spatial distortions using the first (ceiling-mounted) configuration.
* `Study2.2.ipynb`: Analyzes the tracking error using the second (platform-mounted) configuration and plots participant kinematic results.

### Data Access

To run the analysis scripts, the following steps must be taken:

```text
Input Data: Ensure the synchronized multi-channel spatial data arrays (.npy format) are loaded into the local working directory.
Coordinate Output: The computer vision script will automatically extract and compile the 3D metric coordinates into a structured trajectories file (.csv).
