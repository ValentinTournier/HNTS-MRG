# TDT4265 Mini Project: HNTS-MRG Challenge

This repository contains the mini project for the TDT4265 course, focusing on the HNTS-MRG challenge. The project involves implementing and evaluating machine learning models to address the challenge requirements.

## Project Structure

- **`src/`**: Contains the source code for the project.
- **`data/`**: Includes datasets used for training and evaluation.
- **`results_UNet_v1/`**: Stores the results and model outputs.
- **`README.md`**: Overview of the project and instructions.

## Getting Started

1. Clone the repository:
    ```bash
    git clone git@github.com:ValentinTournier/HNTS-MRG.git
    cd HNTS-MRG
    ```
2. Create a venv:
    ```bash
    python3 -m venv .venv
    ```

3. Activate the venv:
    ```bash
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Train the model:
    ```bash
    .venv/bin/pyhtin src/train.py
    ```

5. Test the model and create visualisation:
    ```bash
    .venv/bin/pyhtin src/test.py
    ```

6. Generate loss plot:
    ```bash
    .venv/bin/pyhtin src/create_plot.py
    ```

## Report

Below, the project presentation :
[![Project Presentation](video_img.jpg)](https://youtu.be/D7G8Os90p9U)

## License

This project follows the license of the used libraries.