# Face Alignment and Cropping

This repository contains a Python script to align and crop faces in images using facial landmarks. It uses **dlib** for face detection and landmark extraction, **OpenCV** for image processing, and **Matplotlib** for visualization.

## Features

- Detects faces and extracts 68 facial landmarks.
- Aligns the face by calculating the rotation angle between the eyes.
- Crops the image to focus on the face.
- Displays the original, rotated, and cropped images.

## Requirements

- **Python 3.x**
- Libraries:
  - `opencv-python`
  - `dlib`
  - `numpy`
  - `matplotlib`

Install the required libraries using:

```bash
pip install opencv-python dlib numpy matplotlib
```

## Usage

1. Place your input image in the `./data/` folder and name it `portrait.jpg`.
2. Download the pre-trained facial landmark model (`shape_predictor_68_face_landmarks.dat`) from dlib's website and place it in the `./data/` folder.
3. Run the script:
   ```bash
   python face_alignment.py
   ```
4. The script will display the original, rotated, and cropped images.

## Example Output
![Figure_1](https://github.com/user-attachments/assets/3b6f6987-2eb0-439c-8323-5cd318862515)
