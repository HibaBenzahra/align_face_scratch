import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# ------------------- STEP 1: Read the input image ------------------- #

# Load the image
img = cv2.imread("./data/portrait.jpg")

# ------------------- STEP 2: Detect face and landmarks ------------------- #

# Initialize the face detector and landmark predictor
hog_face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')  # Path to the .dat model

# Detect faces in the image
faces = hog_face_model(img, 1)

# Predict landmarks for the first face found
landmarks = landmark_model(img, faces[0])

# ------------------- STEP 3: Extract eye coordinates ------------------- #

# Extract the coordinates of the landmarks for both eyes

# Left Eye: Landmarks 36 to 41
left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]

# Right Eye: Landmarks 42 to 47
right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

# ------------------- STEP 4: Calculate eye center and rotation angle ------------------- #

# For the left eye: Compute the bounding box
lmax_x, lmax_y = max(left_eye_points, key=lambda x: (x[0], x[1]))
lmin_x, lmin_y = min(left_eye_points, key=lambda x: (x[0], x[1]))

# For the right eye: Compute the bounding box
rmax_x, rmax_y = max(right_eye_points, key=lambda x: (x[0], x[1]))
rmin_x, rmin_y = min(right_eye_points, key=lambda x: (x[0], x[1]))

# Calculate the center of each eye
lcenter_x, lcenter_y = (lmax_x + lmin_x) // 2, (lmax_y + lmin_y) // 2
rcenter_x, rcenter_y = (rmax_x + rmin_x) // 2, (rmax_y + rmin_y) // 2

# Compute the horizontal and vertical distances between the eye centers
X = np.abs(lcenter_x - rcenter_x)
Y = np.abs(lcenter_y - rcenter_y)

# Calculate the angle between the eyes
alpha = np.degrees(np.arcsin(X / np.sqrt(X**2 + Y**2)))
angle = 90 - alpha

# ------------------- STEP 5: Rotate the image based on eye alignment ------------------- #

# Get the image dimensions
h, w = img.shape[:2]

# Decide the rotation direction based on eye centers' vertical position
if lcenter_y > rcenter_x:
    # Rotate clockwise
    rotation_matrix = cv2.getRotationMatrix2D((lcenter_x, lcenter_y), angle, 1.0)
else:
    # Rotate counterclockwise
    rotation_matrix = cv2.getRotationMatrix2D((lcenter_x, lcenter_y), -angle, 1.0)

# Apply the rotation
rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))

# ------------------- STEP 6: Crop image to face only ------------------- #

# Detect the face in the rotated image
faces2 = hog_face_model(rotated_image, 1)

# Crop the image to the detected face area
cropped_img = rotated_image[faces2[0].top():faces2[0].bottom(), faces2[0].left():faces2[0].right()]

# Draw the bounding box around the face in the rotated image
cv2.rectangle(rotated_image, (faces2[0].left(), faces2[0].top()), (faces2[0].right(), faces2[0].bottom()), (255, 255, 0), 2)

# ------------------- STEP 7: Display the images ------------------- #

# Convert images from BGR (OpenCV) to RGB (Matplotlib)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

# Create a 1x3 grid of subplots
plt.figure(figsize=(10, 5))

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Display the rotated image
plt.subplot(1, 3, 2)
plt.imshow(rotated_image)
plt.title('Rotated Image')
plt.axis('off')

# Display the cropped image
plt.subplot(1, 3, 3)
plt.imshow(cropped_img)
plt.title('Cropped Image')
plt.axis('off')

# Show all images in one window
plt.show()
