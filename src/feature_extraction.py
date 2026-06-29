import cv2
import numpy as np

def create_hog_descriptor():
    
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog_descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog_descriptor

def extract_hog_features(images):

    hog_descriptor = create_hog_descriptor()
    feature_vectors = []

    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (3,3), 0)
        # Resize the image to the size expected by the HOG descriptor
        resized_image = cv2.resize(blurred_image, (64, 64))
        
        # Compute HOG features
        hog_features = hog_descriptor.compute(resized_image)
        # Flatten the HOG feature because it might return a 2D array and we want a 1D array for each image
        hog_features_flattened = hog_features.flatten()

        feature_vectors.append(hog_features_flattened)

    # The machine learning models learn better with numerical data --> convert the list of feature vectors to a numpy array of type float32 which is lighter thanfloat64
    return np.array(feature_vectors, dtype=np.float32)
    