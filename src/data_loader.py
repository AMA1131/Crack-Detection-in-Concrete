import cv2
import os



def load_data(positive_path, negative_path, max_images=500):
    """
    Load images from positive and negative directories and return a list of tuples (image, label).
    
    Args:
        positive_path (str): Path to the directory containing positive images.
        negative_path (str): Path to the directory containing negative images.

    Returns:
        list: A list of tuples (image, label) where image is a loaded image and label is 1 for positive and 0 for negative.
    """
    image_extensions = (".jpg", ".jpeg", ".png")
    images = []
    labels = []
    
    # Load positive images
    for filename in os.listdir(positive_path)[:max_images]:  # Limit to first max_images images for testing
        if filename.endswith(image_extensions):
            
            img_path = os.path.join(positive_path, filename)
            image = cv2.imread(img_path)

            if image is not None:
                images.append(image)
                labels.append(1)
    
    # Load negative images
    for filename in os.listdir(negative_path)[:max_images]:  # Limit to first max_images images for testing
        if filename.endswith(image_extensions):
            
            img_path = os.path.join(negative_path, filename)
            image = cv2.imread(img_path)

            if image is not None:
                images.append(image)
                labels.append(0)

    return images, labels