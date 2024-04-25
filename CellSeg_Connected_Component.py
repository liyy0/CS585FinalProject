import cv2
import numpy as np
from skimage import measure, io, morphology
import os

def calculate_metrics(predicted, truth):
    # Resize images to match (use the dimensions of the gold standard image)
    predicted_resized = cv2.resize(predicted, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Calculate Pixel Accuracy
    pixel_accuracy = np.mean(predicted_resized == truth)
    
    # Calculate IoU
    intersection = np.logical_and(predicted_resized == 255, truth == 255).sum()
    union = np.logical_or(predicted_resized == 255, truth == 255).sum()
    iou = intersection / union if union != 0 else 0
    
    return pixel_accuracy, iou

# Ensure the vis directory exists
output_dir = 'vis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Directory of images
image_dir = 'BU-BIL/BU-BIL_Dataset5/RawImages'
gold_standard_dir = 'BU-BIL/BU-BIL_Dataset5/GoldStandard'

# Get list of files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

pixel_accuracies = []
ious = []

for filename in image_files:
    original_image = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
    gold_standard_image = cv2.imread(os.path.join(gold_standard_dir, 'seg_' + filename), cv2.IMREAD_GRAYSCALE)

    if original_image is None or gold_standard_image is None:
        print(f"Error reading image {filename}, skipping...")
        continue

    # Thresholding and inversion
    mean_threshold = np.mean(original_image[original_image > 0])
    _, binary_image = cv2.threshold(original_image, mean_threshold, 255, cv2.THRESH_BINARY)

    # Label and find the largest component
    labeled_image = measure.label(binary_image)
    properties = measure.regionprops(labeled_image)
    largest_component = max(properties, key=lambda x: x.area) if properties else None
    largest_component_mask = (labeled_image == largest_component.label).astype(np.uint8) * 255
    # Post-process the largest component mask to remove small objects
    bool_mask = largest_component_mask.astype(bool)
    # Remove objects smaller than a certain size (adjust the area_threshold as needed)
    cleaned_mask = morphology.remove_small_objects(bool_mask, min_size=300, connectivity=1)
    # Convert boolean array back to uint8 mask
    largest_component_mask = 255 - (cleaned_mask * 255).astype(np.uint8)

    # Save the largest component mask
    cv2.imwrite(f'{output_dir}/seg_{filename}', largest_component_mask)

    # Calculate metrics
    pa, iou = calculate_metrics(largest_component_mask, gold_standard_image)
    pixel_accuracies.append(pa)
    ious.append(iou)

# Calculate and print average metrics
mean_pixel_accuracy = np.mean(pixel_accuracies)
mean_iou = np.mean(ious)
print(f"Average Pixel Accuracy: {mean_pixel_accuracy:.4f}")
print(f"Average IoU: {mean_iou:.4f}")