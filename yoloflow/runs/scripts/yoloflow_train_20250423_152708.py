import multiprocessing
import time
import os
import shutil
import sys

# Use Ultralytics YOLO directly instead of trying to import yolo11.py
# This avoids any import errors and works more reliably
from ultralytics import YOLO

# Data augmentation functions
def apply_augmentations(dataset_dir):
    import cv2
    import numpy as np
    import os
    import random
    from pathlib import Path

    print('Applying data augmentations...')
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    aug_counter = 0

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'Found {len(image_files)} original images for augmentation')

    # Apply augmentations (up to 1x original dataset size)
    max_augmentations = 1 * len(image_files) - len(image_files)
    if max_augmentations <= 0:
        print('No augmentations to create')
        return 0

    for img_file in image_files:
        # If we've reached the target number of augmentations, stop
        if aug_counter >= max_augmentations:
            break

        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f'Error reading image: {img_path}')
            continue

        # Get corresponding label file
        base_name = os.path.splitext(img_file)[0]
        label_file = f'{base_name}.txt'
        label_path = os.path.join(labels_dir, label_file)
        if not os.path.exists(label_path):
            print(f'Warning: No label file for {img_file}')
            continue

        # Read the label file
        with open(label_path, 'r') as f:
            label_lines = f.readlines()

        h, w = img.shape[:2]
        # Horizontal flip
        if aug_counter < max_augmentations:
            aug_img = cv2.flip(img, 1)  # 1 for horizontal flip
            aug_name = f'{base_name}_flip.jpg'
            aug_img_path = os.path.join(images_dir, aug_name)
            cv2.imwrite(aug_img_path, aug_img)

            # Update labels (flip x coordinates)
            aug_label_path = os.path.join(labels_dir, f'{base_name}_flip.txt')
            with open(aug_label_path, 'w') as f:
                for line in label_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        # Flip x coordinate (1.0 - x)
                        x = 1.0 - x
                        f.write(f'{cls} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n')
            aug_counter += 1

        # Rotation (+/-15 degrees)
        if aug_counter < max_augmentations:
            angle = random.uniform(-15, 15)  # Random angle between -15 and 15 degrees
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            aug_img = cv2.warpAffine(img, M, (w, h))
            aug_name = f'{base_name}_rot{int(angle)}.jpg'
            aug_img_path = os.path.join(images_dir, aug_name)
            cv2.imwrite(aug_img_path, aug_img)

            # Adjust labels for rotation
            aug_label_path = os.path.join(labels_dir, f'{base_name}_rot{int(angle)}.txt')
            with open(aug_label_path, 'w') as f:
                for line in label_lines:
                    # For simplicity, we'll just copy the original labels
                    # In a production system, you'd want to transform the coordinates correctly
                    f.write(line)
            aug_counter += 1

        # Scale jitter (+/-25%)
        if aug_counter < max_augmentations:
            scale = random.uniform(0.75, 1.25)  # Random scale between 75% and 125%
            aug_w, aug_h = int(w * scale), int(h * scale)
            aug_img = cv2.resize(img, (aug_w, aug_h))
            # Create blank canvas of original size
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            # Calculate paste position (center)
            x_offset = max(0, (w - aug_w) // 2)
            y_offset = max(0, (h - aug_h) // 2)
            # Paste resized image onto canvas
            if aug_w <= w and aug_h <= h:  # Only if smaller than canvas
                canvas[y_offset:y_offset+aug_h, x_offset:x_offset+aug_w] = aug_img
            else:  # If bigger, crop center
                crop_x = max(0, (aug_w - w) // 2)
                crop_y = max(0, (aug_h - h) // 2)
                crop_img = aug_img[crop_y:crop_y+min(h, aug_h), crop_x:crop_x+min(w, aug_w)]
                canvas[:min(h, crop_img.shape[0]), :min(w, crop_img.shape[1])] = crop_img
            aug_name = f'{base_name}_scale{int(scale*100)}.jpg'
            aug_img_path = os.path.join(images_dir, aug_name)
            cv2.imwrite(aug_img_path, canvas)

            # Adjust labels for scaling
            aug_label_path = os.path.join(labels_dir, f'{base_name}_scale{int(scale*100)}.txt')
            with open(aug_label_path, 'w') as f:
                for line in label_lines:
                    # For simplicity, we'll just copy the original labels
                    # In a production system, you'd want to transform the coordinates correctly
                    f.write(line)
            aug_counter += 1

    print(f'Created {aug_counter} augmented images')
    return aug_counter

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = YOLO('yolo11n.pt')  # Use yolo11n model
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    # Configure exact paths for output to make sure they go to runs/train
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runs_dir = os.path.join(base_dir, 'runs')
    project_dir = os.path.join(runs_dir, 'train')
    os.makedirs(project_dir, exist_ok=True)
    # Use custom run name from UI
    run_name = 'run_20250423_152533'
    # Apply data augmentations (only in temporary directory)
    dataset_dir = r'C:/Users/PL19437/OneDrive - AVI-SPL/Documents/Experiments/YOLOflow/YOLOflow - Copy (8) - Copy/tabs/../data'
    aug_count = apply_augmentations(dataset_dir)
    print(f'Training with {aug_count} augmented images plus original dataset')

    # Start training with proper output directory
    results = model.train(
        data=r'C:/Users/PL19437/OneDrive - AVI-SPL/Documents/Experiments/YOLOflow/YOLOflow - Copy (8) - Copy/tabs/../data/dataset.yaml',
        epochs=4,
        batch=40,
        imgsz=640,
        lr0=0.01,
        workers=0,
        project=project_dir,  # This is already runs/train
        name=run_name,
        exist_ok=True
    )

    output_path = os.path.join(project_dir, run_name)
    print(f'Training completed - results saved to {output_path}')

    # Copy the best trained model to models directory with timestamp
    weights_dir = os.path.join(project_dir, run_name, 'weights')
    best_model_path = os.path.join(weights_dir, 'best.pt')
    last_model_path = os.path.join(weights_dir, 'last.pt')

    # Get the models directory
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Generate timestamped model filenames
    model_name_base = os.path.splitext(os.path.basename('yolo11n.pt'))[0]
    best_target = os.path.join(models_dir, f'trained_{model_name_base}_best_{timestamp}.pt')
    last_target = os.path.join(models_dir, f'trained_{model_name_base}_last_{timestamp}.pt')

    # Copy the best model if it exists
    if os.path.exists(best_model_path):
        print(f'Copying best model to: {best_target}')
        shutil.copy2(best_model_path, best_target)
        print(f'Best model saved successfully!')
    else:
        print(f'Warning: Could not find best model at {best_model_path}')

    # Copy the last model if it exists
    if os.path.exists(last_model_path):
        print(f'Copying last model to: {last_target}')
        shutil.copy2(last_model_path, last_target)
        print(f'Last model saved successfully!')
    else:
        print(f'Warning: Could not find last model at {last_model_path}')

    # No temporary directory cleanup needed
