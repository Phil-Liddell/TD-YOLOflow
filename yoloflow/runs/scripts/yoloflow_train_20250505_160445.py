import multiprocessing
import time
import os
import shutil
import sys

# Use Ultralytics YOLO directly instead of trying to import yolo11.py
# This avoids any import errors and works more reliably
from ultralytics import YOLO

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
    run_name = 'Banana Model'
    # Start training with proper output directory
    results = model.train(
        data=r'C:/Users/PL19437/Documents/Experiments/TDYoloflow/yoloflow/tabs/../data/dataset.yaml',
        epochs=20,
        batch=16,
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
