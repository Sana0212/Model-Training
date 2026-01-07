# New Project

This project contains the refactored training and testing scripts for the classifier.

## Structure

- `classifier/`: Contains the classifier implementations.
  - `multi_class_classifier.py`: ResNet101V2 based multi-class classifier.
  - `single_class_classifier.py`: MobileNetV2 based single-class classifier.
- `train_multi_class.py`: Script to train multi-class models.
- `train_single_class.py`: Script to train single-class models.
- `test.py`: Script for testing/verification.

## Usage

### Training Multi-Class Model
```
python train_multi_class.py --dir <path_to_data> --epochs 100
```

### Training Single-Class Model
```
python train_single_class.py --dir <path_to_data> --epochs 100
```
