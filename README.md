# Multilabel Image Classification using ResNet-18

This project implements a multilabel image classification system using a
ResNet-18 deep learning model. Each image can belong to multiple classes
simultaneously. The model is trained using PyTorch and supports handling
missing labels using masking.

---

## Dataset
- Images are stored inside `images/images/`
- Labels are provided in `labels.txt`
- Each row in `labels.txt` follows the format:
  ```
  image_name label1 label2 label3 label4
  ```
- `NA` represents missing labels and is ignored during loss calculation

---

## Project Structure
```
├── DATA.py          # Custom PyTorch Dataset class
├── TRAINING.py      # Model training script
├── INFERENCE.py     # Inference on single images
├── labels.txt       # Multilabel annotations
├── README.md        # Project documentation
```

---

## Model Architecture
- Backbone: ResNet-18 (pretrained on ImageNet)
- Output Layer: Fully connected layer with 4 neurons
- Task Type: Multilabel classification
- Activation: Sigmoid (applied during inference)

---

## Loss Function
- Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
- Handles class imbalance using `pos_weight`
- Missing labels (`-1`) are masked during loss computation

---

## Training
Run the training script using:
```bash
python TRAINING.py
```

Training configuration:
- Batch size: 32
- Epochs: 10
- Optimizer: Adam
- Learning rate: 1e-4
- Output model file: `multilabel_resnet18.pth`

---

## Inference
Run inference on a single image:
```bash
python INFERENCE.py
```

Inference steps:
1. Load trained model weights
2. Preprocess input image
3. Apply sigmoid activation
4. Predict labels using threshold = 0.5

---

## Sample Output
```
PROBABILITIES --> [[0.80 0.55 0.69 0.37]]
PREDICTED LABELS --> [[1 1 1 0]]
```

---

## Key Features
- Multilabel image classification
- Handles missing labels gracefully
- Uses transfer learning with ResNet-18
- Clean and modular PyTorch codebase

---

## Future Improvements
- Add validation and evaluation metrics (F1-score, mAP)
- Support batch inference
- Add class name mapping for predictions
- Experiment with deeper architectures

---

## Author
Developed by **Mukulgopal Mandal**

## Acknowledgement

This project was developed as part of a multilabel image classification assignment and follows standard deep learning practices.
