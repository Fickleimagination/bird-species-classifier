# Bird Species Identification Using Deep Learning

**Team members:** [Name 1, USN] and [Name 2, USN]  
**Course:** UE24CS352A — Machine Learning

## 1. Problem statement
Identify the bird species shown in an input photograph. This is a supervised, fine-grained, 200-class image-classification problem.

## 2. Dataset
We used the original Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset. It contains 11,788 images representing 200 bird species. We used the dataset's official training/test partition and carved a validation subset from the official training partition. Describe the metadata files and preprocessing used.

## 3. Methodology
Describe the image resizing, augmentation, label encoding, loss function, optimizer, batch size, epoch count, and validation strategy. Compare:
- CNN baseline trained from scratch (TensorFlow).
- MobileNetV2 transfer-learning model initialized with ImageNet weights, frozen backbone (TensorFlow).
- MobileNetV2 two-stage fine-tune: head training with stronger augmentation, then the top 40 backbone layers unfrozen at a low learning rate (TensorFlow).
- Logistic regression on frozen, self-supervised DINOv2 ViT-B/14 features with flip averaging; regularisation strength chosen on the validation split (PyTorch + scikit-learn).

Explain why the transfer-learning approach is appropriate for a limited dataset and computational budget, and why a stronger pretrained backbone helped more than regularising MobileNetV2. Do not claim that ImageNet or DINOv2 pretraining is independent of CUB; discuss possible dataset overlap as a limitation.

## 4. Implementation
Describe `src/data.py`, `src/features.py`, `train.py`, `train_features.py`, `evaluate.py`, and `app.py`. Include a screenshot of the working upload-and-predict interface. Give exact commands used to train, evaluate, and run the demo.

## 5. Results
Fill in only after running evaluation on the official test split.

| Model | Validation accuracy | Test top-1 accuracy | Test top-5 accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| CNN baseline | [measured result] | [measured result] | [measured result] | [measured result] |
| MobileNetV2 transfer | [measured result] | [measured result] | [measured result] | [measured result] |
| MobileNetV2 fine-tuned | [measured result] | [measured result] | [measured result] | [measured result] |
| DINOv2 + logistic regression | [measured result] | [measured result] | [measured result] | [measured result] |

Insert training/validation accuracy curves and discuss errors or confusing species. Do not use the test split to tune hyperparameters.

## 6. Conclusion and limitations
State which model performed better on the held-out test split and by how much. Discuss fine-grained visual similarities, image/background variation, computational limits, and possible overlap between the pretraining data (ImageNet, LVD-142M) and CUB. Propose a realistic next step.

## References
- Wah, C. et al. *The Caltech-UCSD Birds-200-2011 Dataset*. California Institute of Technology, 2011.
- Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR, 2018.
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR, 2024.
- TensorFlow/Keras documentation: https://www.tensorflow.org/
- PyTorch documentation: https://pytorch.org/ and PyTorch Image Models (timm): https://github.com/huggingface/pytorch-image-models
