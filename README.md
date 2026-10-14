# Bird Species Identification using Deep Learning

A supervised 200-class image-classification project using the original **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset.

**Course:** UE24CS352A — Machine Learning (mini project)  
**Team:** Bhavesh (CS115) and Bhavyaa (CS116)

Four models are compared:

1. a small CNN trained from scratch (TensorFlow);
2. MobileNetV2 transfer learning with a frozen backbone (TensorFlow);
3. MobileNetV2 with a two-stage fine-tune (TensorFlow);
4. a logistic-regression classifier on frozen DINOv2 ViT-B/14 features (PyTorch). This is the best model and the one used by the demo.

## Project contents

- `train.py`: trains a small CNN baseline or a MobileNetV2 transfer-learning model (TensorFlow).
- `train_features.py`: trains a logistic-regression classifier on frozen DINOv2 image features (PyTorch); this is the model used by the demo.
- `evaluate.py`: evaluates a saved model on the official test split and writes metrics.
- `app.py`: Streamlit image-upload demo.
- `src/data.py`: reads the original CUB metadata and builds TensorFlow datasets.
- `src/features.py`: loads the DINOv2 backbone and extracts image features.
- `requirements.txt`: Python dependencies.
- `reports/report_template.md`: report structure; fill in measured results after training.
- `reports/presentation_outline.md`: slide-by-slide presentation outline.

## 1. Download the original dataset

Download CUB-200-2011 from the official Caltech page:
https://www.vision.caltech.edu/datasets/cub_200_2011/

Extract the archive so this file exists:

`data/CUB_200_2011/images.txt`

Expected metadata files include `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, and `classes.txt`. If extraction creates a nested folder, move/rename the extracted `CUB_200_2011` folder to `data/CUB_200_2011`.

Do not upload the dataset to GitHub.

## 2. Setup on macOS

Use Python 3.12 (or 3.11). TensorFlow 2.18 does not support Python 3.13 or newer. From this project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.12` is not installed, install it with Homebrew (`brew install python@3.12`) and rerun the commands. TensorFlow runs on the CPU on macOS; PyTorch (used by `train_features.py` and the demo) uses the Apple-silicon GPU automatically.

## 3. Train the models

Run from the project root with the virtual environment activated.

Train the CNN baseline:

```bash
python train.py --model baseline --epochs 12
```

Train the transfer-learning model:

```bash
python train.py --model transfer --epochs 8
```

Train the fine-tuned model (higher accuracy, longer training):

```bash
python train.py --model finetune --epochs 20 --finetune-epochs 10
```

This runs in two stages. Stage 1 trains the classifier head on a frozen MobileNetV2 with a 1e-3 learning rate and extra augmentation (zoom, rotation, translation, contrast). Stage 2 unfreezes the top 40 backbone layers (BatchNorm layers stay frozen) and continues at 1e-5. Early stopping applies to each stage. Use `--unfreeze-layers` and `--finetune-lr` to change the second stage. The model is saved as `models/finetune.keras`.

Train the DINOv2 feature model (highest accuracy; used by the demo):

```bash
python train_features.py --backbone dinov2_base
```

This runs a frozen, self-supervised DINOv2 ViT-B/14 backbone (PyTorch, via `timm`) over every image once, averaging each image's features with those of its mirror image, and caches the features in `outputs/features/`. It then fits a standardised logistic-regression classifier, chooses the regularisation strength `C` on the validation split, and evaluates the chosen classifier once on the test split. It runs on the Apple-silicon GPU (MPS) when available, otherwise CUDA or CPU. The first run downloads the backbone weights. The classifier is saved as `models/dinov2_base.joblib`; metrics and reports are written to `outputs/` without a separate `evaluate.py` step. `--backbone dinov2_large` uses the larger ViT-L/14 backbone, which is about four times slower.

The transfer model uses MobileNetV2 initialized with ImageNet weights, then adapts its classifier to the 200 CUB species. The first run downloads pretrained weights, so an internet connection is required. Training time depends on hardware. You can lower `--epochs` to test the pipeline quickly.

Model files and class names are saved under `models/`. Training logs and history are saved under `outputs/`.

## 4. Evaluate

`train_features.py` evaluates the DINOv2 model itself. Evaluate any saved TensorFlow model against the official CUB test split:

```bash
python evaluate.py --model models/transfer.keras
```

Or:

```bash
python evaluate.py --model models/baseline.keras
python evaluate.py --model models/finetune.keras
```

The script prints top-1 accuracy, top-5 accuracy, and macro F1, and writes a classification report and confusion matrix to `outputs/`. Do not report results until the relevant training and evaluation commands have completed successfully.

## 5. Run the demo

After running `train_features.py`:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. Upload a bird image to see the predicted species and top-five class scores. The app requires `models/dinov2_base.joblib` from `train_features.py`.

## 6. Recommended execution order

1. Confirm the dataset path and run one short baseline training run.
2. Train the baseline, transfer and fine-tuned models, and run `train_features.py`.
3. Evaluate the TensorFlow models using the official test split.
4. Record the actual results in `reports/report_template.md`.
5. Test the Streamlit app with several images.
6. Complete the report and slides, then rehearse the live demo.

## 7. GitHub

Create a **private** repository and share it with the faculty and assigned TAs. From the project root:

```bash
git init -b main
git add .
git commit -m "Add bird species classification project"
gh repo create bird-species-classifier --private --source . --push
```

This creates a private repository on your GitHub account and pushes the project (requires the GitHub CLI, `gh`, logged in with `gh auth login`).

**What is in the repository:** the code, documentation, report figures (`reports/figures/`), the trained models in `models/` (`baseline.keras`, `transfer.keras`, `finetune.keras`, `dinov2_base.joblib`, `class_names.json`), and each model's test metrics and classification report in `outputs/`. With the trained models included, the demo runs straight after cloning; the DINOv2 backbone weights download automatically on first use. `dinov2_base.joblib` was saved with scikit-learn 1.9.1, so load it with a matching version.

**What is not in the repository:** the dataset, the virtual environment, cached DINOv2 features, the `*_final.keras` last-epoch copies, and other generated outputs. `.gitignore` excludes them. `.gitignore` also excludes `*.keras` files by default, so a newly trained Keras model must be added with `git add -f`.

## Measured results

Measured on an Apple M2 (8 GB) with seed 42. Validation is 900 images held out from the official training split. Test is the official test split of 5,794 images.

| Model | Train accuracy | Validation accuracy | Test top-1 | Test top-5 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| CNN baseline (12 epochs) | 2.9% | 3.8% | 3.2% | 13.0% | 0.018 |
| MobileNetV2 transfer (8 epochs) | 51.6% | 44.9% | 45.8% | 77.2% | 0.444 |
| MobileNetV2 fine-tuned (15 + 8 epochs) | 90.5% | 57.4% | 57.5% | 85.8% | 0.579 |
| DINOv2 ViT-B/14 + logistic regression (C = 0.01) | 98.2% | 88.6% | 89.8% | 98.7% | 0.897 |

For the Keras models, training accuracy is the value logged during training, with augmentation and dropout active. For DINOv2, it is measured on the training features without augmentation.

## Methodology notes

- The official CUB train/test split is preserved. A validation subset is carved from the official training split for model selection.
- The test split must not be used for tuning.
- All four models take images as input; this differs from the reference report's classical models, which used pre-annotated visual attributes.
- MobileNetV2 uses ImageNet-pretrained weights. Mention possible overlap between ImageNet pretraining data and CUB images as a limitation.
- DINOv2 was pretrained by Meta on a large curated web image collection (LVD-142M) without labels. The same overlap limitation applies.
- A model's softmax score is a model score, not a guarantee that its prediction is correct.
