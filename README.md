# Industrial Steel Surface Defect Detection

## Project Overview

This project implements a **Computer Vision system** designed to automate the detection and classification of surface defects in industrial steel production.

Using the **NEU Surface Defect Database**, the model identifies six distinct types of defects: *crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches*. The solution focuses not just on model accuracy, but on delivering a **production-ready codebase** that emphasizes modularity, reproducibility, and maintainability.

## Technical Approach

The core objective was to transition from a research-oriented notebook environment to a robust software engineering artifact.

* **Architecture:** Implemented **ResNet18** using Transfer Learning. This architecture was selected to balance high accuracy with inference latency, suitable for deployment in edge-computing scenarios typical of industrial settings.
* **Framework:** Built entirely in **PyTorch**.
* **Data Pipeline:** Custom `Dataset` and `DataLoader` classes were implemented to handle image preprocessing and normalization, ensuring the pipeline is resilient to inconsistent input data.
* **Engineering Practices:**
    * **Decoupled Configuration:** All hyperparameters and paths are managed via a central configuration file, avoiding hardcoded values.
    * **Type Hinting:** Strict typing is used across the codebase to ensure interface clarity and reduce runtime errors.
    * **Checkpointing:** The training loop automatically saves the best-performing model based on validation loss, preventing overfitting.

## Project Structure

The repository follows a standard data science project structure, separating source code, configuration, and artifacts.

```text
surface_defect_detection/
├── data/                 # Raw and processed data (Not tracked by Git)
├── models/               # Model checkpoints and binaries (Not tracked by Git)
├── notebooks/            # Exploratory Data Analysis (EDA) and prototyping
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py         # Central configuration (Paths, Hyperparams)
│   ├── dataset.py        # Custom PyTorch Dataset class
│   ├── model.py          # CNN Architecture definition
│   ├── train.py          # Training loop execution
│   └── inference.py      # Prediction script for new images
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

```

## Setup and Installation

### Prerequisites

* Python 3.8+
* Virtual Environment (recommended)

### Installation Steps

1. **Clone the repository:**
```bash
git clone [https://github.com/rlucendo/surface_defect_detection.git](https://github.com/rlucendo/surface_defect_detection.git)

```


2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Data Setup:**
* Download the **NEU Surface Defect Database** from Kaggle.
* Extract the contents into the `data/` directory.
* Ensure the structure follows `data/NEU-DET/train`, `data/NEU-DET/valid`, etc., or adjust `src/config.py` accordingly.



## Usage

### Training

To train the model from scratch, execute the training script. The script logs the training progress and validation metrics to the console.

```bash
python src/train.py

```

*The best model will be saved automatically to the `models/` directory.*

### Inference

To classify a single image using the trained model:

```bash
python src/inference.py --image_path "data/sample_image.jpg"

```

## Future Improvements

To scale this MVP into a fully operational production system, the following steps are recommended:

* **Dockerization:** Containerize the application to ensure consistency across development and production environments.
* **API Deployment:** Wrap the inference logic in a **FastAPI** service to allow integration with external manufacturing systems.
* **MLOps Integration:** Implement experiment tracking (e.g., MLflow) to monitor metrics over time and manage model versioning.
* **Data Augmentation:** Implement more aggressive augmentation techniques (rotation, lighting changes) to improve robustness against real-world factory lighting conditions.

## Author

**Rubén Lucendo**
*AI Engineering & Tech Ops*


