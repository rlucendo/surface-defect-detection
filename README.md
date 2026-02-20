# Industrial Steel Surface Defect Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_A_TU_COLAB)

## 1. Project summary

In the metal manufacturing industry, quality control is much more than a mere reputation safeguard, it is a critical safety system, given the high-stakes environments in which these materials are used.

This project is not just an image classifier, it is an **automated visual inspection system** designed to identify defects in steel surfaces. My goal was to engineer a solution that is **modular, reproducible, and ready for production deployment**.

---

## 2. Dataset

The model is trained on the **NEU Surface Defect Database**, detecting six specific defect types critical in metallurgy:

| Class | Description |
| :--- | :--- |
| **Crazing** | Network of fine cracks on the surface. |
| **Inclusion** | Non-metallic particles trapped in the steel. |
| **Patches** | Localized surface irregularities. |
| **Pitted** | Small craters or cavities. |
| **Rolled-in Scale** | Oxide scale pressed into the surface during rolling. |
| **Scratches** | Linear abrasions caused by mechanical contact. |

![Defect Types](https://drive.google.com/uc?id=1T3nHEOfKDG0Ut-nkRNI7ohYl-CMcokep)

---

## 3. Architecture

Since a solid foundation is just as important as the final outcome, this section outlines the project's architecture, designed specifically for modularity, scalability, and long-term maintenance. Data ingestion, model definition, and training logic are decoupled to ensure maintainability.

### Directory Structure
```text
surface_defect_detection/
├── data/                 # Raw and processed data (git ignored)
├── models/               # Trained artifacts (git ignored)
├── notebooks/            # The lab to test to train and test the model
├── src/                  # The cource code
│   ├── config.py         # Hyperparameters & paths
│   ├── prepare_data.py   # ETL pipeline
│   ├── dataset.py        # Custom PyTorch dataset
│   ├── model.py          # ResNet18 architecture definition
│   ├── train.py          # Training loop with validation & checkpointing
│   ├── utils.py          # Metrics and logging tools
│   └── inference.py      # Production inference script
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

### Tech stack
* **Core:** Python 3.10+, PyTorch.
* **Model:** ResNet18. Chosen for its optimal balance between accuracy and inference speed (FPS) for edge devices.
* **ETL:** XML parsing para asegurar un control preciso durante el entrenamiento y la evaluación.

### Technical Decisions
This README covers the high-level setup. However, every parameter in this project, was an argumented engineering choice. 

If you are interested in the "Why" behind the code, please read the **[Architecture & design decisions](ARCHITECTURE.md)** document.

---

## 4. Methodology

Aiming for an optimal balance between reliability and efficiency, I employed the following strategy to train the model:

1.  **Data ingestion:** I have implemented a custom ETL script (`prepare_data.py`) that parses the XML files directly, ensuring data integrity over relying on filenames.
2.  **Preprocessing:** Images are resized to 224x224 and normalized using ImageNet (`mean=[0.485...]`, `std=[0.229...]`) to align with the pre-trained weights.
3.  **Training:**
    * **Transfer learning:** Leveraged pre-trained ResNet18 weights.
    * **Optimizer:** Adam (`lr=0.001`) for adaptive learning rate management.
    * **Robustness:** Implemented random rotation and color jittering to simulate variable factory lighting conditions.

---

## 5. Performance & results

I had as a result **>99% accuracy** on the validation set.

### Confusion matrix
The model shows a high diagonal density, confirming minimal confusion between visually similar textures (e.g. scratches vs. crazing).

![Confusion Matrix](https://drive.google.com/uc?id=1jszE37ccE3SCSKA4eRiSlLAeLkqFpf8e)

### Explainability (XAI)
To ensure the model isn't "cheating" (e.g. looking at background noise), I have implemented **Grad-CAM** in the evaluation lab. The heatmaps confirm the network focuses specifically on the defect patterns.

![Grad-CAM](https://drive.google.com/uc?id=1rfzwPCFpQa-EOwQ58J-HjRYQkn2TUwfT)

---

## 6. How to use & replicate

### Prerequisites
* Git
* Python 3.8+
* A clean virtual environment (recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rlucendo/surface_defect_detection.git
    cd surface_defect_detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Scenario A: Train from scratch
The data pipeline is automated.
1.  **Download data:** Place the `neu-surface-defect-database` zip in `data/raw`.
2.  **Run pipeline:**
    ```bash
    python src/prepare_data.py  # ETL: XML parsing & organization
    python src/train.py         # Launches training loop
    ```

### Scenario B: Production inference
Classify new images using the trained model.

```bash
python src/inference.py --image "tests/sample_scratch.jpg" --model "models/model_best.pth"
```

**Output example:**
```json
{
    "filename": "sample_scratch.jpg",
    "prediction": "scratches",
    "confidence": 0.9985
}
```

---

## 7. Future roadmap

To scale this MVP into a fully operational factory solution:

* [ ] **Dockerization:** Containerize the inference script for consistent deployment.
* [ ] **API layer:** Wrap `inference.py` in a **FastAPI** service for real-time camera integration.
* [ ] **Active learning:** Implement a loop to feed uncertain predictions back to human labelers.

---

## Author

**Rubén Lucendo**  
*AI Engineer & Product Builder*

Building systems that bridge the gap between theory and business value.