# Industrial Steel Surface Defect Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_A_TU_COLAB)

## 1. The Challenge: Quality Beyond the Human Eye

In high-stakes manufacturing, quality control is the silent guardian of reputation. While human operators are skilled at identifying patterns, fatigue is inevitable. Machines, however, never blink.

This project is not just an image classifier; it is a **robust, automated visual inspection system** designed to identify defects in steel surfaces. My goal was to move beyond the "notebook experiment" phase and engineer a solution that is **modular, reproducible, and ready for production deployment**.

---

## 2. The Data: NEU-DET

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

## 3. Project Architecture

Just as a building needs a solid foundation, this project is structured to separate concerns. Data ingestion, model definition, and training logic are decoupled to ensure maintainability.

### Directory Structure
```text
surface_defect_detection/
├── data/                 # Raw and processed ingredients (GitIgnored)
├── models/               # Trained artifacts (GitIgnored)
├── notebooks/            # The lab: EDA and prototypes
├── src/                  # The production machinery
│   ├── config.py         # Central control panel (Hyperparameters & Paths)
│   ├── prepare_data.py   # ETL Pipeline: From raw XMLs to structured folders
│   ├── dataset.py        # Custom PyTorch Dataset (The Loader)
│   ├── model.py          # ResNet18 Architecture Definition
│   ├── train.py          # Training Loop with validation & checkpointing
│   ├── utils.py          # Metrics and logging tools
│   └── inference.py      # Production inference script
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

### The Tech Stack
* **Core:** Python 3.10+, PyTorch.
* **Model:** ResNet18 (Transfer Learning). Chosen for its optimal balance between accuracy and inference speed (FPS) for edge devices.
* **ETL:** XML Parsing (Pascal VOC) for ground-truth label extraction.

---

## 4. Methodology & Research

Training a model is like forging metal: you need the right temperature (hyperparameters) and the right technique (optimizer) to get a strong result.

1.  **Data Ingestion:** I implemented a custom ETL script (`prepare_data.py`) that parses Pascal VOC XML files directly, ensuring data integrity over relying on filenames.
2.  **Preprocessing:** Images are resized to 224x224 and normalized using ImageNet statistics (`mean=[0.485...]`, `std=[0.229...]`) to align with the pre-trained weights.
3.  **Strategy:**
    * **Transfer Learning:** Leveraged pre-trained ResNet18 weights.
    * **Optimizer:** Adam (`lr=0.001`) for adaptive learning rate management.
    * **Robustness:** Implemented Random Rotation and Color Jittering to simulate variable factory lighting conditions.

---

## 5. Performance & Results

We achieved **>99% Accuracy** on the validation set. However, in industry, "Accuracy" is vanity; "Recall" is sanity.

### Confusion Matrix
The model shows high diagonal density, confirming minimal confusion between visually similar textures (e.g., Scratches vs. Crazing).

![Confusion Matrix](https://drive.google.com/uc?id=1jszE37ccE3SCSKA4eRiSlLAeLkqFpf8e)

### Explainability (XAI)
To ensure the model isn't "cheating" (e.g., looking at background noise), I implemented **Grad-CAM**. The heatmaps confirm the network focuses specifically on the defect patterns.

![Grad-CAM](https://drive.google.com/uc?id=1rfzwPCFpQa-EOwQ58J-HjRYQkn2TUwfT)

---

## 6. How to Use & Replicate

### Prerequisites
* Git
* Python 3.8+
* A clean virtual environment (recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/surface_defect_detection.git
    cd surface_defect_detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Scenario A: Train from Scratch
The data pipeline is automated.
1.  **Download Data:** Place the `neu-surface-defect-database` zip in `data/raw`.
2.  **Run Pipeline:**
    ```bash
    python src/prepare_data.py  # ETL: XML parsing & organization
    python src/train.py         # Launches training loop
    ```

### Scenario B: Production Inference
Classify new images immediately using the trained model.

```bash
python src/inference.py --image "tests/sample_scratch.jpg" --model "models/model_best.pth"
```

**Output Example:**
```json
{
    "filename": "sample_scratch.jpg",
    "prediction": "scratches",
    "confidence": 0.9985
}
```

---

## 7. Future Roadmap

To scale this MVP into a fully operational factory solution:

* [ ] **Dockerization:** Containerize the inference script for consistent deployment.
* [ ] **API Layer:** Wrap `inference.py` in a **FastAPI** service for real-time camera integration.
* [ ] **Active Learning:** Implement a loop to feed uncertain predictions back to human labelers.

---

## Author

**Rubén Lucendo**
*AI Engineer & Product Builder*

I build systems that bridge the gap between technical complexity and business value.