# Industrial Steel Surface Defect Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 1. The Challenge: Quality Beyond the Human Eye

In high-stakes manufacturing, quality control is the silent guardian of reputation. While human operators are skilled at identifying patterns, fatigue is inevitable. Machines, however, never blink.

This project is not just an image classifier; it is a **robust, automated visual inspection system** designed to identify six specific types of defects in steel surfaces: *crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches*.

My goal was to move beyond the "notebook experiment" phase and engineer a solution that is **modular, reproducible, and ready for production deployment**.

---

## 2. Project Architecture

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
* **Data Format:** NEU-DET Dataset (Pascal VOC annotations).

---

## 3. The Training Process (Research & Validation)

Training a model is like forging metal: you need the right temperature (hyperparameters) and the right technique (optimizer) to get a strong result.

### Methodology
1.  **Data Ingestion:** I implemented a custom ETL script (`prepare_data.py`) that parses Pascal VOC XML files. This ensures we rely on ground-truth labels rather than fragile filenames.
2.  **Preprocessing:** Images are resized to 224x224 and normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
3.  **Strategy:**
    * **Transfer Learning:** Leveraged pre-trained weights to accelerate convergence.
    * **Optimizer:** Adam (`lr=0.001`) for adaptive learning rate management.
    * **Checkpointing:** The system automatically saves the model only when Validation Loss improves, preventing overfitting.

### Performance & Metrics
We achieved **>99% Accuracy** on the validation set. However, accuracy can be deceiving. In an industrial setting, a False Negative (missing a defect) is costly.

* **Confusion Matrix:** The model shows high diagonal density, confirming minimal confusion between similar textures like *scratches* and *crazing*.
* **Explainability (Grad-CAM):** To ensure the model isn't "cheating" (looking at background noise), I implemented Grad-CAM visualization. The heatmaps confirm the network focuses specifically on the defect patterns.

> **View the Training Log:** You can inspect the full training run, metrics, and visualizations in the accompanying [Colab Notebook](LINK_A_TU_COLAB).

---

## 4. How to Use & Replicate

Whether you want to retrain the system from scratch or use the pre-trained model for inference, follow these steps.

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

### Scenario A: I want to Train the Model
The data pipeline is automated. You don't need to manually sort files.

1.  **Download Data:** Place the `neu-surface-defect-database` zip from Kaggle into `data/raw` (or use the Kaggle API as shown in the notebook).
2.  **Run the ETL Pipeline:**
    This script reads the XMLs and organizes images into class folders.
    ```bash
    python src/prepare_data.py
    ```
3.  **Launch Training:**
    ```bash
    python src/train.py
    ```
    *The best model will be saved to `models/model_best.pth`.*

### Scenario B: I want to Predict (Inference)
If you have a trained model (`.pth`), you can classify new images immediately.

```bash
python src/inference.py --image "path/to/test_image.jpg" --model "models/model_best.pth"
```

**Output Example:**
```json
{
    "filename": "test_image.jpg",
    "prediction": "scratches",
    "confidence": 0.9985
}
```

---

## 5. Future Roadmap

To scale this MVP into a fully operational factory solution, the next steps are:

* [ ] **Dockerization:** Containerize the inference script for consistent deployment.
* [ ] **API Layer:** Wrap `inference.py` in a **FastAPI** service for real-time camera integration.
* [ ] **Data Augmentation:** Implement rotation and lighting jitter to handle variable factory lighting conditions.

---

## Author

**Rubén Lucendo**
*AI Engineer & Product Builder*

I build systems that bridge the gap between technical complexity and business value.