# 🏗️ GCP Real Estate Classification Pipeline

This project is a complete, automated pipeline that classifies real estate buildings in Madrid as **`clasica`** or **`noclasica`** based on street and cadastral images using deep learning models. It utilizes Google Cloud Platform (GCS, Vision, Maps), Airtable, YOLOv8, and TensorFlow for image segmentation and classification.

---

## 🔧 Features

- 📥 **Airtable Integration**: Downloads property metadata including address and construction year.
- 🌍 **Street View + Cadastral Image Capture**: Fetches and stores images via Google Maps API and Spanish Cadastre API.
- 🧠 **YOLOv8 Image Segmentation**: Segments the main building from noisy images.
- 🏷️ **MobileNetV2 Classifier**: Classifies images into `clasica` or `noclasica`.
- 🧪 **Train/Validation Splits**: Automatically handles train/val split on newly segmented data.
- 🔁 **Model Retraining**: Fine-tunes the model only when necessary based on training intervals.
- 📤 **GCS Upload**: Stores segmented images and results in Parquet format on GCS.
- 📊 **Final Classification Rules**: Applies business logic based on year, confidence, and model predictions.

---

## 🗂️ Project Structure

├── main.py # Main entrypoint
├── GCPValidationClassification # Core class with all functions
├── segment.pt # YOLOv8 model (GCS)
├── models/
│ └── fine_tuned_classification_model.h5 # TensorFlow model (GCS)
├── all_fincas.parquet # Final result with AI classifications
├── df_fincas_valoradas.parquet # Intermediate result with model predictions


---

## 🚀 How It Works

1. **Download Metadata**: Pull data from Airtable.
2. **Check & Fetch Images**: Ensure each property has images, fetch if missing.
3. **Segment Buildings**: Use YOLO to segment out the main building.
4. **Train/Test Split**: Organize segmented images into structured datasets.
5. **Fine-Tune Classifier**: Retrain MobileNetV2 if the interval condition is met.
6. **Run Inference**: Apply the model to all images and generate predictions.
7. **Apply Business Rules**: Combine model output with domain logic for final classification.
8. **Export Parquet Files**: Upload enriched datasets to GCS.

---

## 🧠 Model Information

- **Segmentation**: YOLOv8 trained on manually labeled building masks.
- **Classifier**: MobileNetV2 with fine-tuning and class balancing.
- **Classes**: 
  - `Clasica`: Traditional/Old architecture
  - `Noclasica`: Modern/Non-traditional architecture

---

## 📦 Dependencies

- TensorFlow / Keras
- OpenCV / Albumentations
- Google Cloud Storage
- Google Maps API
- PyAirtable
- Unidecode / Pandas / Numpy
- YOLOv8 (Ultralytics)
- dotenv

---

## 🛠️ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate GCP
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"

# .env file setup
GOOGLE_APPLICATION_CREDENTIALS=...
CONFIG_FILE_NAME=...
AIRTABLE_API_KEY=...
AIRTABLE_BASE_NAME=...
GOOGLE_MAPS_API_KEY=...
