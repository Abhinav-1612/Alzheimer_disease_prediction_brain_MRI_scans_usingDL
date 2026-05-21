# NeuroAI: Alzheimer's Disease Diagnostic Platform

Attention-Gated Hybrid Deep Learning (CNN + Swin Transformer) for Brain MRI staging.

NeuroAI is a clinically oriented diagnostic web application for multiclass staging of Alzheimer's disease using T1-weighted MRI scans. By moving beyond standalone CNNs, this system employs an Attention-Gated Hybrid Architecture that dynamically fuses local micro-textures extracted via EfficientNet-B0 with global macro-geometry mapped by a Swin-Tiny Transformer.

The model achieves 99.84% test accuracy on a strictly isolated, leak-free test vault, after resolving severe dataset imbalance using a Conditional Wasserstein GAN (W-GAN).

To bridge the gap between AI performance and medical trust, this Streamlit platform features real-time diagnostic staging, automated clinical precautions, and Dual-Spectrum Explainable AI (GradCAM) heatmaps to visually validate the model's reasoning.

## Core Features

- **4-Class Prediction:** Staging for Non-Demented, Very Mild, Mild, and Moderate Demented.
- **Hybrid Parallel Architecture:** EfficientNet-B0 for local texture extraction plus Swin-Tiny Transformer for global morphology.
- **Dynamic Fusion:** Custom Cross-Attention Squeeze-and-Excitation gate helps prevent feature imbalance.
- **Explainable AI:** Integrated GradCAM generates clinical heatmaps highlighting disease-relevant regions such as ventricular enlargement and hippocampal atrophy.
- **Dataset:** Kaggle Augmented Alzheimer MRI Dataset, mathematically balanced to 10,240 images using W-GAN.

Dataset: <https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset>

## Files Required for Deployment

- `app.py` - The main Streamlit entry point and UI dashboard.
- `best_attention_hybrid_model.pth` - The serialized PyTorch model weights.
- `requirements.txt` - Python dependencies for Streamlit Community Cloud.
- `.gitattributes` - Configures Git LFS to track the `.pth` model file.

## Run Locally

To run this application on your local machine, ensure you have Python 3.10+ installed.

Clone this repository:

```bash
git clone https://github.com/Abhinav-1612/Alzheimer_disease_prediction_brain_MRI_scans_usingDL.git
cd Alzheimer_disease_prediction_brain_MRI_scans_usingDL
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit server:

```bash
streamlit run app.py
```

## Git LFS Note

The hybrid `.pth` model file is larger than GitHub's standard 100 MB file limit. You must use Git LFS to track and push this file:

```bash
# Install Git LFS
git lfs install

# Track PyTorch model files
git lfs track "*.pth"

# Add the Git LFS configuration file
git add .gitattributes
```

## Important Notes

- **Headless OpenCV:** `requirements.txt` uses `opencv-python-headless` instead of `opencv-python`. Standard OpenCV can require system-level GUI libraries that are often unavailable in cloud server environments.
- **Model Path:** The backend expects `best_attention_hybrid_model.pth` to be located in the same root directory as `app.py`.
- **Stateless Inference:** Uploaded patient MRIs exist only temporarily in server memory and are discarded after inference.
- **Medical Disclaimer:** This application is a theoretical research tool designed to demonstrate advanced machine learning architectures. It is for educational and research assistance only and should never replace formal medical diagnosis or consultation with a certified neurologist.
