# Alzheimer Disease Prediction Brain MRI Scans Using DL

This project contains a Streamlit app for MRI-based Alzheimer stage prediction.

Dataset: <https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset>

## Files required for deployment

- `app.py` - Streamlit entrypoint
- `best_attention_hybrid_model.pth` - trained PyTorch model weights
- `requirements.txt` - Python dependencies for Streamlit Cloud
- `.gitattributes` - tracks `.pth` model files with Git LFS

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io>.
3. Click **Create app**.
4. Select this GitHub repo, branch, and set the main file path to `app.py`.
5. In **Advanced settings**, choose Python `3.12`.
6. Deploy the app and watch the logs for dependency or model-loading errors.

## Git LFS note

The model file is larger than GitHub's normal 100 MB file limit, so `.pth` files must be tracked with Git LFS:

```bash
git lfs install
git lfs track "*.pth"
```

## Notes

- Use `opencv-python-headless` on cloud deployments instead of `opencv-python`.
- The app expects `best_attention_hybrid_model.pth` to be in the same folder as `app.py`.
- This app is for research assistance only and should not replace medical diagnosis.
