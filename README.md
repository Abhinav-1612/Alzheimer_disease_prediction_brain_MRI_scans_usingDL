# Alzheimer Prediction Streamlit App

This project contains a Streamlit app for MRI-based Alzheimer stage prediction.

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

1. Create a GitHub repository for this project.
2. Install Git LFS, because `best_attention_hybrid_model.pth` is larger than GitHub's normal 100 MB file limit.

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes app.py requirements.txt README.md best_attention_hybrid_model.pth
git commit -m "Prepare Streamlit deployment"
git push
```

3. Go to <https://share.streamlit.io>.
4. Click **Create app**.
5. Select your GitHub repo, branch, and set the main file path to `app.py`.
6. In **Advanced settings**, choose Python `3.12`.
7. Deploy the app and watch the logs for dependency or model-loading errors.

## Notes

- Use `opencv-python-headless` on cloud deployments instead of `opencv-python`.
- The app expects `best_attention_hybrid_model.pth` to be in the same folder as `app.py`.
- This app is for research assistance only and should not replace medical diagnosis.
