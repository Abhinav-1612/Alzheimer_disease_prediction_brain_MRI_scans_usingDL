import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="NeuroAI Diagnostic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');

/* ── Root / Background ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #020818 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Animated grid background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: gridMove 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes gridMove {
    0%   { background-position: 0 0; }
    100% { background-position: 50px 50px; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030d1a 0%, #050f20 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #00d4ff !important; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.8rem, 4vw, 3rem);
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #0088cc 40%, #7c3aed 80%, #00d4ff 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
    text-align: center;
    letter-spacing: 2px;
    margin: 0 0 0.3rem 0;
    text-shadow: none;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Subtitle ── */
.hero-sub {
    text-align: center;
    color: #64748b;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    margin-bottom: 2rem;
}

/* ── Section headings ── */
.section-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #00d4ff;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 1.2rem;
    transition: border-color 0.3s;
}
.glass-card:hover { border-color: rgba(0,212,255,0.3); }

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.badge-healthy  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.badge-caution  { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid #f59e0b; }
.badge-warning  { background: rgba(239,68,68,0.15);   color: #ef4444; border: 1px solid #ef4444; }
.badge-critical { background: rgba(168,85,247,0.15);  color: #a855f7; border: 1px solid #a855f7; }

/* ── Diagnosis card ── */
.diagnosis-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.06) 0%, rgba(124,58,237,0.06) 100%);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.diagnosis-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff);
    background-size: 200% 100%;
    animation: shimmer 2.5s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.diagnosis-card h3 { margin: 0 0 0.5rem 0; color: #f1f5f9; font-size: 1.1rem; }
.diagnosis-card ul { padding-left: 1.2rem; margin: 0.5rem 0 0 0; }
.diagnosis-card li { color: #94a3b8; margin-bottom: 0.35rem; font-size: 0.9rem; line-height: 1.5; }

/* ── Warning card ── */
.warning-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.06) 0%, rgba(245,158,11,0.04) 100%);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.warning-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #ef4444);
    background-size: 200% 100%;
    animation: shimmer 2.5s linear infinite;
}
.warning-card h3 { margin: 0 0 0.5rem 0; color: #f1f5f9; font-size: 1.1rem; }
.warning-card ul { padding-left: 1.2rem; margin: 0.5rem 0 0 0; }
.warning-card li { color: #94a3b8; margin-bottom: 0.35rem; font-size: 0.9rem; line-height: 1.5; }

/* ── Probability bar ── */
.prob-bar-wrap { margin-bottom: 0.6rem; }
.prob-label { display: flex; justify-content: space-between; margin-bottom: 3px; font-size: 0.8rem; }
.prob-label span:first-child { color: #94a3b8; }
.prob-label span:last-child  { color: #00d4ff; font-weight: 700; }
.prob-track {
    height: 8px; border-radius: 4px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}
.prob-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    transition: width 0.8s ease;
}

/* ── Divider ── */
.neon-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
    margin: 1.5rem 0;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(0,212,255,0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,212,255,0.6) !important; }

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00d4ff 0%, #0088cc 50%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 1.5px !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Streamlit spinner */
[data-testid="stSpinner"] > div { border-top-color: #00d4ff !important; }

/* Matplotlib figure */
[data-testid="stImage"] { border-radius: 12px; overflow: hidden; }

/* Progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg, #00d4ff, #7c3aed) !important; }

/* Metric cards override */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,212,255,0.12) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Orbitron', sans-serif !important; }

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #020818; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.6); }

/* Caption text */
.stCaption, caption { color: #475569 !important; }

/* Image caption */
[data-testid="stImageCaption"] { color: #64748b !important; font-size: 0.8rem !important; text-align: center !important; }

/* Selectbox / inputs */
[data-baseweb="select"] { background: rgba(255,255,255,0.04) !important; border-color: rgba(0,212,255,0.2) !important; }

</style>
""", unsafe_allow_html=True)


# 3. MODEL ARCHITECTURE

class AttentionHybridModel(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6):
        super(AttentionHybridModel, self).__init__()

        effnet = models.efficientnet_b0(weights=None)
        self.cnn_backbone = effnet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        swin = models.swin_t(weights=None)
        self.swin_backbone = swin.features

        self.cnn_proj  = nn.Linear(1280, 512)
        self.swin_proj = nn.Linear(768,  512)

        self.attention_gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, 1024),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x_cnn  = self.cnn_pool(self.cnn_backbone(x)).view(x.size(0), -1)
        x_swin = self.swin_backbone(x).permute(0, 3, 1, 2)
        x_swin = torch.mean(x_swin, dim=(2, 3))

        p_cnn  = self.cnn_proj(x_cnn)
        p_swin = self.swin_proj(x_swin)

        combined          = torch.cat((p_cnn, p_swin), dim=1)
        attention_weights = self.attention_gate(combined)
        gated_features    = combined * attention_weights

        return self.classifier(gated_features)


# ─────────────────────────────────────────────
# 4. CACHED MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionHybridModel(num_classes=4, dropout_rate=0.6).to(device)
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_attention_hybrid_model.pth')
        raw_state_dict = torch.load(model_path, map_location=device)
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
        model.load_state_dict(clean_state_dict)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None, device


model, device = load_model()

# ─────────────────────────────────────────────
# 5. MRI VALIDATION HEURISTIC
# ─────────────────────────────────────────────
def is_valid_brain_mri(img: Image.Image):
    """
    Lightweight heuristic to detect if an image is likely a brain MRI scan.
    Checks three properties characteristic of MRI scans:
      1. Near-grayscale coloring  (MRIs have very low colour saturation)
      2. Significant dark background (MRIs have large black regions around the brain)
      3. Roughly square aspect ratio (typical MRI slice dimensions)
    Returns (bool, str) — (is_valid, reason_message)
    """
    img_rgb = np.array(img.convert('RGB')).astype(np.float32)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # ── Check 1: Channel similarity (grayscale test) ──────────────────
    # In a greyscale MRI, R ≈ G ≈ B for every pixel.
    # Raised threshold to 30 to tolerate JPEG colour artifacts and
    # slight colour casts in real-world MRI exports.
    mean_color_diff = (np.mean(np.abs(r - g)) +
                       np.mean(np.abs(r - b)) +
                       np.mean(np.abs(g - b))) / 3
    if mean_color_diff > 30:
        return False, (
            "The image appears to be a **colour photograph**, not a brain MRI scan. "
            "MRI scans are greyscale (R ≈ G ≈ B channels)."
        )

    # ── Check 2: Dark background ratio ───────────────────────────────
    # Brain MRI slices have a dark background around the brain.
    # Threshold raised to intensity < 55 so navy/dark-grey backgrounds
    # (common in saved MRI screenshots) are counted as "dark".
    # Minimum ratio lowered to 5 % to handle tightly-cropped scans.
    gray = np.mean(img_rgb, axis=2)
    dark_ratio = float(np.sum(gray < 55)) / gray.size
    if dark_ratio < 0.05:
        return False, (
            "The image lacks the **dark background** typical of brain MRI scans. "
            "Please upload a T1-weighted axial MRI slice with a dark border."
        )

    # ── Check 3: Aspect ratio ─────────────────────────────────────────
    # MRI slices are nearly square; very elongated images are likely
    # panoramas, screenshots, or other non-medical images.
    w, h = img.size
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 3.5:
        return False, (
            f"The image aspect ratio ({aspect:.1f}:1) is too extreme for a brain MRI scan. "
            "MRI slices are typically square or near-square."
        )

    return True, ""


# ─────────────────────────────────────────────
# 6. CLINICAL DATA
# ─────────────────────────────────────────────
classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

clinical_advice = {
    'NonDemented': {
        'status': 'Healthy — No Dementia Detected',
        'icon': '✅',
        'card_class': 'diagnosis-card',
        'badge': 'badge-healthy',
        'badge_text': 'CLEAR',
        'precautions': [
            "Maintain a balanced diet rich in Omega-3 fatty acids.",
            "Engage in regular cardiovascular exercise (≥30 min/day).",
            "Stimulate cognition via puzzles, reading, and skill acquisition.",
            "Schedule annual neurological checkups as a preventive measure."
        ]
    },
    'VeryMildDemented': {
        'status': 'Very Mild Cognitive Impairment',
        'icon': '⚠️',
        'card_class': 'warning-card',
        'badge': 'badge-caution',
        'badge_text': 'MONITOR',
        'precautions': [
            "Schedule a formal cognitive assessment with a neurologist.",
            "Begin daily memory-tracking journaling.",
            "Optimize home safety — adequate lighting, remove trip hazards.",
            "Discuss early-intervention medications with your physician."
        ]
    },
    'MildDemented': {
        'status': 'Mild Dementia Detected',
        'icon': '🚨',
        'card_class': 'warning-card',
        'badge': 'badge-warning',
        'badge_text': 'ALERT',
        'precautions': [
            "Strict adherence to neurologist-prescribed medications.",
            "Establish rigid daily routines to minimise confusion.",
            "Involve family members in financial and medical planning.",
            "Consider GPS-enabled wearables if the patient travels alone."
        ]
    },
    'ModerateDemented': {
        'status': 'Moderate Dementia Detected',
        'icon': '🏥',
        'card_class': 'warning-card',
        'badge': 'badge-critical',
        'badge_text': 'CRITICAL',
        'precautions': [
            "24/7 supervision or dedicated care facility recommended.",
            "Implement fall-prevention and anti-wandering protocols.",
            "Use simple, calm communication — avoid correcting memory lapses.",
            "Seek caregiver support groups to prevent family burnout."
        ]
    }
}

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem;">🧠</div>
        <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem; color:#00d4ff; letter-spacing:2px; font-weight:700;">NeuroAI</div>
        <div style="font-size:0.75rem; color:#475569; margin-top:4px; letter-spacing:1px;">DIAGNOSTIC PLATFORM</div>
    </div>
    <hr style="border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.3),transparent); margin:0.5rem 0 1.5rem;">
    """, unsafe_allow_html=True)

    st.markdown("**MODEL ARCHITECTURE**")
    st.markdown("""
    <div class="glass-card" style="font-size:0.82rem; line-height:1.8;">
        <div style="color:#00d4ff; margin-bottom:0.4rem;">🔷 EfficientNet-B0</div>
        <div style="color:#64748b; padding-left:0.8rem;">Local texture extraction → 512-d proj</div>
        <div style="color:#7c3aed; margin: 0.4rem 0;">🔷 Swin Transformer-T</div>
        <div style="color:#64748b; padding-left:0.8rem;">Global structure analysis → 512-d proj</div>
        <div style="color:#f59e0b; margin-top:0.6rem;">🔀 Cross-Attention Gate</div>
        <div style="color:#64748b; padding-left:0.8rem;">Dynamic feature gating (SiLU + Sigmoid)</div>
        <div style="color:#10b981; margin-top:0.4rem;">🧠 Deep Classifier MLP</div>
        <div style="color:#64748b; padding-left:0.8rem;">BN → GELU → Dropout → 4 classes</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**CLASSIFICATION TARGETS**")
    targets = [
        ("✅", "Non-Demented",      "#10b981"),
        ("⚠️", "Very Mild Dementia", "#f59e0b"),
        ("🚨", "Mild Dementia",      "#ef4444"),
        ("🏥", "Moderate Dementia",  "#a855f7"),
    ]
    for icon, label, color in targets:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
            <span style="font-size:0.9rem;">{icon}</span>
            <span style="font-size:0.82rem; color:{color};">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    device_str = "🚀 CUDA GPU" if torch.cuda.is_available() else "💻 CPU"
    st.markdown(f"""
    <div style="text-align:center; padding:0.6rem; border:1px solid rgba(0,212,255,0.15); border-radius:8px; font-size:0.78rem; color:#475569;">
        Compute: <span style="color:#00d4ff; font-weight:700;">{device_str}</span>
    </div>
    """, unsafe_allow_html=True)


# 7. HERO HEADER
st.markdown('<h1 class="hero-title">🧠 NeuroAI Diagnostic Platform</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload a T1-weighted MRI scan — our Hybrid Swin-EfficientNet model delivers clinical-grade Alzheimer\'s staging with Explainable AI heatmaps.</p>',
    unsafe_allow_html=True
)

# Metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", "Hybrid CNN+Swin")
m2.metric("Classes", "4 Stages")
m3.metric("XAI", "GradCAM")
m4.metric("Input", "256 × 256 px")

st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 8. MAIN LAYOUT
# ─────────────────────────────────────────────
col_upload, col_results = st.columns([1, 2], gap="large")

# ── LEFT: Upload panel ──────────────────────
with col_upload:
    st.markdown('<div class="section-label">📤 Input Scan</div>', unsafe_allow_html=True)

    # Use a styled container instead of split open/close div tags
    with st.container():
        st.markdown("""
        <style>
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(0,212,255,0.12);
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your MRI image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        analyze_button = False
        mri_valid      = False   # gate that controls the results panel

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')

            # ── Validate before showing anything ──
            mri_valid, reason = is_valid_brain_mri(img)

            if not mri_valid:
                # Show the uploaded image (small) so user can see what they picked
                st.image(img, caption="Uploaded image", use_container_width=True)
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg,rgba(239,68,68,0.10) 0%,rgba(245,158,11,0.06) 100%);
                    border: 1.5px solid rgba(239,68,68,0.45);
                    border-radius: 14px;
                    padding: 1.1rem 1.3rem;
                    margin-top: 0.8rem;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position:absolute; top:0; left:0; right:0; height:3px;
                        background:linear-gradient(90deg,#ef4444,#f59e0b,#ef4444);
                        background-size:200% 100%;
                        animation:shimmer 2s linear infinite;
                    "></div>
                    <div style="font-size:1.05rem; font-weight:700; color:#ef4444; margin-bottom:0.4rem;">
                        ⛔ Invalid Image
                    </div>
                    <div style="font-size:0.85rem; color:#fca5a5; line-height:1.6;">
                        {reason}
                    </div>
                    <div style="margin-top:0.8rem; font-size:0.78rem; color:#64748b; border-top:1px solid rgba(239,68,68,0.2); padding-top:0.6rem;">
                        <strong style="color:#94a3b8;">✅ What to upload:</strong><br>
                        A <em>T1-weighted axial brain MRI scan</em> — greyscale, square-ish,
                        with a dark background around the brain tissue.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.image(img, caption="📷 Patient MRI Scan", use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_button = st.button("⚡ RUN CLINICAL ANALYSIS")

        else:
            mri_valid = False
            st.markdown("""
            <div style="text-align:center; padding:2rem 1rem; color:#334155;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">🖼️</div>
                <div style="font-size:0.85rem; line-height:1.6;">
                    Supported formats:<br>
                    <span style="color:#00d4ff;">JPG · JPEG · PNG</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Info box
    st.markdown("""
    <div class="glass-card" style="font-size:0.82rem; line-height:1.8; color:#64748b;">
        <div style="color:#00d4ff; font-weight:700; margin-bottom:0.5rem;">ℹ️ How It Works</div>
        <div>① Upload a T1-weighted axial MRI slice</div>
        <div>② AI extracts CNN textures + Swin global features</div>
        <div>③ Hybrid fusion layer predicts dementia stage</div>
        <div>④ GradCAM generates XAI attention heatmaps</div>
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT: Results panel ─────────────────────
with col_results:
    if uploaded_file is not None and not mri_valid:
        # Invalid image was uploaded — show guidance panel
        st.markdown("""
        <div style="
            height: 420px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1.5px dashed rgba(239,68,68,0.3);
            border-radius: 16px;
            gap: 1rem;
            padding: 2rem;
        ">
            <div style="font-size:3.5rem;">🧠</div>
            <div style="font-family:'Orbitron',sans-serif; font-size:0.85rem; color:#ef4444; letter-spacing:2px;">INVALID INPUT</div>
            <div style="font-size:0.88rem; color:#64748b; text-align:center; line-height:1.8; max-width:320px;">
                Please upload a valid <strong style="color:#94a3b8;">T1-weighted axial brain MRI</strong> image.
                The system only accepts greyscale MRI scans with a dark background.
            </div>
            <div style="
                margin-top:0.5rem;
                padding: 0.6rem 1.4rem;
                border: 1px solid rgba(239,68,68,0.3);
                border-radius: 8px;
                font-size:0.78rem;
                color:#ef4444;
            ">⬅️ Replace the image to continue</div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file is None:
        # Placeholder state
        st.markdown("""
        <div style="
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1.5px dashed rgba(0,212,255,0.15);
            border-radius: 16px;
            color: #1e293b;
            gap: 1rem;
        ">
            <div style="font-size:3rem; opacity:0.4;">📊</div>
            <div style="font-size:0.9rem; color:#334155; text-align:center; opacity:0.7;">
                Analysis results will appear here<br>after you upload and run the scan
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif not analyze_button:
        st.markdown("""
        <div style="
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1.5px dashed rgba(0,212,255,0.15);
            border-radius: 16px;
            gap: 1rem;
        ">
            <div style="font-size:3rem;">🔬</div>
            <div style="font-size:0.9rem; color:#475569; text-align:center;">
                MRI scan loaded. Click <strong style="color:#00d4ff;">RUN CLINICAL ANALYSIS</strong> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── INFERENCE ───────────────────────
        with st.spinner("🔄 Neural networks processing MRI scan..."):
            img_tensor = val_transforms(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output[0], dim=0).cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_class = classes[pred_idx]
                confidence = float(probs[pred_idx]) * 100

        # ── RESULTS HEADER ───────────────────
        st.markdown('<div class="section-label">📋 Diagnostic Report</div>', unsafe_allow_html=True)

        advice = clinical_advice[pred_class]
        badge_html = f'<span class="badge {advice["badge"]}">{advice["badge_text"]}</span>'

        card_items = "".join([f"<li>{item}</li>" for item in advice['precautions']])
        card_html = f"""
        <div class="{advice['card_class']}">
            {badge_html}
            <h3>{advice['icon']} {advice['status']}</h3>
            <div style="color:#64748b; font-size:0.85rem; margin-bottom:0.8rem;">
                Confidence: <strong style="color:#00d4ff;">{confidence:.2f}%</strong>
                &nbsp;|&nbsp; Class: <strong style="color:#f1f5f9;">{pred_class}</strong>
            </div>
            <div style="font-size:0.82rem; color:#00d4ff; font-weight:700; margin-bottom:0.4rem;">
                RECOMMENDED CLINICAL ACTIONS
            </div>
            <ul>{card_items}</ul>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # ── PROBABILITY BARS ─────────────────
        st.markdown('<div class="section-label" style="margin-top:1rem;">📊 Class Probabilities</div>', unsafe_allow_html=True)
        labels_friendly = {
            'MildDemented':     'Mild Dementia',
            'ModerateDemented': 'Moderate Dementia',
            'NonDemented':      'Non-Demented',
            'VeryMildDemented': 'Very Mild Dementia',
        }

        # Render each bar as its own self-contained st.markdown call
        # to avoid raw HTML leaking from large dynamic HTML strings
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(0,212,255,0.12);
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
        ">
        """, unsafe_allow_html=True)

        for i, cls in enumerate(classes):
            pct = float(probs[i]) * 100
            is_pred = (i == pred_idx)
            label_color  = "#00d4ff" if is_pred else "#94a3b8"
            label_weight = "700"     if is_pred else "400"
            pct_color    = "#00d4ff" if is_pred else "#475569"
            bar_bg       = "linear-gradient(90deg,#00d4ff,#7c3aed)" if is_pred else "rgba(100,116,139,0.35)"

            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.82rem;">
                    <span style="color:{label_color}; font-weight:{label_weight};">{labels_friendly[cls]}</span>
                    <span style="color:{pct_color}; font-weight:700;">{pct:.1f}%</span>
                </div>
                <div style="height:8px; border-radius:4px; background:rgba(255,255,255,0.06); overflow:hidden;">
                    <div style="height:100%; width:{pct:.1f}%; border-radius:4px; background:{bar_bg};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── XAI HEATMAPS ─────────────────────
        st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">👁️ Explainable AI — GradCAM Attention Maps</div>', unsafe_allow_html=True)

        xai_info = """
        <div class="glass-card" style="display:flex; gap:2rem; font-size:0.82rem;">
            <div>
                <span style="color:#00bfff; font-weight:700;">■ CNN (Jet)</span>
                <div style="color:#64748b; margin-top:2px;">Local micro-textures:<br>ventricle size, cortical thinning</div>
            </div>
            <div>
                <span style="color:#ff4500; font-weight:700;">■ Swin (Hot)</span>
                <div style="color:#64748b; margin-top:2px;">Global morphology:<br>brain symmetry & volume</div>
            </div>
        </div>
        """
        st.markdown(xai_info, unsafe_allow_html=True)

        with st.spinner("🎨 Generating GradCAM explanations..."):
            rgb_img = np.array(img.resize((256, 256))) / 255.0

            cam_cnn   = GradCAM(model=model, target_layers=[model.cnn_backbone[-1]])
            cam_swin  = GradCAM(model=model, target_layers=[model.swin_backbone[-1]])
            gray_cnn  = cam_cnn(input_tensor=img_tensor)[0, :]
            gray_swin = cam_swin(input_tensor=img_tensor)[0, :]

            heat_cnn  = show_cam_on_image(rgb_img, gray_cnn,  use_rgb=True, colormap=cv2.COLORMAP_JET)
            heat_swin = show_cam_on_image(rgb_img, gray_swin, use_rgb=True, colormap=cv2.COLORMAP_HOT)

            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            fig.patch.set_facecolor('#020818')

            for ax, heat, title, color in [
                (axes[0], heat_cnn,  "Local Texture Focus (CNN Backbone)",   "#00d4ff"),
                (axes[1], heat_swin, "Global Symmetry Focus (Swin Backbone)", "#ff4500"),
            ]:
                ax.imshow(heat)
                ax.set_facecolor('#020818')
                ax.set_title(title, fontsize=11, color=color, fontweight='bold', pad=10)
                ax.axis('off')
                for spine in ax.spines.values():
                    spine.set_edgecolor((0, 212/255, 255/255, 0.2))

            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

        # ── FOOTER NOTE ──────────────────────
        st.markdown("""
        <div style="text-align:center; margin-top:1.5rem; padding:0.8rem; border:1px solid rgba(239,68,68,0.15); border-radius:10px; font-size:0.78rem; color:#64748b;">
            ⚠️ <strong style="color:#ef4444;">Clinical Disclaimer:</strong>
            This AI output is intended for research assistance only and does <em>not</em> replace professional medical diagnosis.
            Always consult a qualified neurologist for clinical decisions.
        </div>
        """, unsafe_allow_html=True)