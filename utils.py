# utils.py
import streamlit as st
import os
import cv2
from ultralytics import YOLO

@st.cache_resource
def load_yolo_models():
    """Load YOLO models with caching"""
    try:
        hazard_model_path = os.path.join('models', 'combine-row-17.pt')
        ppe_model_path = os.path.join('models', 'ppe-v3-manual-aug-200.pt')

        if not os.path.exists(hazard_model_path) or not os.path.exists(ppe_model_path):
            st.error("Model files not found! Please make sure the model files are in the 'models' directory.")
            return None, None
        
        hazard_model = YOLO(hazard_model_path)
        ppe_model = YOLO(ppe_model_path)
        return hazard_model, ppe_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None      
    

def histogram_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return round(similarity * 100, 2)  # CORREL returns in [0, 1]