# main.py
import streamlit as st
import os
import tempfile
import time
from PIL import Image
from collections import defaultdict
import shutil

# Import from other project files
from utils import load_yolo_models
from video_processor import SmartEyeProcessor
from report_generator import generate_pdf_report

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Eye - Hazard and PPE Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main UI ---
st.title("👁️ Smart Eye: Advanced Hazard & PPE Detection System")

# --- Sidebar ---
st.sidebar.header("📝 User Information")
user_id = st.sidebar.text_input("Enter User ID:", placeholder="e.g., SAFETY_001")
if not user_id:
    user_id = f"user_{int(time.time())}"
    st.sidebar.info(f"Auto-generated User ID: {user_id}")

st.sidebar.header("🎯 Detection Features")
st.sidebar.header("Model Classes")
st.sidebar.markdown("**Hazard Detection Model:**")
st.sidebar.code("- Heavy-vehicles\n- fire\n- forklift\n- ladder\n- person")
st.sidebar.markdown("**PPE Detection Model:**")
st.sidebar.code("- boots, no_boots\n- gloves, no_gloves\n- helmet, no_helmet\n- vest, no_vest")
st.sidebar.header("🎯 Reports Features")
st.sidebar.markdown("""
- Executive summary
- Detailed incident analysis
- Individual incident screenshots
- Severity-based recommendations
- Professional PDF formatting
""")

# --- Model Loading ---
hazard_model, ppe_model = load_yolo_models()

if hazard_model and ppe_model:
    
    # <-- 1. DEFAULT VIDEO OPTION
    st.header("Select Video Source")
    input_option = st.radio(
        "Choose an option:",
        ('Upload a video', 'Use a default video'),
        label_visibility="collapsed"
    )

    input_video_path = None
    video_name = None
    
    if input_option == 'Upload a video':
        uploaded_file = st.file_uploader(
            "Choose a video file for safety analysis...", 
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload industrial safety videos (construction sites, manufacturing floors, etc.)"
        )
        if uploaded_file:
            video_name = uploaded_file.name
            temp_dir = tempfile.mkdtemp()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir)
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name
    else: # Default video option
        default_video_path = 'videos/default_safety_video1.mp4'
        if os.path.exists(default_video_path):
            st.info(f"Using default video: `{default_video_path}`")
            input_video_path = default_video_path
            video_name = os.path.basename(default_video_path)
            temp_dir = tempfile.mkdtemp() # Create temp dir for outputs
        else:
            st.warning(f"Default video not found at `{default_video_path}`. Please create the `videos` directory and place a video file there.")
            
    # --- Video Processing ---
    if input_video_path:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📹 Video: {video_name}")
            st.info(f"👤 User ID: {user_id}")
        with col2:
            st.video(input_video_path)

        if st.button("🚀 Start Safety Analysis", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing video for safety hazards... This may take several minutes."):
                try:
                    processor = SmartEyeProcessor(hazard_model, ppe_model)
                    output_video_path = processor.process_video(input_video_path, user_id, temp_dir)
                    
                    report_path = generate_pdf_report(
                        output_dir=temp_dir,
                        user_id=processor.user_id,
                        video_name=processor.video_name,
                        processing_start_time=processor.processing_start_time,
                        hazards_detected=processor.hazards_detected,
                        frame_snapshots=processor.frame_snapshots,
                        hazard_zones=processor.hazard_zones
                    )
                    
                    st.session_state.processing_data['report_path'] = report_path
                    st.success("✅ Safety analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

    # Display results if processing data exists
    if 'processing_data' in st.session_state and st.session_state.processing_data:
        data = st.session_state.processing_data
        
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Display results summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Hazards", len(data['hazards_detected']))
        
        with col2:
            critical_count = sum(1 for h in data['hazards_detected'] if h['severity'] == 'CRITICAL')
            st.metric("Critical Hazards", critical_count, delta="🚨" if critical_count > 0 else "✅")
        
        with col3:
            high_count = sum(1 for h in data['hazards_detected'] if h['severity'] == 'HIGH')
            st.metric("High Risk", high_count, delta="⚠️" if high_count > 0 else "✅")
        
        # Hazard breakdown
        if data['hazards_detected']:
            st.subheader("🚨 Safety Alert Summary")
            
            hazard_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            
            for hazard in data['hazards_detected']:
                hazard_counts[hazard['type']] += 1
                severity_counts[hazard['severity']] += 1
            
            # Display by severity
            if severity_counts.get('CRITICAL', 0) > 0:
                st.error(f"**CRITICAL HAZARDS DETECTED ({severity_counts['CRITICAL']} incidents)**")
                for hazard_type, count in hazard_counts.items():
                    if any(h['type'] == hazard_type and h['severity'] == 'CRITICAL' for h in data['hazards_detected']):
                        st.error(f"- {hazard_type.replace('_', ' ').title()}: {count} incidents")

            if severity_counts.get('HIGH', 0) > 0:
                st.warning(f"**HIGH RISK SITUATIONS ({severity_counts['HIGH']} incidents)**")
                for hazard_type, count in hazard_counts.items():
                     if any(h['type'] == hazard_type and h['severity'] == 'HIGH' for h in data['hazards_detected']):
                        st.warning(f"- {hazard_type.replace('_', ' ').title()}: {count} incidents")
            
            if severity_counts.get('INFO', 0) > 0:
                st.info(f"**SAFETY OBSERVATIONS ({severity_counts['INFO']} incidents)**")
                for hazard_type, count in hazard_counts.items():
                    if any(h['type'] == hazard_type and h['severity'] == 'INFO' for h in data['hazards_detected']):
                        st.info(f"- {hazard_type.replace('_', ' ').title()}: {count} incidents")
            
            # Display hazard screenshots
            if data['frame_snapshots']:
                st.subheader("📸 Hazard Evidence Gallery")
                
                snapshots_by_type = defaultdict(list)
                for snapshot in data['frame_snapshots'][:12]:
                    snapshots_by_type[snapshot['hazard']['type']].append(snapshot)
                
                for hazard_type, snapshots in snapshots_by_type.items():
                    with st.expander(f"{hazard_type.replace('_', ' ').title()} - {len(snapshots)} incidents"):
                        cols = st.columns(min(3, len(snapshots)))
                        for i, snapshot in enumerate(snapshots[:6]):
                            with cols[i % 3]:
                                if os.path.exists(snapshot['path']):
                                    img = Image.open(snapshot['path'])
                                    st.image(img, caption=f"Frame {snapshot['frame_number']} ({snapshot['timestamp']:.1f}s)", use_container_width=True)
        
        else:
            st.success("✅ **NO SAFETY INCIDENTS DETECTED**")
            st.info("Standard operations observed. Continue current safety protocols.") 
        
        st.subheader("📥 Download Results")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            st.markdown("**📹 Processed Video**")
            if 'output_video_path' in data and os.path.exists(data['output_video_path']):
                with open(data['output_video_path'], "rb") as video_file:
                    st.download_button(label="Download Processed Video", data=video_file.read(), file_name=f"safety_analysis_{data['user_id']}.mp4", mime="video/mp4", use_container_width=True)
            else:
                st.warning("Processed video not available")
        
        with download_col2:
            st.markdown("**📊 Safety Report**")
            if 'report_path' in data and os.path.exists(data['report_path']):
                with open(data['report_path'], "rb") as pdf_file:
                    st.download_button(label="Download Safety Report (PDF)", data=pdf_file.read(), file_name=f"safety_report_{data['user_id']}.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.warning("Safety report not available")
             
        # Clear results button
        if st.button("🗑️ Clear Results", type="secondary"):
            if 'processing_data' in st.session_state:
                try:
                    if 'output_video_path' in st.session_state.processing_data:
                        output_path = st.session_state.processing_data['output_video_path']
                        if os.path.exists(output_path):
                            import shutil
                            shutil.rmtree(os.path.dirname(output_path), ignore_errors=True)
                except:
                    pass
                del st.session_state.processing_data
            st.rerun()


else:
    st.error("❌ **Models could not be loaded**")
    st.info("Please ensure the model files are present in the 'models' directory.")
    st.code("models/combine-row-17.pt\nmodels/ppe-v3-manual-aug-200.pt")

# --- Footer ---
st.markdown("---")
st.info("💡 **Tip**: For best results, use high-quality videos with clear visibility of personnel and equipment.")