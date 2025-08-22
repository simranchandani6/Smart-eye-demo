import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
import time
from PIL import Image
from collections import defaultdict
from datetime import datetime
import streamlit as st

class SmartEyeProcessor:
    def __init__(self, hazard_model, ppe_model):
        self.hazard_model = hazard_model
        self.ppe_model = ppe_model
        
        # Configuration
        self.CONF_THRESHOLD = 0.5
        self.LADDER_CONF_THRESHOLD = 0.65
        self.NMS_THRESHOLD = 0.4
        self.SIMILARITY_THRESHOLD = 60.0
        
        # Color maps
        self.HAZARD_COLOR_MAP = {
            'Heavy-vehicles': (255, 0, 0), 'fire': (0, 0, 255), 'forklift': (255, 165, 0),
            'ladder': (255, 255, 0), 'person': (0, 255, 255)
        }
        self.PPE_COLOR_MAP = {
            'boots': (0, 128, 0), 'gloves': (128, 0, 128), 'helmet': (0, 255, 0),
            'vest': (255, 192, 203), 'no_boots': (100, 100, 100), 'no_gloves': (100, 100, 100),
            'no_helmet': (100, 100, 100), 'no_vest': (100, 100, 100)
        }
        self.DEFAULT_COLOR = (200, 200, 200)
        
        # Hazard tracking, cooldowns, and report data
        self.fire_continuous_frames = 0
        self.continuous_fire_cooldown = 0
        self.fire_person_cooldown = 0
        self.ladder_person_cooldown = 0
        self.ppe_violation_cooldown = 0
        self.heavy_vehicle_fire_cooldown = 0
        self.fire_forklift_cooldown = 0
        self.person_forklift_cooldown = 0
        self.ppe_violation_tracking = defaultdict(int)
        
        self.hazards_detected = []
        self.frame_snapshots = []
        self.hazard_zones = defaultdict(list)
        self.user_id = None
        self.video_name = None
        self.processing_start_time = None
        
        self.ppe_violation_window = 2
        self.fire_heavy_window = 80
        self.fire_heavy_min = 10
        self.ladder_person_threshold = 75
        
        self.fire_heavy_history = []


    def histogram_similarity(self, img1_path, img2_path):
        # Read images in grayscale for faster processing
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        # Resize img2 to match img1 shape
        if img2.shape != img1.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

        # Calculate normalized histograms and flatten for faster comparison
        hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return round(similarity * 100, 2)

    def calculate_overlap_percentage(self, box1: tuple, box2: tuple) -> float:
        """Calculate intersection over union (IoU) percentage"""
        if len(box1) != 4 or len(box2) != 4:
            raise ValueError("Both boxes must be tuples of length 4 (x1, y1, x2, y2).")
        
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x1_inter < x2_inter and y1_inter < y2_inter:
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            return (intersection / union) * 100 if union > 0 else 0
        return 0
    
    def calculate_distance(self, box1, box2):
        """Calculate distance between centers of two bounding boxes (optimized)"""
        box1 = np.asarray(box1)
        box2 = np.asarray(box2)
        center1 = (box1[:2] + box1[2:]) / 2
        center2 = (box2[:2] + box2[2:]) / 2
        return float(np.linalg.norm(center1 - center2))
    
    def is_ppe_inside_person(self, ppe_box, person_box, threshold=0.5):
        """Check if PPE detection is inside person bounding box (optimized)"""
        ppe_box = np.asarray(ppe_box)
        person_box = np.asarray(person_box)
        ppe_center = (ppe_box[:2] + ppe_box[2:]) / 2
        return (person_box[0] <= ppe_center[0] <= person_box[2] and 
                person_box[1] <= ppe_center[1] <= person_box[3])
    

    
    def process_video(self, video_path, user_id, output_dir):
        """Process video and detect hazards"""
        if 'processing_data' not in st.session_state:
            st.session_state.processing_data = {}

        self.user_id = user_id
        self.video_name = os.path.basename(video_path)
        self.processing_start_time = datetime.now()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = os.path.join(output_dir, f"processed_{self.video_name}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        person_ppe_tracking = defaultdict(list)
        fire_heavy_buffer = []

        # Precompute model names for faster lookup
        hazard_names = self.hazard_model.names
        ppe_names = self.ppe_model.names

        SKIP_FRAMES = 4
        last_hazard_results = None
        last_ppe_results = None

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps

                # Do inference only on every (SKIP_FRAMES+1)th frame, reuse previous results otherwise
                if (frame_count - 1) % (SKIP_FRAMES + 1) == 0:
                    hazard_results = self.hazard_model(frame, verbose=False)
                    ppe_results = self.ppe_model(frame, verbose=False)
                    last_hazard_results = hazard_results
                    last_ppe_results = ppe_results
                else:
                    hazard_results = last_hazard_results
                    ppe_results = last_ppe_results
                
                all_boxes, all_confidences, all_class_names, all_colors = [], [], [], []

                def fast_append_detections(results, model_names, color_map):
                    if len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        cls_ids = results[0].boxes.cls.cpu().int().numpy()
                        for i in range(len(boxes)):
                            class_name = model_names[cls_ids[i]]
                            conf = confs[i]
                            threshold = self.LADDER_CONF_THRESHOLD if class_name == 'ladder' else self.CONF_THRESHOLD
                            if conf > threshold:
                                all_boxes.append(boxes[i])
                                all_confidences.append(conf)
                                all_class_names.append(class_name)
                                all_colors.append(color_map.get(class_name, self.DEFAULT_COLOR))

                fast_append_detections(hazard_results, hazard_names, self.HAZARD_COLOR_MAP)
                fast_append_detections(ppe_results, ppe_names, self.PPE_COLOR_MAP)
                
                final_indices = []
                if all_boxes:
                    xywh_boxes = np.array([[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in all_boxes])
                    all_confidences_np = np.array(all_confidences)
                    all_class_names_np = np.array(all_class_names)
                    unique_class_names = set(all_class_names)
                    for class_name in unique_class_names:
                        class_indices = np.where(all_class_names_np == class_name)[0]
                        class_boxes = xywh_boxes[class_indices].tolist()
                        class_confidences = all_confidences_np[class_indices].tolist()
                        if class_boxes:
                            indices_after_nms = cv2.dnn.NMSBoxes(class_boxes, class_confidences, self.CONF_THRESHOLD, self.NMS_THRESHOLD)
                            if len(indices_after_nms) > 0:
                                final_indices.extend([class_indices[i] for i in indices_after_nms.flatten()])
                
                hazards_in_frame = []
                # Grouped cooldown logic for efficiency
                for attr in [
                    "continuous_fire_cooldown",
                    "ladder_person_cooldown",
                    "fire_person_cooldown",
                    "ppe_violation_cooldown",
                    "heavy_vehicle_fire_cooldown",
                    "fire_forklift_cooldown",
                    "person_forklift_cooldown"
                ]:
                    if getattr(self, attr) > 0:
                        setattr(self, attr, getattr(self, attr) - 1)
                
                fire_detections = [(i, all_boxes[i]) for i in final_indices if all_class_names[i] == 'fire']
                person_detections = [(i, all_boxes[i]) for i in final_indices if all_class_names[i] == 'person']
                ladder_detections = [(i, all_boxes[i]) for i in final_indices if all_class_names[i] == 'ladder']
                heavy_vehicle_detections = [(i, all_boxes[i]) for i in final_indices if all_class_names[i] == 'Heavy-vehicles']
                forklift_detections = [(i, all_boxes[i]) for i in final_indices if all_class_names[i] == 'forklift']
                ppe_detections = [(i, all_boxes[i], all_class_names[i]) for i in final_indices 
                                if all_class_names[i] in ['no_boots', 'no_gloves', 'no_helmet', 'no_vest']]
                
                if self.continuous_fire_cooldown == 0:
                    if fire_detections:
                        self.fire_continuous_frames += 1
                        if self.fire_continuous_frames >= (3 * fps):
                            hazards_in_frame.append({
                                'type': 'FIRE_CONTINUOUS',
                                'severity': 'CRITICAL',
                                'description': 'Fire detected continuously',
                                'boxes': [box for _, box in fire_detections],
                                'timestamp': current_time
                            })
                            self.fire_continuous_frames = 0
                            self.continuous_fire_cooldown = 60 * fps
                    else:
                        self.fire_continuous_frames = 0
                
                # Optimize PPE violation tracking using NumPy
                person_indices_np = np.array([idx for idx, _ in person_detections])
                person_boxes_np = np.array([box for _, box in person_detections])
                ppe_indices_np = np.array([idx for idx, _, _ in ppe_detections])
                ppe_boxes_np = np.array([box for _, box, _ in ppe_detections])
                ppe_classes_np = np.array([cls for _, _, cls in ppe_detections])

                for pi, person_box in zip(person_indices_np, person_boxes_np):
                    person_key = f"person_{pi}"
                    # Vectorized PPE-inside-person detection
                    if ppe_boxes_np.shape[0] > 0:
                        ppe_centers = (ppe_boxes_np[:, :2] + ppe_boxes_np[:, 2:]) / 2
                        inside_mask = (
                            (person_box[0] <= ppe_centers[:, 0]) & (ppe_centers[:, 0] <= person_box[2]) &
                            (person_box[1] <= ppe_centers[:, 1]) & (ppe_centers[:, 1] <= person_box[3])
                        )
                        no_mask = np.char.startswith(ppe_classes_np.astype(str), "no_")
                        ppe_inside_mask = inside_mask & no_mask
                        no_classes = ppe_classes_np[ppe_inside_mask].tolist()
                    else:
                        no_classes = []
                    person_ppe_tracking[person_key].append((frame_count, len(no_classes), list(no_classes), person_box))
                    # Keep only recent window
                    person_ppe_tracking[person_key] = [x for x in person_ppe_tracking[person_key] if frame_count - x[0] < self.ppe_violation_window * fps]
                    if len(person_ppe_tracking[person_key]) >= self.ppe_violation_window * fps:
                        # Use NumPy argmax for max_no selection
                        counts = np.array([x[1] for x in person_ppe_tracking[person_key]])
                        idx_max = np.argmax(counts)
                        max_no = person_ppe_tracking[person_key][idx_max]
                        if max_no[1] > 0 and self.ppe_violation_cooldown == 0:
                            hazards_in_frame.append({
                                'type': 'PPE_VIOLATION',
                                'severity': 'HIGH',
                                'description': f'PPE violation(s): {", ".join(max_no[2])}',
                                'boxes': [max_no[3]],
                                'timestamp': current_time,
                                'ppe_type': ",".join(max_no[2])
                            })
                            self.ppe_violation_cooldown = 60
                        person_ppe_tracking[person_key] = []

                if self.fire_person_cooldown == 0:
                    # Vectorized fire-person proximity
                    if fire_detections and person_detections:
                        fire_boxes_np = np.array([box for _, box in fire_detections])
                        person_boxes_np2 = np.array([box for _, box in person_detections])
                        for f_idx, fire_box in enumerate(fire_boxes_np):
                            overlaps = np.array([self.calculate_overlap_percentage(fire_box, person_box) for person_box in person_boxes_np2])
                            mask = overlaps >= 10.0
                            if np.any(mask):
                                person_box = person_boxes_np2[np.argmax(mask)]
                                hazards_in_frame.append({
                                    'type': 'FIRE_PERSON_PROXIMITY',
                                    'severity': 'CRITICAL',
                                    'description': f'Person detected near fire ',
                                    'boxes': [fire_box, person_box],
                                    'timestamp': current_time
                                })
                                self.fire_person_cooldown = 60
                                break
                
                if self.fire_forklift_cooldown == 0:
                    # Vectorized fire-forklift proximity
                    if fire_detections and forklift_detections:
                        fire_boxes_np = np.array([box for _, box in fire_detections])
                        forklift_boxes_np = np.array([box for _, box in forklift_detections])
                        for fire_box in fire_boxes_np:
                            overlaps = np.array([self.calculate_overlap_percentage(fire_box, forklift_box) for forklift_box in forklift_boxes_np])
                            distances = np.array([self.calculate_distance(fire_box, forklift_box) for forklift_box in forklift_boxes_np])
                            mask = (overlaps >= 5.0) | (distances < 100)
                            if np.any(mask):
                                forklift_box = forklift_boxes_np[np.argmax(mask)]
                                hazards_in_frame.append({
                                    'type': 'FIRE_FORKLIFT_PROXIMITY',
                                    'severity': 'MEDIUM',
                                    'description': f'Forklift near fire',
                                    'boxes': [fire_box, forklift_box],
                                    'timestamp': current_time
                                })
                                self.fire_forklift_cooldown = 60
                                break

                fire_heavy_found_this_frame = False
                interacting_boxes_this_frame = []
                fire_heavy_found_this_frame = False
                if heavy_vehicle_detections and fire_detections:
                    vehicle_boxes_np = np.array([box for _, box in heavy_vehicle_detections])
                    fire_boxes_np = np.array([box for _, box in fire_detections])
                    for vehicle_box in vehicle_boxes_np:
                        overlaps = np.array([self.calculate_overlap_percentage(vehicle_box, fire_box) for fire_box in fire_boxes_np])
                        distances = np.array([self.calculate_distance(vehicle_box, fire_box) for fire_box in fire_boxes_np])
                        mask = (overlaps >= 5.0) | (distances < 100)
                        if np.any(mask):
                            fire_box = fire_boxes_np[np.argmax(mask)]
                            fire_heavy_found_this_frame = True
                            interacting_boxes_this_frame = [vehicle_box, fire_box]
                            break
                
                fire_heavy_buffer.append((fire_heavy_found_this_frame, interacting_boxes_this_frame))
                if len(fire_heavy_buffer) > self.fire_heavy_window:
                    fire_heavy_buffer.pop(0)

                positive_detections = sum(1 for found, _ in fire_heavy_buffer if found)
                
                if positive_detections >= self.fire_heavy_min and self.heavy_vehicle_fire_cooldown == 0:
                    most_recent_boxes = []
                    for found, boxes in reversed(fire_heavy_buffer):
                        if found:
                            most_recent_boxes = boxes
                            break
                    
                    if most_recent_boxes:
                        hazards_in_frame.append({
                            'type': 'HEAVY_VEHICLE_FIRE',
                            'severity': 'MEDIUM',
                            'description': f'Heavy vehicle near fire detected',
                            'boxes': most_recent_boxes,
                            'timestamp': current_time
                        })
                        self.heavy_vehicle_fire_cooldown = self.fire_heavy_window
                        fire_heavy_buffer = []


                if self.ladder_person_cooldown == 0:
                    # Vectorized ladder-person proximity
                    if ladder_detections and person_detections:
                        ladder_boxes_np = np.array([box for _, box in ladder_detections])
                        person_boxes_np3 = np.array([box for _, box in person_detections])
                        for ladder_box in ladder_boxes_np:
                            overlaps = np.array([self.calculate_overlap_percentage(ladder_box, person_box) for person_box in person_boxes_np3])
                            mask = overlaps >= self.ladder_person_threshold
                            if np.any(mask):
                                person_box = person_boxes_np3[np.argmax(mask)]
                                hazards_in_frame.append({
                                    'type': 'LADDER_CLIMBING',
                                    'severity': 'INFO',
                                    'description': f'Person on/near ladder',
                                    'boxes': [ladder_box, person_box],
                                    'timestamp': current_time
                                })
                                self.ladder_person_cooldown = 60
                                break

                if self.person_forklift_cooldown == 0:
                    # Vectorized person-in-forklift detection
                    if person_detections and forklift_detections:
                        person_boxes_np4 = np.array([box for _, box in person_detections])
                        forklift_boxes_np2 = np.array([box for _, box in forklift_detections])
                        for person_box in person_boxes_np4:
                            mask = np.all([
                                (person_box[0] >= forklift_boxes_np2[:, 0]),
                                (person_box[1] >= forklift_boxes_np2[:, 1]),
                                (person_box[2] <= forklift_boxes_np2[:, 2]),
                                (person_box[3] <= forklift_boxes_np2[:, 3])
                            ], axis=0)
                            if np.any(mask):
                                forklift_box = forklift_boxes_np2[np.argmax(mask)]
                                hazards_in_frame.append({
                                    'type': 'PERSON_IN_FORKLIFT',
                                    'severity': 'INFO',
                                    'description': 'Person detected inside forklift',
                                    'boxes': [person_box, forklift_box],
                                    'timestamp': current_time
                                })
                                self.person_forklift_cooldown = 60 * fps
                                break

                for i in final_indices:
                    box = all_boxes[i]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    label = f"{all_class_names[i]}: {all_confidences[i]:.2f}"
                    color = all_colors[i]
                    
                    is_hazard = False
                    for hazard in hazards_in_frame:
                        for hazard_box in hazard['boxes']:
                            if np.array_equal(box, hazard_box):
                                is_hazard = True
                                if hazard['severity'] == 'CRITICAL':
                                    color = (0, 0, 255)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                                elif hazard['severity'] == 'HIGH':
                                    color = (0, 255, 255)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                else:
                                    color = (0, 165, 255)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                break
                        if is_hazard:
                            break
                    
                    if not is_hazard:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                message_y_offset = 10
                for hazard in hazards_in_frame:
                    if hazard['severity'] == 'CRITICAL':
                        bg_color, text_color = (0, 0, 255), (255, 255, 255)
                    elif hazard['severity'] == 'HIGH':
                        bg_color, text_color = (0, 255, 255), (0, 0, 0)
                    else:
                        bg_color, text_color = (0, 165, 255), (255, 255, 255)
                    
                    message = f"{hazard['type']}"
                    (text_w, text_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (10, message_y_offset), (min(10 + text_w + 20, frame_width - 10), message_y_offset + 40), bg_color, -1)
                    cv2.putText(frame, message, (20, message_y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    message_y_offset += 50
                
                if hazards_in_frame:
                    self.hazards_detected.extend(hazards_in_frame)
                    for i, hazard in enumerate(hazards_in_frame):
                        temp_snapshot_path = os.path.join(output_dir, f"temp_frame.jpg")
                        cv2.imwrite(temp_snapshot_path, frame)

                        is_duplicate = False
                        for existing_snapshot in self.frame_snapshots:
                            if existing_snapshot['hazard']['type'] == hazard['type']:
                                similarity = self.histogram_similarity(temp_snapshot_path, existing_snapshot['path'])
                                if similarity > self.SIMILARITY_THRESHOLD:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            snapshot_path = os.path.join(output_dir, f"hazard_frame_{frame_count}_{i}.jpg")
                            os.rename(temp_snapshot_path, snapshot_path)
                            self.frame_snapshots.append({
                                'frame_number': frame_count,
                                'timestamp': current_time,
                                'path': snapshot_path,
                                'hazard': hazard
                            })
                        else:
                            os.remove(temp_snapshot_path)
                        
                        for box in hazard['boxes']:
                            center_x = int((box[0] + box[2]) / 2)
                            center_y = int((box[1] + box[3]) / 2)
                            self.hazard_zones[hazard['type']].append((center_x, center_y))
                
                out.write(frame)
                
                if total_frames > 0:
                    percent_complete = int((frame_count / total_frames) * 100)
                    progress_bar.progress(percent_complete)
                status_text.text(f"Processing frame {frame_count}/{total_frames}...")
       
        finally:
            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()
        
        st.session_state.processing_data = {
            'output_video_path': output_video_path,
            'hazards_detected': self.hazards_detected,
            'frame_snapshots': self.frame_snapshots,
            'user_id': self.user_id,
            'video_name': self.video_name,
            'processing_start_time': self.processing_start_time,
            'hazard_zones': self.hazard_zones
        }
        
        return output_video_path
