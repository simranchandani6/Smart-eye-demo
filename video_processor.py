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
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            return 0.0

        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return round(similarity * 100, 2)

    def calculate_overlap_percentage(self, box1, box2):
        """Calculate intersection over union (IoU) percentage"""
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
        """Calculate distance between centers of two bounding boxes"""
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2
        
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def is_ppe_inside_person(self, ppe_box, person_box, threshold=0.5):
        """Check if PPE detection is inside person bounding box"""
        ppe_center_x = (ppe_box[0] + ppe_box[2]) / 2
        ppe_center_y = (ppe_box[1] + ppe_box[3]) / 2
        
        return (person_box[0] <= ppe_center_x <= person_box[2] and 
                person_box[1] <= ppe_center_y <= person_box[3])
    

    
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

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / fps

                hazard_results = self.hazard_model(frame, verbose=False)
                ppe_results = self.ppe_model(frame, verbose=False)
                
                all_boxes, all_confidences, all_class_names, all_colors = [], [], [], []
                
                def append_detections(results, model, color_map):
                    if len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu()
                        confs = results[0].boxes.conf.cpu()
                        cls_ids = results[0].boxes.cls.cpu().int()
                        for i in range(len(boxes)):
                            class_name = model.names[cls_ids[i].item()]
                            conf = confs[i].item()
                            
                            threshold = self.LADDER_CONF_THRESHOLD if class_name == 'ladder' else self.CONF_THRESHOLD
                            
                            if conf > threshold:
                                all_boxes.append(boxes[i].numpy())
                                all_confidences.append(conf)
                                all_class_names.append(class_name)
                                all_colors.append(color_map.get(class_name, self.DEFAULT_COLOR))
                
                append_detections(hazard_results, self.hazard_model, self.HAZARD_COLOR_MAP)
                append_detections(ppe_results, self.ppe_model, self.PPE_COLOR_MAP)
                
                final_indices = []
                if len(all_boxes) > 0:
                    xywh_boxes = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in all_boxes]
                    unique_class_names = set(all_class_names)
                    
                    for class_name in unique_class_names:
                        class_indices = [i for i, name in enumerate(all_class_names) if name == class_name]
                        class_boxes = [xywh_boxes[i] for i in class_indices]
                        class_confidences = [all_confidences[i] for i in class_indices]
                        
                        if len(class_boxes) > 0:
                            indices_after_nms = cv2.dnn.NMSBoxes(class_boxes, class_confidences, 
                                                               self.CONF_THRESHOLD, self.NMS_THRESHOLD)
                            if len(indices_after_nms) > 0:
                                final_indices.extend([class_indices[i] for i in indices_after_nms.flatten()])
                
                hazards_in_frame = []
                
                if self.continuous_fire_cooldown > 0:
                    self.continuous_fire_cooldown -= 1
                if self.ladder_person_cooldown > 0:
                    self.ladder_person_cooldown -= 1
                if self.fire_person_cooldown > 0:
                    self.fire_person_cooldown -= 1
                if self.ppe_violation_cooldown > 0:
                    self.ppe_violation_cooldown -= 1
                if self.heavy_vehicle_fire_cooldown > 0:
                    self.heavy_vehicle_fire_cooldown -= 1
                if self.fire_forklift_cooldown > 0:
                    self.fire_forklift_cooldown -= 1
                if self.person_forklift_cooldown > 0:
                    self.person_forklift_cooldown -= 1
                
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
                
                for person_idx, person_box in person_detections:
                    person_key = f"person_{person_idx}"
                    no_classes = []
                    for ppe_idx, ppe_box, ppe_class in ppe_detections:
                        if self.is_ppe_inside_person(ppe_box, person_box) and ppe_class.startswith("no_"):
                            no_classes.append(ppe_class)
                    person_ppe_tracking[person_key].append((frame_count, len(no_classes), list(no_classes), person_box))
                    
                    person_ppe_tracking[person_key] = [
                        x for x in person_ppe_tracking[person_key] if frame_count - x[0] < self.ppe_violation_window * fps
                    ]
                    
                    if len(person_ppe_tracking[person_key]) >= self.ppe_violation_window * fps:
                        max_no = max(person_ppe_tracking[person_key], key=lambda x: x[1])
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
                    triggered = False
                    for fire_idx, fire_box in fire_detections:
                        for person_idx, person_box in person_detections:
                            overlap_pct = self.calculate_overlap_percentage(fire_box, person_box)
                            if overlap_pct >= 10.0:
                                hazards_in_frame.append({
                                    'type': 'FIRE_PERSON_PROXIMITY',
                                    'severity': 'CRITICAL',
                                    'description': f'Person detected near fire ',
                                    'boxes': [fire_box, person_box],
                                    'timestamp': current_time
                                })
                                self.fire_person_cooldown = 60
                                triggered = True
                                break
                        if triggered:
                            break
                
                if self.fire_forklift_cooldown == 0:
                    triggered = False
                    for fire_idx, fire_box in fire_detections:
                        for forklift_idx, forklift_box in forklift_detections:
                            overlap_pct = self.calculate_overlap_percentage(fire_box, forklift_box)
                            distance = self.calculate_distance(fire_box, forklift_box)
                            if overlap_pct >= 5.0 or distance < 100:
                                hazards_in_frame.append({
                                    'type': 'FIRE_FORKLIFT_PROXIMITY',
                                    'severity': 'MEDIUM',
                                    'description': f'Forklift near fire',
                                    'boxes': [fire_box, forklift_box],
                                    'timestamp': current_time
                                })
                                self.fire_forklift_cooldown = 60
                                triggered = True
                                break
                        if triggered:
                            break

                fire_heavy_found_this_frame = False
                interacting_boxes_this_frame = []
                interaction_detected = False
                for _, vehicle_box in heavy_vehicle_detections:
                    for _, fire_box in fire_detections:
                        overlap_pct = self.calculate_overlap_percentage(vehicle_box, fire_box)
                        distance = self.calculate_distance(vehicle_box, fire_box)
                        if overlap_pct >= 5.0 or distance < 100:
                            fire_heavy_found_this_frame = True
                            interacting_boxes_this_frame = [vehicle_box, fire_box]
                            interaction_detected = True
                            break
                    if interaction_detected:
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
                    triggered = False
                    for ladder_idx, ladder_box in ladder_detections:
                        for person_idx, person_box in person_detections:
                            overlap_pct = self.calculate_overlap_percentage(ladder_box, person_box)
                            if overlap_pct >= self.ladder_person_threshold:
                                hazards_in_frame.append({
                                    'type': 'LADDER_CLIMBING',
                                    'severity': 'INFO',
                                    'description': f'Person on/near ladder',
                                    'boxes': [ladder_box, person_box],
                                    'timestamp': current_time
                                })
                                self.ladder_person_cooldown = 60
                                triggered = True
                                break
                        if triggered:
                            break

                if self.person_forklift_cooldown == 0:
                    triggered = False
                    for person_idx, person_box in person_detections:
                        for forklift_idx, forklift_box in forklift_detections:
                            if (person_box[0] >= forklift_box[0] and person_box[1] >= forklift_box[1] and
                                person_box[2] <= forklift_box[2] and person_box[3] <= forklift_box[3]):
                                hazards_in_frame.append({
                                    'type': 'PERSON_IN_FORKLIFT',
                                    'severity': 'INFO',
                                    'description': 'Person detected inside forklift',
                                    'boxes': [person_box, forklift_box],
                                    'timestamp': current_time
                                })
                                self.person_forklift_cooldown = 60 * fps
                                triggered = True
                                break
                        if triggered:
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
