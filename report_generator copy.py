# report_generator.py
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from datetime import datetime
from collections import defaultdict
import cv2

def generate_pdf_report(output_dir, user_id, video_name, processing_start_time, hazards_detected, frame_snapshots, hazard_zones):
    """Generate comprehensive PDF report with improved UI"""
    report_path = os.path.join(output_dir, f"safety_report_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    with PdfPages(report_path) as pdf:
        # Set style for consistent formatting
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        # Page 1: Executive Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        gs = fig.add_gridspec(4, 1, height_ratios=[0.8, 1.2, 1.5, 0.5], hspace=0.3)
        
        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off')
        
        ax_header.text(0.5, 0.8, 'SMART EYE SAFETY ANALYSIS REPORT', 
                      fontsize=24, fontweight='bold', ha='center', 
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        ax_header.text(0.5, 0.4, f'Analysis Report - {datetime.now().strftime("%B %d, %Y")}', 
                      fontsize=14, ha='center', style='italic')
        
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis('off')
        
        info_data = [
            ['User ID:', user_id],
            ['Video File:', video_name],
            ['Analysis Date:', processing_start_time.strftime('%Y-%m-%d %H:%M:%S')],
            ['Processing Duration:', f'{(datetime.now() - processing_start_time).total_seconds():.1f} seconds']
        ]
        
        y_pos = 0.8
        for label, value in info_data:
            ax_info.text(0.1, y_pos, label, fontweight='bold', fontsize=12)
            ax_info.text(0.4, y_pos, str(value), fontsize=12)
            y_pos -= 0.2
        
        ax_info.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='gray', linewidth=1))
        
        ax_results = fig.add_subplot(gs[2])
        ax_results.axis('off')
        
        hazard_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for hazard in hazards_detected:
            hazard_counts[hazard['type']] += 1
            severity_counts[hazard['severity']] += 1
        
        total_hazards = len(hazards_detected)
        
        if total_hazards == 0:
            status_text = "✅ NO CRITICAL SAFETY INCIDENTS DETECTED"
            status_color = 'green'
            ax_results.text(0.5, 0.9, status_text, fontsize=16, fontweight='bold', 
                           ha='center', color=status_color,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
            
            summary_text = "Standard operations observed throughout the video analysis.\nAll safety protocols appear to be followed correctly."
            ax_results.text(0.1, 0.7, summary_text, fontsize=12, ha='left')
            
        else:
            critical_count = severity_counts.get('CRITICAL', 0)
            high_count = severity_counts.get('HIGH', 0)
            
            if critical_count > 0:
                status_color = 'red'
                status_text = f"{critical_count} CRITICAL SAFETY VIOLATIONS DETECTED"
                bg_color = 'lightcoral'
            elif high_count > 0:
                status_color = 'orange'
                status_text = f"{high_count} HIGH-RISK SITUATIONS IDENTIFIED"
                bg_color = 'lightyellow'
            else:
                status_color = 'blue'
                status_text = f"{severity_counts.get('MEDIUM', 0) + severity_counts.get('INFO', 0)} SAFETY OBSERVATIONS RECORDED"
                bg_color = 'lightblue'
            
            ax_results.text(0.5, 0.9, status_text, fontsize=14, fontweight='bold',
                           ha='center', color=status_color,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, alpha=0.5))
            
            y_pos = 0.7
            ax_results.text(0.1, y_pos, "INCIDENT BREAKDOWN:", fontweight='bold', fontsize=12)
            y_pos -= 0.08
            
            for hazard_type, count in hazard_counts.items():
                hazard_display = hazard_type.replace('_', ' ').title()
                ax_results.text(0.15, y_pos, f"• {hazard_display}:", fontsize=11)
                ax_results.text(0.6, y_pos, f"{count} incidents", fontsize=11, fontweight='bold')
                y_pos -= 0.06
        
        ax_recommendations = fig.add_subplot(gs[3])
        ax_recommendations.axis('off')
        
        recommendations_text = "IMMEDIATE ACTIONS REQUIRED:\n"
        
        if severity_counts.get('CRITICAL', 0) > 0:
            recommendations_text += "🔴 Emergency response activation • Immediate area evacuation • Safety protocol review"
        elif severity_counts.get('HIGH', 0) > 0:
            recommendations_text += "🟡 Enhanced supervision required • Safety training review • Procedure updates"
        elif severity_counts.get('MEDIUM', 0) > 0 or severity_counts.get('INFO', 0) > 0:
            recommendations_text += "🔵 PPE compliance monitoring • Equipment inspections • Training updates"
        else:
            recommendations_text += "✅ Continue current protocols • Regular monitoring • Preventive maintenance"
        
        ax_recommendations.text(0.1, 0.5, recommendations_text, fontsize=10, ha='left',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
       
            # Page 2: Detailed Statistics (Remove heat map)
        if hazards_detected:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            fig.suptitle('DETAILED SAFETY ANALYSIS', fontsize=16, fontweight='bold', y=1.02)
            plt.subplots_adjust(top=0.90)
            
            # Hazard type distribution
            hazard_types = list(hazard_counts.keys())
            hazard_values = list(hazard_counts.values())
            
            colors = ['#FF4444' if 'FIRE' in ht or 'HEAVY_VEHICLE' in ht 
                        else '#FFA500' if 'LADDER' in ht 
                        else '#4169E1' for ht in hazard_types]
            
            bars = ax1.bar(range(len(hazard_types)), hazard_values, color=colors, alpha=0.8)
            ax1.set_title('Incidents by Type', fontweight='bold', pad=20)
            ax1.set_xticks(range(len(hazard_types)))
            ax1.set_xticklabels([ht.replace('_', '\n') for ht in hazard_types], 
                                rotation=0, ha='center', fontsize=9)
            ax1.set_ylabel('Number of Incidents')
            ax1.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            # Severity distribution
            if severity_counts:
                severity_types = list(severity_counts.keys())
                severity_values = list(severity_counts.values())
                severity_colors = {'CRITICAL': '#FF4444', 'HIGH': '#FFA500', 'MEDIUM': '#4169E1', 'INFO': 'royalblue'}
                
                wedges, texts, autotexts = ax2.pie(severity_values, labels=severity_types, autopct='%1.0f%%',
                        colors=[severity_colors.get(st, 'gray') for st in severity_types],
                        startangle=90, textprops={'fontsize': 10})
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(11)
                
                ax2.set_title('Incidents by Severity Level', fontweight='bold', pad=20)
            
            # Timeline of hazards
            timestamps = [h['timestamp'] for h in hazards_detected]
            ax3.hist(timestamps, bins=min(20, len(set(timestamps))), alpha=0.7, color='#FF6B6B', edgecolor='black')
            ax3.set_title('Incident Timeline Distribution', fontweight='bold', pad=20)
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Number of Incidents')
            ax3.grid(axis='y', alpha=0.3)
            
            # Incident frequency by minute
            timestamps_minutes = [int(t/60) for t in timestamps]
            if timestamps_minutes:
                minute_counts = defaultdict(int)
                for minute in timestamps_minutes:
                    minute_counts[minute] += 1
                
                minutes = list(minute_counts.keys())
                counts = list(minute_counts.values())
                
                ax4.bar(minutes, counts, alpha=0.8, color='#4ECDC4', edgecolor='black')
                ax4.set_title('Incidents per Minute', fontweight='bold', pad=20)
                ax4.set_xlabel('Time (minutes)')
                ax4.set_ylabel('Number of Incidents')
                ax4.grid(axis='y', alpha=0.3)
                
                for i, v in enumerate(counts):
                    ax4.text(minutes[i], v + 0.05*max(counts), str(v), 
                            ha='center', va='bottom', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No time-based data available', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, style='italic')
                ax4.set_title('Incidents per Minute', fontweight='bold', pad=20)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Pages 3+: Individual hazard details with screenshots
        for i, snapshot in enumerate(frame_snapshots[:8]):  # Limit to first 8
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            
            gs = fig.add_gridspec(3, 2, height_ratios=[0.2, 2, 0.8], width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
            
            ax_title = fig.add_subplot(gs[0, :])
            ax_title.axis('off')
            ax_title.text(0.5, 0.5, f'INCIDENT DETAIL #{i+1}', 
                            fontsize=18, fontweight='bold', ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
            
            ax_img = fig.add_subplot(gs[1, 0])
            if os.path.exists(snapshot['path']):
                img = cv2.imread(snapshot['path'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax_img.imshow(img_rgb)
                ax_img.set_title(f"Frame {snapshot['frame_number']} - {snapshot['timestamp']:.1f}s", 
                                fontweight='bold', pad=10)
                ax_img.axis('off')
                
                for spine in ax_img.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_color('black')
            
            ax_details = fig.add_subplot(gs[1, 1])
            ax_details.axis('off')
            
            hazard = snapshot['hazard']
            
            details_lines = [
                ("INCIDENT TYPE:", hazard['type'].replace('_', ' ').title()),
                ("SEVERITY LEVEL:", hazard['severity']),
                ("TIMESTAMP:", f"{hazard['timestamp']:.2f} seconds"),
                ("FRAME NUMBER:", str(snapshot['frame_number'])),
                ("", ""),
                ("DESCRIPTION:", ""),
                ("", hazard['description'])
            ]
            
            y_pos = 0.95
            for label, value in details_lines:
                if label == "SEVERITY LEVEL:":
                    color_map = {'CRITICAL': 'red', 'HIGH': 'orange', 'INFO': 'blue', 'MEDIUM': 'gold'}
                    color = color_map.get(hazard['severity'], 'black')
                    ax_details.text(0.05, y_pos, label, fontweight='bold', fontsize=11)
                    ax_details.text(0.05, y_pos-0.08, value, fontweight='bold', fontsize=11, color=color,
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.2))
                    y_pos -= 0.16
                elif label and value:
                    ax_details.text(0.05, y_pos, label, fontweight='bold', fontsize=11)
                    ax_details.text(0.05, y_pos-0.08, str(value), fontsize=10)
                    y_pos -= 0.16
                elif not label and not value:
                    y_pos -= 0.05
                else:
                    ax_details.text(0.05, y_pos, value, fontsize=10, style='italic', wrap=True)
                    y_pos -= 0.12
            
            ax_recommendations = fig.add_subplot(gs[2, :])
            ax_recommendations.axis('off')
            
            recommendations_title = "RECOMMENDED IMMEDIATE ACTIONS:\n"
            ax_recommendations.text(0.02, 0.85, recommendations_title, 
                                    fontweight='bold', fontsize=12, color='darkred')
            
            recommendations = []
            if hazard['type'] == 'FIRE_CONTINUOUS':
                recommendations = ["EMERGENCY: Activate fire suppression systems immediately", "Evacuate all personnel from affected area", "Alert emergency services and fire department"]
            elif hazard['type'] == 'FIRE_PERSON_PROXIMITY':
                recommendations = ["EMERGENCY: Immediate personnel evacuation required", "Deploy fire suppression equipment", "Emergency medical team on standby"]
            elif hazard['type'] == 'HEAVY_VEHICLE_FIRE':
                recommendations = ["STOP: Halt all heavy vehicle operations immediately", "Move vehicles away from fire source if safe", "Deploy industrial fire suppression systems"]
            elif hazard['type'] == 'FIRE_FORKLIFT_PROXIMITY':
                recommendations = ["HALT: Stop forklift operation near the fire immediately", "Move forklift to a safe distance if possible", "Assess fire risk and deploy appropriate extinguisher"]
            elif hazard['type'] == 'LADDER_CLIMBING':
                recommendations = ["Verify proper three-point contact technique", "Ensure ladder stability and proper securing", "Confirm safety harness and fall protection"]
            elif hazard['type'] == 'PPE_VIOLATION':
                ppe_type = hazard.get('ppe_type', 'unknown').replace('no_', '')
                recommendations = [f"Provide proper {ppe_type} immediately", "Stop work until PPE compliance achieved", "Conduct PPE compliance training"]
            elif hazard['type'] == 'PERSON_IN_FORKLIFT':
                recommendations = ["Verify forklift operator certification and authorization", "Ensure proper safety protocols are followed", "Check if operator is using required safety equipment"]
            
            y_pos = 0.6
            for rec in recommendations:
                ax_recommendations.text(0.05, y_pos, f"• {rec}", fontsize=10, ha='left')
                y_pos -= 0.15
            
            ax_recommendations.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.8, fill=False, edgecolor='gray', linewidth=1))
            
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close()
    
    return report_path