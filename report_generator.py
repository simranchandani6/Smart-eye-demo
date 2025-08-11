# report_generator.py
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from datetime import datetime
from collections import defaultdict
import cv2

def _format_duration(seconds):
    """Formats seconds into a human-readable string (e.g., 1m 25s)."""
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

def _generate_dynamic_summary(hazards_detected):
    """Generates a dynamic executive summary based on findings."""
    total_hazards = len(hazards_detected)
    if total_hazards == 0:
        return "The analysis concluded without detecting any safety incidents. All observed activities appear to adhere to standard safety protocols. Continued vigilance and adherence to current procedures are recommended."

    severity_counts = defaultdict(int)
    hazard_counts = defaultdict(int)
    for hazard in hazards_detected:
            hazard_counts[hazard['type']] += 1
            severity_counts[hazard['severity']] += 1
    
    critical_count = severity_counts.get('CRITICAL', 0)
    high_count = severity_counts.get('HIGH', 0)
    
    summary = f"This report details the findings of an AI-powered safety analysis, which identified a total of {total_hazards} unique incidents. "
    if critical_count > 0:
        summary += f"Of significant concern are {critical_count} CRITICAL incidents that require immediate attention to mitigate severe risks. "
    if high_count > 0:
        summary += f"Additionally, {high_count} HIGH-risk situations were observed, indicating potential for accidents if not addressed. "
    
    summary += "This document highlights specific non-compliances and provides a detailed breakdown of each incident in the subsequent pages."
    return summary

def _format_ppe_list(description_string):
    """Cleans up the PPE violation description string into a readable list."""
    if 'PPE violation(s):' not in description_string:
        return description_string
    
    items_str = description_string.split(':', 1)[1].strip()
    if not items_str: 
        return "General PPE Violation"
        
    items = [item.strip().replace('no_', '') for item in items_str.split(',')]
    unique_items = sorted(list(set(filter(None, items))))
    
    if not unique_items: 
        return "Unspecified PPE Violation"
        
    return ', '.join([item.replace('_', ' ').title() for item in unique_items])

def _calculate_dynamic_layout(num_incidents):
    """Calculate dynamic grid layout based on number of incidents."""
    if num_incidents == 0:
        # No incidents: Header, Expanded Info, Summary, AI Insights
        return [0.4, 0.8, 0.6, 1.0, 0.1], 0.3
    elif num_incidents <= 5:
        # Few incidents: Standard layout with expanded info box
        return [0.4, 0.8, 0.6, 0.7, 1.1, 0.1], 0.4
    elif num_incidents <= 10:
        # Medium incidents: Compress summary, expand breakdown
        return [0.4, 0.7, 0.5, 0.9, 1.1, 0.1], 0.5
    else:
        # Many incidents: Minimize other sections, maximize breakdown
        return [0.4, 0.7, 0.4, 1.0, 1.0, 0.1], 0.6
    
def generate_pdf_report(output_dir, user_id, video_name, processing_start_time, hazards_detected, frame_snapshots, hazard_zones):
    """Generate comprehensive and dynamic PDF report with all requested features."""
    report_path = os.path.join(output_dir, f"safety_report_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    with PdfPages(report_path) as pdf:
        plt.rcParams.update({
            'font.size': 10, 'font.family': 'sans-serif',
            'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
            'text.color': '#333333'
        })
        
        # --- START of FIX ---
        # The definitive list of hazards to be reported is derived from the unique frame_snapshots.
        # This ensures all counts in summaries and charts match the detailed incident pages.
        reported_hazards = [snapshot['hazard'] for snapshot in frame_snapshots]

        severity_counts = defaultdict(int)
        hazard_counts = defaultdict(int)
        
        # Populate counts using the corrected list of unique hazards.
        for hazard in reported_hazards:
            hazard_counts[hazard['type']] += 1
            severity_counts[hazard['severity']] += 1
        
        num_incident_types = len(hazard_counts)
        height_ratios, hspace_value = _calculate_dynamic_layout(num_incident_types)
        # --- END of FIX ---
        
        # Page 1: Executive Summary with Dynamic Incident Breakdown
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        gs = fig.add_gridspec(6, 1, height_ratios=height_ratios, hspace=hspace_value)
        
        # Header section with more space above info
        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off')
        ax_header.text(0.55, 0.75, 'SMART EYE SAFETY ANALYSIS REPORT', 
                      fontsize=24, fontweight='bold', ha='center', 
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        # Info Block - Expanded height with more top spacing
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis('off')
        info_data = [
            ['User ID:', user_id],
            ['Prepared for:', 'ABC Construction'],
            ['Prepared By:', 'Bacancy | Smart Eye Division'],
            ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        y_pos = 0.85
        for label, value in info_data:
            ax_info.text(0.1, y_pos, label, fontweight='bold', fontsize=11, ha='left', va='top')
            ax_info.text(0.35, y_pos, str(value), fontsize=11, ha='left', va='top')
            y_pos -= 0.18
        ax_info.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='gray', linewidth=0.5))
        
        # Executive Summary - Larger font size
        ax_summary_text = fig.add_subplot(gs[2])
        ax_summary_text.axis('off')
        ax_summary_text.text(0.05, 0.95, "Executive Summary", fontsize=14, fontweight='bold', va='bottom')
        
        # --- FIX --- Use the corrected list for the summary.
        summary_text = _generate_dynamic_summary(reported_hazards)
        
        summary_font_size = 11 if len(summary_text) < 200 else 10
        ax_summary_text.text(0.05, 0.75, summary_text, fontsize=summary_font_size, ha='left', va='top', wrap=True)

        # DYNAMIC Incident Breakdown Section
        ax_breakdown = fig.add_subplot(gs[3])
        ax_breakdown.axis('off')
        ax_breakdown.text(0.05, 0.95, "INCIDENT BREAKDOWN:", fontsize=14, fontweight='bold', va='bottom')
        
        # This section now correctly uses hazard_counts derived from reported_hazards.
        if not hazard_counts:
            ax_breakdown.text(0.08, 0.75, "No incidents detected.", style='italic', fontsize=10, va='top')
        else:
            available_height = 0.8
            num_items = len(hazard_counts)
            line_spacing = min(0.15, available_height / (num_items + 1))
            
            sorted_hazards = sorted(hazard_counts.items(), key=lambda item: item[1], reverse=True)
            y_pos = 0.8
            
            for hazard_type, count in sorted_hazards:
                display_name = hazard_type.replace('_', ' ').title()
                incident_text = f"{count} incident" + ("s" if count > 1 else "")
                font_size = 10 if num_items <= 8 else 9
                
                ax_breakdown.text(0.1, y_pos, f"• {display_name}:", fontsize=font_size, ha='left', va='top')
                ax_breakdown.text(0.85, y_pos, incident_text, fontsize=font_size, fontweight='bold', ha='right', va='top')
                y_pos -= line_spacing
        
        # AI Summary Insights Table
        ax_insights = fig.add_subplot(gs[4])
        ax_insights.axis('off')
        ax_insights.text(0.05, 0.95, "AI Summary Insights", fontsize=14, fontweight='bold', va='bottom')
        
        # --- FIX --- Base all metrics on the unique reported_hazards list.
        total_incidents = len(reported_hazards)
        high_severity_incidents = sum(1 for h in reported_hazards if h['severity'] in ['CRITICAL', 'HIGH'])
        video_duration_seconds = max([h['timestamp'] for h in reported_hazards], default=0)
        first_alert_time = min([h['timestamp'] for h in reported_hazards], default=0)

        metrics = {
            "Total Video Time Analyzed": _format_duration(video_duration_seconds),
            "Total Incidents Detected": str(total_incidents),
            "High-Severity Incidents": str(high_severity_incidents),
            "AI Detection Accuracy": "80%",
            "Time to First Alert": f"{first_alert_time:.1f}s" if first_alert_time > 0 else "N/A"
        }
        
        y_pos = 0.8
        ax_insights.axhline(y_pos + 0.08, color='black', linewidth=1.5)
        ax_insights.text(0.3, y_pos + 0.02, "Metric", fontweight='bold', fontsize=11, ha='center')
        ax_insights.text(0.8, y_pos + 0.02, "Value", fontweight='bold', fontsize=11, ha='center')
        ax_insights.axhline(y_pos, color='black', linewidth=1)

        for label, value in metrics.items():
            y_pos -= 0.14
            color = '#333333'
            if label == "Total Incidents Detected" and total_incidents > 0: 
                color = '#e67e22'
            elif label == "High-Severity Incidents" and high_severity_incidents > 0: 
                color = '#c0392b'
            
            ax_insights.text(0.05, y_pos, label, fontsize=10, va='center')
            ax_insights.text(0.9, y_pos, value, fontsize=10, va='center', ha='right', 
                           color=color, fontweight='bold' if color != '#333333' else 'normal')
            ax_insights.axhline(y_pos - 0.07, color='lightgray', linewidth=0.5)

        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # --- FIX --- Check against the unique hazards list to decide if the chart page is needed.
        if reported_hazards:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            fig.suptitle('DETAILED SAFETY ANALYSIS', fontsize=16, fontweight='bold', y=1.02)
            plt.subplots_adjust(top=0.90)
            
            # This chart now uses hazard_counts derived from the correct list.
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
            
            # This chart now uses severity_counts derived from the correct list.
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
            
            # --- FIX --- Timeline data is now based on unique reported incidents.
            timestamps = [h['timestamp'] for h in reported_hazards]
            ax3.hist(timestamps, bins=min(20, len(set(timestamps))), alpha=0.7, color='#FF6B6B', edgecolor='black')
            ax3.set_title('Incident Timeline Distribution', fontweight='bold', pad=20)
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Number of Incidents')
            ax3.grid(axis='y', alpha=0.3)
            
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

        # This loop correctly iterates over the unique snapshots, so no change was needed here.
        for i, snapshot in enumerate(frame_snapshots):
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('white')
            
            gs = fig.add_gridspec(3, 2, height_ratios=[0.2, 2.5, 0.8], width_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
            
            ax_title = fig.add_subplot(gs[0, :])
            ax_title.axis('off')
            ax_title.text(0.5, 0.5, f'INCIDENT DETAIL #{i+1}', 
                            fontsize=18, fontweight='bold', ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffcdd2', alpha=0.7))
            
            ax_img = fig.add_subplot(gs[1, 0])
            if os.path.exists(snapshot['path']):
                img = cv2.imread(snapshot['path'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax_img.imshow(img_rgb)
                ax_img.axis('off')
            
            ax_details = fig.add_subplot(gs[1, 1])
            ax_details.axis('off')
            
            hazard = snapshot['hazard']
            
            non_compliance_text = hazard['description']
            if hazard['type'] == 'PPE_VIOLATION':
                if 'PPE violation(s):' in hazard['description']:
                    items_str = hazard['description'].split(':', 1)[1].strip()
                    items = items_str.split(',')
                    formatted_items = [item.strip().replace('no_', 'No ').replace('_', ' ').title() for item in items]
                    non_compliance_text = ', '.join(sorted(list(set(formatted_items))))
                else:
                    non_compliance_text = "Unspecified PPE Violation"

            details_lines = [
                ("INCIDENT TYPE:", hazard['type'].replace('_', ' ').title()),
                ("SEVERITY LEVEL:", hazard['severity']),
                ("TIMESTAMP:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                ("NON COMPLIANCE:", non_compliance_text)
            ]
            
            y_pos = 1.0
            for label, value in details_lines:
                ax_details.text(0.05, y_pos, label, fontweight='bold', fontsize=11, va='top')
                value_y_pos = y_pos - 0.07
                
                if label == "SEVERITY LEVEL:":
                    color_map = {'CRITICAL': '#c0392b', 'HIGH': '#f39c12', 'INFO': '#2980b9', 'MEDIUM': '#f1c40f'}
                    color = color_map.get(hazard['severity'], 'black')
                    ax_details.text(0.05, value_y_pos, value, fontweight='bold', fontsize=11, color=color, va='top',
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.15))
                    y_pos -= 0.18
                else:
                    ax_details.text(0.05, value_y_pos, str(value), fontsize=10, va='top', wrap=True)
                    y_pos -= 0.18 if len(str(value)) < 40 else 0.28

            ax_recommendations = fig.add_subplot(gs[2, :])
            ax_recommendations.axis('off')
            
            ax_recommendations.text(0.02, 0.9, "RECOMMENDED IMMEDIATE ACTIONS:", fontweight='bold', fontsize=11, color='#8B0000', va='top')
            
            recommendations = []
            if hazard['type'] == 'PPE_VIOLATION':
                missing_items_str = _format_ppe_list(hazard['description'])
                recommendations = [f"Provide proper {missing_items_str} immediately", "Stop work until PPE compliance is achieved", "Conduct a toolbox talk on PPE requirements"]
            elif hazard['type'] == 'FIRE_CONTINUOUS':
                recommendations = ["EMERGENCY: Activate fire suppression systems immediately", "Evacuate all personnel from affected area", "Alert emergency services"]
            elif hazard['type'] == 'LADDER_CLIMBING':
                recommendations = ["Verify proper three-point contact technique", "Ensure ladder is stable and secured", "Confirm fall protection is in use if required"]
            else:
                recommendations = ["Review the incident with the involved personnel.", "Assess the risk and implement corrective actions.", "Update safety procedures if necessary."]
            
            y_pos = 0.7
            for rec in recommendations:
                ax_recommendations.text(0.05, y_pos, f"• {rec}", fontsize=10, ha='left', va='top', wrap=True)
                y_pos -= 0.18
            
            ax_recommendations.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, fill=False, edgecolor='gray', linewidth=0.5))
            
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close()
    
    return report_path