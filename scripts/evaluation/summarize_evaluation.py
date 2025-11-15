import json
import os
from collections import defaultdict
import pandas as pd

def analyze_detection_results(file_path):
    """Analyzes a single detection JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    summary = defaultdict(lambda: defaultdict(list))
    
    if 'results' in data and 'by_condition' in data['results']:
        for condition, images in data['results']['by_condition'].items():
            if not images:
                continue

            total_images = len(images)
            images_with_detections = 0
            total_detections = 0
            confidences = []

            for img in images:
                num_detections = img.get('num_detections', 0)
                if num_detections > 0:
                    images_with_detections += 1
                    total_detections += num_detections
                    for detection in img.get('detections', []):
                        confidences.append(detection.get('confidence', 0.0))

            detection_rate = (images_with_detections / total_images) * 100 if total_images > 0 else 0
            avg_detections_per_image = total_detections / total_images if total_images > 0 else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            summary[condition] = {
                'total_images': total_images,
                'images_with_detections': images_with_detections,
                'detection_rate_percent': round(detection_rate, 2),
                'avg_detections_per_image': round(avg_detections_per_image, 2),
                'avg_confidence': round(avg_confidence, 4)
            }
            
    # Calculate overall summary
    all_images = [img for cond_images in data.get('results', {}).get('by_condition', {}).values() for img in cond_images]
    if all_images:
        total_images = len(all_images)
        images_with_detections = 0
        total_detections = 0
        confidences = []
        for img in all_images:
            num_detections = img.get('num_detections', 0)
            if num_detections > 0:
                images_with_detections += 1
                total_detections += num_detections
                for detection in img.get('detections', []):
                    confidences.append(detection.get('confidence', 0.0))
        
        detection_rate = (images_with_detections / total_images) * 100 if total_images > 0 else 0
        avg_detections_per_image = total_detections / total_images if total_images > 0 else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        summary['overall'] = {
            'total_images': total_images,
            'images_with_detections': images_with_detections,
            'detection_rate_percent': round(detection_rate, 2),
            'avg_detections_per_image': round(avg_detections_per_image, 2),
            'avg_confidence': round(avg_confidence, 4)
        }

    return dict(summary)

def main():
    """Main function to process all evaluation files."""
    evaluation_dir = 'outputs/evaluation'
    output_file = 'outputs/evaluation/evaluation_summary.json'
    
    files_to_analyze = [
        'detection_baseline.json',
        'detection_finetuned.json',
        'detection_finetuned_v1.json',
        'detection_finetuned_v2.json'
    ]
    
    all_summaries = {}

    for filename in files_to_analyze:
        file_path = os.path.join(evaluation_dir, filename)
        if os.path.exists(file_path):
            model_name = filename.replace('.json', '')
            print(f"Analyzing {model_name}...")
            all_summaries[model_name] = analyze_detection_results(file_path)
        else:
            print(f"File not found: {file_path}")

    with open(output_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
        
    print(f"Summary saved to {output_file}")

    # Optional: Print a markdown table for quick view
    print("\n--- Markdown Summary Table ---")
    md_table = "| Model | Condition | Detection Rate (%) | Avg Detections/Image | Avg Confidence |\n"
    md_table += "|---|---|---|---|---|\n"
    for model, summary in all_summaries.items():
        for condition, metrics in summary.items():
            md_table += f"| {model} | {condition} | {metrics['detection_rate_percent']} | {metrics['avg_detections_per_image']} | {metrics['avg_confidence']} |\n"
    print(md_table)


if __name__ == "__main__":
    main()
