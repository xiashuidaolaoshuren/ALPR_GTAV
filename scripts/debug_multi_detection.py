"""
Debug script to visualize images with multiple detections.
Shows why NMS didn't filter them (low IoU = different regions).
"""

import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.model import load_detection_model, detect_plates

def main():
    # Images with multiple detections
    test_images = [
        'outputs/test_images/day_clear_angle_00024.jpg',
        'outputs/test_images/day_clear_angle_00033.jpg'
    ]
    
    # Load model
    print("Loading detection model...")
    model = load_detection_model('models/detection/yolov8n.pt', device='auto')
    
    for img_path in test_images:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path}")
        print('='*60)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_path}")
            continue
        
        # Detect plates
        detections = detect_plates(image, model, conf_threshold=0.25, iou_threshold=0.45)
        
        print(f"\nFound {len(detections)} detection(s):")
        for i, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
            w, h = x2 - x1, y2 - y1
            area = w * h
            print(f"  Detection {i}: ({x1},{y1})-({x2},{y2}) "
                  f"size={w}x{h} area={area}px conf={conf:.3f}")
            
            # Draw bounding box
            color = (0, 255, 0) if i == 1 else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Det{i}: {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate IoU between detections if multiple
        if len(detections) >= 2:
            from src.detection.model import calculate_iou
            
            print("\nIoU Matrix:")
            for i in range(len(detections)):
                for j in range(i+1, len(detections)):
                    iou = calculate_iou(detections[i][:4], detections[j][:4])
                    print(f"  Det{i+1} <-> Det{j+1}: IoU = {iou:.4f}")
                    
                    # Check overlap
                    if iou > 0.3:
                        print(f"    → HIGH OVERLAP (should be filtered by NMS)")
                    elif iou > 0.1:
                        print(f"    → MEDIUM OVERLAP")
                    else:
                        print(f"    → LOW/NO OVERLAP (different objects)")
        
        # Save annotated image
        output_path = f"outputs/debug_{Path(img_path).name}"
        cv2.imwrite(output_path, image)
        print(f"\nAnnotated image saved to: {output_path}")

if __name__ == '__main__':
    main()
