# Label Studio Project Configuration

This directory contains configuration files and templates for Label Studio annotation projects.

## Project Setup

### 1. Start Label Studio
```bash
python scripts/annotation/start_annotation.py
```

### 2. Create Project
1. Access http://localhost:8080
2. Click "Create Project"
3. Enter project details:
   - **Project Name:** GTA_V_License_Plate_Detection
   - **Description:** License plate detection dataset for GTA V ALPR system

### 3. Configure Labeling Interface

Use the following XML configuration for the labeling interface:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="license_plate" background="#FF6600"/>
  </RectangleLabels>
</View>
```

**Interface Settings:**
- ✅ Enable zoom and pan
- ✅ Show image metadata
- ✅ Enable keyboard shortcuts
- ✅ Auto-save annotations

### 4. Import Data

**Option A: Local Files**
- Navigate to "Import" tab
- Select "Upload Files"
- Choose images from: `outputs/test_images/`
- Recommended: Start with 50-100 images

**Option B: Directory Import**
- Use "Import from Directory" feature
- Point to: `D:\Felix_stuff\ALPR_GTA5\outputs\test_images`
- Label Studio will scan and import all images

### 5. Configure Export

**Export Format:** YOLO
- Go to Settings → Export
- Select "YOLO" as export format
- Configure output structure:
  ```
  datasets/lpr/
  ├── images/
  └── labels/
  ```

## Annotation Workflow

See detailed instructions in: [docs/annotation_guide.md](../annotation_guide.md)

### Quick Reference
1. Review image
2. Draw tight bounding boxes around license plates
3. Ensure 2-3 pixels padding
4. Verify all characters are within the box
5. Submit annotation (Ctrl+Enter)

## Quality Control

### Target Metrics
- **Annotation Accuracy:** >95%
- **Inter-annotator Agreement:** >90%
- **Coverage:** All visible plates annotated

### Review Process
1. Annotate 20-30 images as pilot batch
2. Review for consistency
3. Adjust guidelines if needed
4. Scale to full dataset

## Export and Training

### YOLO Format Output
Each annotation creates a `.txt` file with format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 0 (license_plate)
- Coordinates: Normalized to [0, 1] range
- `x_center`, `y_center`: Center of bounding box
- `width`, `height`: Box dimensions

### Example
```
0 0.512 0.723 0.089 0.045
```

## Backup and Version Control

### Regular Backups
- Export annotations every 50-100 images
- Save to: `datasets/lpr/backups/YYYY-MM-DD/`
- Include both images and labels

### Version Control
- Track annotation progress in project log
- Document any schema changes
- Maintain annotation statistics

## Troubleshooting

### Common Issues

**Issue: Images not displaying**
- Check file paths are correct
- Verify supported formats (.jpg, .png)
- Ensure Label Studio has read permissions

**Issue: Slow performance**
- Reduce batch size (annotate 20-50 at a time)
- Clear browser cache
- Use Chrome or Firefox

**Issue: Export fails**
- Check disk space
- Verify export directory exists
- Try smaller batches

## Contact

For questions or issues:
- Check: [docs/annotation_guide.md](../annotation_guide.md)
- Label Studio Docs: https://labelstud.io/guide/
- Project repository: Issue tracker

---

**Last Updated:** 2025-10-09
