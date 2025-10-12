# License Plate Annotation Guide

**Project:** GTA V ALPR - License Plate Detection Dataset Creation  
**Tool:** Label Studio  
**Date:** 2025-10-08

---

## Table of Contents

1. [Setup](#setup)
2. [Annotation Rules](#annotation-rules)
3. [Quality Guidelines](#quality-guidelines)
4. [Workflow](#workflow)
5. [Keyboard Shortcuts](#keyboard-shortcuts)
6. [Export Instructions](#export-instructions)
7. [Troubleshooting](#troubleshooting)

---

## Setup

### Prerequisites
- Label Studio installed: `pip install label-studio`
- Project initialized (see setup instructions below)

### Accessing Label Studio

1. **Start Label Studio:**
   ```bash
   label-studio start
   ```
   Or use the startup script:
   ```bash
   python scripts/annotation/start_annotation.py
   ```

2. **Access the web interface:**
   - Open browser and navigate to: http://localhost:8080
   - Login with your account credentials

3. **Open the project:**
   - Select **"GTA_V_License_Plate_Detection"** from the project list
   - Click **"Label"** to start annotating

---

## Annotation Rules

### What to Annotate
- **All visible license plates** in GTA V screenshots, regardless of their readability.
- If you can identify an object as a license plate, you **must** label it.
- Both front and rear license plates should be annotated.

### Bounding Box Requirements

#### ✅ Correct Bounding Boxes:
- **Tight fit:** Box should closely surround the license plate
- **Minimum padding:** 2-3 pixels around all edges
- **Include entire plate:** All characters must be within the box
- **Rectangular shape:** Follow the plate's rectangular outline
- **No rotation:** Use axis-aligned rectangles (Label Studio doesn't support rotated boxes)

#### ❌ Incorrect Bounding Boxes:
- Too loose (excessive padding > 10 pixels)
- Too tight (cutting off parts of the plate)
- Including too much of the vehicle bumper
- Missing characters at the edges
- Overlapping with other objects unnecessarily

### Label Classes and Attributes
- **Bounding Box:** `license_plate` (single class)
- **Readability Attribute (per box):**
  - `clear`: Plate is sharp, in focus, and fully readable.
  - `blurred`: Plate is out of focus, motion-blurred, or otherwise hard to read.
  - `occluded`: Part of the plate is blocked by an object.

**Crucially, you must select a readability option for every bounding box you draw.** This is a required step.

### Special Cases

The goal is to label **every** visible license plate. The `readability` attribute will handle quality assessment.

#### Multiple Plates in One Image:
- **Annotate all visible plates**.
- Each plate gets its own bounding box and its own `readability` tag.

#### Partially Occluded Plates:
- **Annotate them.**
- Draw a tight bounding box around the visible portion of the plate.
- Select the `occluded` readability attribute.

#### Blurry or Low-Quality Plates:
- **Annotate them.**
- Draw a bounding box around the plate.
- Select the `blurred` readability attribute. Even if you can't read any characters, if you can identify it as a plate, it must be labeled.

#### Distant/Small Plates:
- **Annotate them.**
- Use the zoom feature to draw an accurate box.
- If the plate is too small to be certain it's a plate, you can skip it. Otherwise, label it and choose `blurred` or `clear` based on its appearance.

---

## Quality Guidelines

### Annotation Quality Standards

#### High-Quality Annotation:
- Bounding box tightly fits the plate
- All characters are fully contained
- No excessive background included
- Consistent padding (2-3 pixels)
- Correct label class applied

#### What to Avoid:
- Inconsistent box sizes (sometimes loose, sometimes tight)
- Cutting off parts of characters
- Including large portions of vehicle body
- Forgetting to annotate secondary plates in the same image
- Using wrong label class

### Consistency Tips
- **Zoom in** to verify character visibility before annotating
- **Check your previous annotations** to maintain consistency
- **Take breaks** every 30-60 minutes to maintain focus
- **Review your work** before submitting a batch

### When to Skip an Image
- No license plates are visible at all.
- The image is so dark, bright, or corrupted that you cannot identify any objects.

**Do not skip an image just because a license plate is blurry, occluded, or distant.** The purpose of the `readability` attribute is to capture this information. If you can tell it's a plate, you must label it.

---

## Workflow

### Step-by-Step Annotation Process

#### 1. Review Image
- **Examine the image** for visible license plates
- Use **zoom** (scroll wheel) if needed to see details
- **Pan** (click and drag) to navigate around large images
- Identify all license plates that meet annotation criteria

#### 2. Draw Bounding Box
- Click the **"Rectangle"** tool in the toolbar
- Click and drag to draw a box around the license plate
- **Adjust corners** by dragging to fit tightly
- Ensure 2-3 pixels padding around the plate edges

#### 3. Select Readability
- After drawing the box, **you must select one of the readability choices**:
  - `clear`: The plate is sharp and easy to read.
  - `blurred`: The plate is out of focus, motion-blurred, or too distant to read clearly.
  - `occluded`: Part of the plate is blocked by another object.

#### 4. Verify Annotation
- **Double-check** that all characters are within the box.
- Verify the box is tight (not too loose or too tight).
- Confirm the correct label class (`license_plate`) and a `readability` attribute are applied.
- If multiple plates exist, ensure all are annotated with their own attributes.

#### 5. Add Comments (Optional)
- Click **"Add comment"** for highly unusual cases.
- Example: "Plate is reflected in a window, labeling the reflection."

#### 6. Submit Annotation
- Click **"Submit"** button (or press `Ctrl+Enter`).
- Annotation is saved and next image loads automatically.

#### 7. Track Progress
- Monitor the progress bar at the top.
- Review your annotations periodically for consistency.
- Take breaks to maintain annotation quality.

---

## Keyboard Shortcuts

### Essential Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Submit annotation and move to next image |
| `Ctrl + Backspace` | Delete selected bounding box |
| `Ctrl + Z` | Undo last action |
| `Ctrl + Shift + Z` | Redo action |
| `Space` | Pan mode toggle |
| `+` or `=` | Zoom in |
| `-` or `_` | Zoom out |
| `0` | Reset zoom to 100% |

### Navigation Shortcuts
| Shortcut | Action |
|----------|--------|
| `Alt + Left Arrow` | Previous image |
| `Alt + Right Arrow` | Next image |
| `Esc` | Cancel current annotation |

### Tips for Efficiency
- Learn and use shortcuts to speed up annotation
- Use `Space` to quickly switch between pan and select modes
- Use `Ctrl + Z` liberally to fix mistakes
- Use `Ctrl + Enter` to quickly submit and move to next image

---

## Export Instructions

### When to Export
- After completing a batch of 20-50 annotations
- Before taking a long break
- When ready to train/fine-tune the model
- Regularly for backup purposes

### Export Steps

#### 1. Access Export Menu
- Click **"Export"** button in the project view
- Select export format

#### 2. Choose Format
- **Select:** `YOLO` format
- This will generate:
  - `classes.txt` - List of label classes
  - `notes.json` - Annotation metadata
  - Individual `.txt` files for each image with normalized coordinates

#### 3. Configure Export Options
- ✅ **Include images:** Check this option to export images along with annotations
- ✅ **Include label info:** Include class names
- Export to: `datasets/lpr/` directory

#### 4. Download and Organize
```
datasets/lpr/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

#### 5. Verify Export
- Check that `.txt` files contain normalized coordinates (0-1 range)
- Format: `<class_id> <x_center> <y_center> <width> <height>`
- Example: `0 0.512 0.723 0.089 0.045`
- Ensure all annotated images have corresponding label files

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Label Studio won't start
**Solution:**
```bash
# Check if port 8080 is already in use
netstat -ano | findstr :8080

# If in use, stop the process or use a different port
label-studio start --port 8090
```

#### Issue: Images not loading
**Solution:**
- Check image file paths are correct
- Ensure images are in a supported format (.jpg, .png)
- Verify Label Studio has read permissions for the image directory
- Try re-importing the dataset

#### Issue: Cannot draw bounding boxes
**Solution:**
- Ensure you're in annotation mode (not review mode)
- Check that the Rectangle tool is selected
- Refresh the page and try again
- Clear browser cache if problem persists

#### Issue: Annotations not saving
**Solution:**
- Check browser console for errors (F12)
- Ensure stable internet connection
- Try using a different browser (Chrome/Firefox recommended)
- Export current work as backup

#### Issue: Lost annotations
**Solution:**
- Check the "Annotations" tab in the project
- Use the export feature to recover data
- Enable auto-save if available in settings

### Performance Tips
- **Close unused browser tabs** to free up memory
- **Use Chrome or Firefox** for best performance
- **Annotate in batches** (20-50 images at a time)
- **Export regularly** to avoid data loss

---

## Best Practices Summary

### 🎯 Do's:
- ✅ Draw tight bounding boxes (2-3 pixels padding)
- ✅ Annotate all visible plates in an image
- ✅ Zoom in to verify character visibility
- ✅ Use comments for uncertain cases
- ✅ Export regularly for backup
- ✅ Take breaks to maintain quality
- ✅ Review your work periodically

### ⛔ Don'ts:
- ❌ Don't include excessive background
- ❌ Don't cut off parts of characters
- ❌ Don't skip secondary plates without reason
- ❌ Don't rush through annotations
- ❌ Don't annotate blurry/invisible plates
- ❌ Don't forget to submit before closing

---

## Contact and Support

### Questions or Issues?
- **Documentation:** Check this guide first
- **Label Studio Docs:** https://labelstud.io/guide/
- **Project Lead:** [Your contact information]

### Feedback
If you encounter issues not covered in this guide or have suggestions for improvement, please document them and share with the team.

---

**Version:** 1.1  
**Last Updated:** 2025-10-09  
**Author:** ALPR Project Team
