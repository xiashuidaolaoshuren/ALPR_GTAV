# GTA V ALPR Data Acquisition Strategy

## Executive Summary

This document outlines the comprehensive strategy for collecting high-quality license plate images from GTA V gameplay for training and testing the ALPR system. The strategy balances automation possibilities with practical manual collection methods to ensure diverse, representative dataset.

---

## 1. ScriptHookV Integration Research

### 1.1 Overview
**ScriptHookV** is a library that allows custom scripts to access GTA V's internal functions, enabling gameplay automation for data collection.

### 1.2 Capabilities

**Vehicle Spawning & Control:**
- Spawn specific vehicles programmatically
- Control vehicle positioning and orientation
- Teleport vehicles to desired locations
- Set vehicle colors and customizations

**Environment Control:**
- Set time of day (0-24 hours)
- Change weather conditions (clear, rain, fog, etc.)
- Control lighting conditions
- Modify world state

**Camera Control:**
- Position camera at precise coordinates
- Set camera angle and field of view
- Follow vehicles automatically
- Capture screenshots at specified intervals

### 1.3 Installation Requirements

**Prerequisites:**
- GTA V (PC version) with latest update
- ScriptHookV library (http://www.dev-c.com/gtav/scripthookv/)
- Script Hook V .NET (for C# scripts)
- Visual Studio 2019+ (for developing custom scripts)

**Installation Steps:**
1. Download ScriptHookV from official website
2. Extract `ScriptHookV.dll`, `dinput8.dll` to GTA V root directory
3. Create `scripts/` folder in GTA V root
4. Install Script Hook V .NET if using C# scripts
5. Place custom scripts in scripts/ folder

### 1.4 Feasibility Assessment

✅ **Advantages:**
- Fully automated data collection (24/7 operation possible)
- Precise control over capture conditions
- Consistent dataset quality
- Rapid collection (thousands of images per hour)
- Reproducible scenarios

❌ **Challenges:**
- Requires C++/C# programming knowledge
- GTA V updates may break ScriptHookV compatibility
- Anti-cheat concerns (use offline mode only)
- Initial setup complexity (4-8 hours for script development)
- Debugging and testing time required

**Recommendation for Week 1:**  
⚠️ **Manual collection for initial dataset** (simpler, faster for small datasets)  
🔧 **ScriptHookV automation for Week 3+** (when dataset expansion is needed)

---

## 2. Capture Scenario Matrix

### 2.1 Time of Day Conditions (4 variations)

| Condition | In-Game Time | Characteristics | Priority |
|-----------|--------------|-----------------|----------|
| **Day** | 12:00-15:00 | Direct sunlight, high contrast | HIGH |
| **Night** | 22:00-03:00 | Artificial lighting, low visibility | HIGH |
| **Dawn** | 05:00-07:00 | Soft lighting, low angle sun | MEDIUM |
| **Dusk** | 18:00-20:00 | Orange tint, shadows | MEDIUM |

### 2.2 Weather Conditions (5 variations)

| Condition | Characteristics | Difficulty Level | Priority |
|-----------|-----------------|------------------|----------|
| **Clear** | Normal visibility, no effects | Easy | HIGH |
| **Overcast** | Cloudy, diffused light | Easy | HIGH |
| **Rain** | Wet surfaces, reflections | Hard | HIGH |
| **Fog** | Reduced visibility | Hard | MEDIUM |
| **Thunderstorm** | Rain + lightning | Very Hard | LOW |

### 2.3 Camera Angles (6+ variations)

**Front View:**
- 0° (dead center): Most common real-world angle
- ±15° (slight angle): Natural driving perspective

**Rear View:**
- 180° (dead center): Rear plate capture
- ±15° (slight angle): Following vehicle

**Side Views:**
- 45° (front-quarter): Common intersection angle
- 90° (perpendicular): Side street view
- 135° (rear-quarter): Passing vehicle

**Priority:** Front 0°, Rear 180°, Front 45° (capture these first)

### 2.4 Distance Categories (3 variations)

| Distance | Range | Use Case | Difficulty |
|----------|-------|----------|------------|
| **Close** | 2-5 meters | Parking lots, stopped vehicles | Easy |
| **Medium** | 5-15 meters | Traffic monitoring, intersections | Medium |
| **Far** | 15-30 meters | Highway monitoring | Hard |

### 2.5 Lighting Scenarios (4 variations)

- **Direct Sunlight**: High contrast, clear text
- **Shadow**: Vehicle in shadow, plate partially dark
- **Street Lamps**: Night lighting, yellow/orange tint
- **Headlights**: Back-lit from vehicle behind

### 2.6 Occlusion & Challenges (5 variations)

- **Dirt/Mud**: Dirty plates (low priority - rare in GTA V)
- **Partial Occlusion**: Objects partially covering plate
- **Motion Blur**: Moving vehicles (moderate speed)
- **Low Resolution**: Far distance, small plate size
- **Glare**: Reflections on plate surface

### 2.7 Target Scenario Distribution

**Minimum Initial Dataset (Week 1): 100-200 images**

| Scenario | Target Count | Notes |
|----------|--------------|-------|
| Day + Clear + Front | 30 | Core training data |
| Day + Clear + Rear | 20 | Rear plate training |
| Night + Clear + Front | 20 | Low-light training |
| Rain + Any Angle | 15 | Weather variation |
| Various Angles (45°, 90°) | 15 | Angle robustness |

**Expanded Dataset (Week 3+): 1000+ images**

---

## 3. Recording Tools & Settings

### 3.1 Option 1: OBS Studio (Recommended)

**Advantages:**
- Free and open-source
- Highly configurable
- Good performance
- Supports multiple audio/video sources

**Installation:**
1. Download from https://obsproject.com/
2. Install with default settings
3. Configure output settings

**Recommended Settings:**
```
Output Mode: Advanced
Encoder: NVIDIA NVENC H.264 (if NVIDIA GPU) or x264
Rate Control: CBR
Bitrate: 15000-20000 Kbps
Keyframe Interval: 2
Preset: Quality (NVENC) or veryfast (x264)
Profile: high
Resolution: 1920x1080
FPS: 60
```

**Recording Hotkey Setup:**
- Start/Stop Recording: F9
- Enable Replay Buffer for instant capture

### 3.2 Option 2: NVIDIA ShadowPlay

**Advantages:**
- Minimal performance impact
- Instant Replay feature
- Built-in for NVIDIA GPU users

**Requirements:**
- NVIDIA GeForce GTX 600 series or newer
- GeForce Experience installed

**Recommended Settings:**
```
Quality: High
Resolution: In-game (1920x1080)
Frame rate: 60 FPS
Bitrate: 50 Mbps
Audio: In-game audio off (optional)
```

**Usage:**
- Alt + F9: Start/Stop recording
- Alt + F10: Save last 5 minutes (Instant Replay)

### 3.3 Option 3: Windows Game Bar

**Advantages:**
- Built into Windows 10/11
- No additional software needed
- Simple interface

**Limitations:**
- Less configurable than OBS
- Higher performance impact
- Limited format options

**Usage:**
- Windows + G: Open Game Bar
- Windows + Alt + R: Start/Stop recording

**Recommended Settings:**
```
Video quality: High
Frame rate: 60 fps
Audio quality: 192 kbps
Capture location: Dedicated SSD
```

### 3.4 Post-Recording Processing

**File Organization:**
The recommended practice is to organize raw footage into directories based on capture conditions and to use descriptive filenames.

```
GTA5_Recordings/
├── day_clear/
│   ├── day_clear_01.mp4
│   └── day_clear_02.mp4
├── night_clear/
│   └── night_clear_01.mp4
└── metadata.txt
```

**File Naming Convention:**
Files should be renamed to reflect their content, making them easier to manage.
- **Pattern:** `{condition}_{location}_{number}.mp4`
- **Example:** `day_clear_airport_01.mp4`

**Metadata Logging:**
A simple text file (`metadata.txt`) should be maintained to log key details about each recording session.

**`metadata.txt` Example:**
```
day_clear_01.mp4
- Location: Airport parking lot
- Vehicles: 15-20 sedans, 5 trucks
- Quality: Good, clear lighting

night_clear_01.mp4
- Location: Downtown street
- Vehicles: 10 sedans
- Quality: Good, street lamps
```

---

## 4. Manual Data Collection Procedure

### 4.1 Pre-Collection Setup

**Step 1: Configure GTA V Settings**
- Graphics: High quality (better plate readability)
- Resolution: 1920x1080 or higher
- Anti-aliasing: Enabled (smoother text)
- Disable HUD (F1 or through settings)

**Step 2: Choose Collection Location**
- **Recommended Locations:**
  - Los Santos International Airport (open space, good lighting)
  - Rockford Hills (residential, varied angles)
  - Del Perro Freeway (highway scenarios)
  - Vespucci Beach parking lots (stationary vehicles)

**Step 3: Start Recording Software**
- Configure settings as per Section 3
- Test recording quality before main collection
- Verify disk space (20-50 GB per hour of footage)

### 4.2 Collection Workflow

**Phase 1: Stationary Vehicle Capture (Easier)**

1. Find parked vehicles or spawn traffic
2. Approach vehicle from various angles
3. Use character camera controls:
   - Right mouse button: Hold to look around
   - Mouse movement: Adjust angle
   - Mouse wheel: Zoom in/out (if supported by mods)
4. Capture 3-5 seconds at each angle
5. Move to next vehicle

**Target:** 20-30 vehicles per 30-minute session

**Phase 2: Moving Vehicle Capture (More Challenging)**

1. Position character near traffic flow
2. Observe approaching vehicles
3. Follow vehicle with camera
4. Capture 5-10 seconds of footage
5. Use cinematic camera (V key) for smooth tracking

**Target:** 10-15 vehicles per 30-minute session

**Phase 3: Environmental Variations**

1. Change time of day:
   - Quick method: Save game, reload at different time
   - Manual method: Wait in-game (1 min real = 2 hours in-game)
   
2. Change weather:
   - Method 1: Wait for natural weather change
   - Method 2: Use cheat codes (if enabled):
     ```
     ~  (open console)
     makeitrain  (rain)
     ```

3. Capture same scenarios under new conditions

### 4.3 Quality Checklist

For each captured clip, verify:
- [ ] License plate clearly visible in frame
- [ ] Plate is in focus (not blurry)
- [ ] Adequate lighting (text readable)
- [ ] Minimal motion blur
- [ ] At least 3 seconds of usable footage
- [ ] Metadata recorded (time, weather, angle)

### 4.4 Session Planning

**Week 1 Target: 100-200 images**

**Session 1 (2 hours): Day + Clear Weather**
- Location: Airport + Freeway
- Focus: Front and rear angles
- Target: 50 vehicles

**Session 2 (1.5 hours): Night + Clear Weather**
- Location: Downtown + Beach
- Focus: Artificial lighting scenarios
- Target: 30 vehicles

**Session 3 (1 hour): Various Weather**
- Capture rain, overcast conditions
- Various angles
- Target: 20 vehicles

**Total Time Investment: 4.5 hours for initial dataset**

---

## 5. Frame Extraction Strategy

### 5.1 Extraction Parameters

**Frame Rate:**
- Source video: 60 FPS
- Extraction rate: 5-10 FPS (every 6-12 frames)
- Rationale: Reduce redundant frames, keep diversity

**Quality:**
- Format: PNG or JPEG
- Quality: 95% (JPEG) or lossless (PNG)
- Resolution: Original (1920x1080)

**Naming Convention:**
```
{video_filename_base}_{frame_number}.jpg

Examples:
day_clear_01_00123.jpg
night_rain_03_00456.jpg
```

This convention uses the base name of the source video file, ensuring that frames are easily traceable to their original clip.

### 5.2 Automated Extraction Script

**Script:** `scripts/extract_frames.py`

**Pseudocode:**
```python
def extract_frames(video_path, output_dir, fps=5, quality=95):
    # Load video using OpenCV
    video = cv2.VideoCapture(video_path)
    source_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(source_fps / fps)
    
    video_filename_base = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            # Generate filename from video name and frame number
            filename = f"{video_filename_base}_{extracted_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted_count += 1
        
        frame_count += 1
    
    video.release()
    return extracted_count
```

**Usage:**
```bash
python scripts/extract_frames.py --input raw_footage/day_clear/day_clear_01.mp4 \
                                  --output datasets/lpr/train/images/ \
                                  --fps 5 \
                                  --quality 95
```

### 5.3 Manual Frame Selection (Alternative)

For higher quality curation:
1. Use VLC Media Player or similar tool
2. Play video at slow speed (0.25x)
3. Pause on frames with clear plate visibility
4. Press Shift+S to save frame
5. Manually rename and organize

**Trade-off:** Higher quality but more time-consuming (10-15 images/hour vs. 100+ with automation)

---

## 6. Quality Assurance

### 6.1 Image Quality Criteria

**Include:**
- Plate clearly visible and in-focus
- Text readable (at least 4-5 characters identifiable)
- Adequate contrast between plate and background
- Minimal motion blur
- Plate occupies at least 20x50 pixels

**Exclude:**
- Extremely blurry images
- Plate too small (< 15x30 pixels)
- Severe occlusion (> 50% of plate hidden)
- Corrupted or distorted frames
- Duplicate or near-duplicate frames

### 6.2 Annotation Workflow

**Tool:** Label Studio (already in requirements.txt)

1. Import extracted images
2. Configure YOLO format annotation
3. Draw bounding boxes around plates
4. Export in YOLO format to datasets/lpr/

**Quality Control:**
- Double-check annotations for accuracy
- Verify normalized coordinates (0.0-1.0)
- Ensure one annotation per file
- Review annotations periodically

---

## 7. Dataset Scaling Strategy

### 7.1 Week 1: Initial Collection (100-200 images)

**Focus:** Core scenarios for initial testing
- Day + Clear + Front: Primary training data
- Basic angle variations
- Manual collection acceptable

### 7.2 Week 3: Expanded Collection (500-1000 images)

**Focus:** Model fine-tuning and robustness
- All weather conditions
- All angles
- Consider ScriptHookV automation

### 7.3 Week 5+: Large-Scale Collection (2000+ images)

**Focus:** Production-ready model
- Implement ScriptHookV automation
- Augment with synthetic data
- Collect edge cases and failures

---

## 8. Automation Roadmap (Future)

### 8.1 Phase 1: Semi-Automation (Week 3)
- Automated frame extraction
- Batch processing scripts
- Metadata generation

### 8.2 Phase 2: ScriptHookV Integration (Week 4-5)
- Custom C# script for vehicle spawning
- Automated weather/time cycling
- Camera positioning automation
- Screenshot capture at specified intervals

### 8.3 Phase 3: Full Automation (Week 6+)
- End-to-end data pipeline
- Quality filtering algorithms
- Automatic YOLO annotation (assisted)
- Continuous dataset expansion

---

## 9. Budget & Resources

### 9.1 Time Investment

| Activity | Week 1 | Week 3 | Week 5+ |
|----------|--------|--------|---------|
| Manual collection | 4-5 hours | - | - |
| Script development | - | 6-8 hours | 4-6 hours |
| Annotation | 3-4 hours | 8-10 hours | 15-20 hours |
| Quality control | 1-2 hours | 2-3 hours | 4-6 hours |

### 9.2 Storage Requirements

- Raw footage (1080p@60fps): ~3 GB per hour
- Extracted frames (JPEG 95%): ~500 KB per image
- Dataset (1000 images): ~500 MB
- Model checkpoints: ~100-500 MB

**Recommended:** 50-100 GB free space for full project lifecycle

### 9.3 Hardware Requirements

**Minimum:**
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8 GB
- Storage: 50 GB available
- GPU: NVIDIA GTX 1050 Ti (2GB VRAM)

**Recommended:**
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16 GB
- Storage: 100 GB SSD
- GPU: NVIDIA RTX 3060 or better (6GB+ VRAM)

---

## 10. References & Resources

**ScriptHookV:**
- Official website: http://www.dev-c.com/gtav/scripthookv/
- Documentation: http://www.dev-c.com/nativedb/
- Community forum: GTAForums.com

**Recording Tools:**
- OBS Studio: https://obsproject.com/
- OBS Settings Guide: https://obsproject.com/wiki/

**GTA V Modding:**
- GTA5-Mods.com: Community mods and resources
- OpenIV: Asset extraction tool

**Computer Vision:**
- OpenCV Frame Extraction: https://docs.opencv.org/
- Label Studio: https://labelstud.io/

---

## 11. Conclusion

**Week 1 Recommendation:**
✅ **Manual collection** for initial 100-200 images using OBS Studio  
✅ **Focus on core scenarios:** Day/Clear/Front angles  
✅ **Use automated frame extraction** via `extract_frames.py`  
✅ **Annotate with Label Studio** in YOLO format  

**Future Expansion:**
🔧 **Week 3+:** Implement ScriptHookV automation for large-scale collection  
🔧 **Continuous improvement:** Analyze model failures, collect targeted data  

This strategy balances immediate needs (small, high-quality dataset) with future scalability (automated large-scale collection). The manual approach for Week 1 reduces complexity while establishing a solid foundation for dataset expansion.

---

**Document Version:** 1.0  
**Author:** GTA V ALPR Development Team  
**Last Updated:** Week 1, Task 4  
**Status:** Ready for implementation
