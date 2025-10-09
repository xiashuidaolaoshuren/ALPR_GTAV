# GTA V ALPR Data Collection Guide
## Quick Start Guide for Task 5: Initial Test Dataset Collection

**Goal:** Collect 50-100 test images from GTA V gameplay with diverse conditions.

**Target Distribution:**
- ✅ Day + Clear: 10-15 images
- ✅ Day + Rain: 5-10 images  
- ✅ Night + Clear: 10-15 images
- ✅ Night + Rain: 5-10 images
- ✅ Various angles: Front (30%), Rear (30%), Side/Angled (40%)

---

## Option 1: Record Your Own Gameplay (Recommended)

### Prerequisites
- GTA V installed on PC
- Recording software installed (OBS Studio, NVIDIA ShadowPlay, or Windows Game Bar)
- At least 20GB free disk space
- 2-3 hours of time

### Step-by-Step Process

#### 1️⃣ Setup Recording Software

**Option A: OBS Studio (Free, Best Quality)**

**Installation:**
1. Download from https://obsproject.com/download
2. Run installer → Accept defaults → Install
3. Launch OBS Studio

**Configuration Steps:**

**Add Game Capture Source:**
1. In OBS, click **"+"** under "Sources"
2. Select **"Game Capture"**
3. Name it "GTA V" → OK
4. Properties:
   - Mode: **"Capture specific window"**
   - Window: **"Grand Theft Auto V"** (launch GTA V first to see it)
   - Uncheck "Capture mouse cursor"
   - Click OK

**Configure Output Settings:**
1. Click **Settings** (bottom right)
2. **Output** tab → Change to **"Advanced"** mode
3. Recording sub-tab:
   ```
   Format: mp4
   Encoder: NVIDIA NVENC H.264 (NVIDIA GPU) or x264 (AMD/Intel)
   Rate Control: CBR
   Bitrate: 15000 Kbps
   Keyframe Interval: 2
   Preset: Quality (NVENC) or veryfast (x264)
   ```

**Video Settings:**
1. **Video** tab in Settings:
   ```
   Base Resolution: 1920x1080
   Output Resolution: 1920x1080
   FPS: 60
   ```
2. Click Apply → OK

**Set Recording Hotkey:**
1. Settings → **Hotkeys**
2. "Start Recording" → Set to **F9**
3. "Stop Recording" → Set to **F9** (toggles)
4. Apply → OK

**Choose Save Location:**
1. Settings → Output → Recording
2. Recording Path → Browse to folder with space
3. Suggested: Create `D:\GTA5_Recordings\` or similar
4. Click Apply → OK

✅ Keep OBS running in background while you play GTA V

**Option B: NVIDIA ShadowPlay (NVIDIA GPUs)**
1. Open GeForce Experience (Alt + Z)
2. Settings → Recordings:
   - Quality: High
   - Resolution: 1920x1080
   - Frame rate: 60 FPS
3. Use Alt + F9 to start/stop recording

**Option C: Windows Game Bar (Built-in)**
1. Press Windows + G to open Game Bar
2. Settings:
   - Video quality: High
   - Frame rate: 60 fps
3. Use Windows + Alt + R to start/stop recording

#### 2️⃣ Configure GTA V

1. Launch GTA V
2. Settings → Graphics:
   - Graphics Quality: High
   - Resolution: 1920x1080 or higher
   - Anti-aliasing: On
3. Disable HUD:
   - Press F1 in game (or through settings)
   - This removes on-screen UI for cleaner images

#### 3️⃣ Recording Session Plan

**Session 1: Day + Clear Weather (30-45 minutes)**
- Location: Los Santos International Airport or Rockford Hills
- Focus: Front and rear angles
- Target: 15-20 vehicles
- Save videos to: `outputs/raw_footage/day_clear/`
- Naming: `day_clear_01.mp4`, `day_clear_02.mp4`, etc.

**Session 2: Night + Clear Weather (30-45 minutes)**
- Location: Downtown Los Santos or Del Perro
- Focus: Various angles with street lighting
- Target: 15-20 vehicles
- Save videos to: `outputs/raw_footage/night_clear/`

**Session 3: Rainy Conditions (20-30 minutes)**
- Change weather in-game:
  - Wait for natural rain, OR
  - Use weather cheat (if enabled): Open console (~) and type `makeitrain`
- Mix of day and night
- Target: 10-15 vehicles
- Save videos to: `outputs/raw_footage/day_rain/` or `outputs/raw_footage/night_rain/`

**Session 4: Additional Angles (15-30 minutes)**
- Focus on side views, 45° angles
- Various lighting conditions
- Target: 10 vehicles

#### 4️⃣ Recording Tips

**For Stationary Vehicles:**
1. Find parked cars or spawn traffic (if using mods)
2. Walk around vehicle slowly
3. Record 3-5 seconds from each angle:
   - Front (0°)
   - Side (90°)
   - Rear-side (135°)
   - Rear (180°)
4. Hold camera steady, ensure plate is in frame

**For Moving Vehicles:**
1. Position character near traffic
2. Use cinematic camera (V key) for smooth tracking
3. Record 5-10 seconds following vehicle
4. Focus on vehicles driving at moderate speeds

**Camera Controls:**
- Right mouse button: Hold to look around
- Mouse movement: Adjust angle
- Mouse wheel: Zoom (if supported)
- V key: Toggle cinematic camera mode

#### 5️⃣ After Recording

Once you have recorded videos in all 4 condition folders, proceed to "Frame Extraction" section below.

---


## Option 2: Use Public GTA V Footage (Alternative)

If you don't have GTA V or can't record right now, you can use publicly available footage.

### Finding Suitable Footage

**YouTube Search Terms:**
- "GTA 5 gameplay license plates"
- "GTA V traffic police POV"
- "GTA 5 driving gameplay 4K"

**Requirements:**
- ✅ High quality (1080p or better)
- ✅ Clear view of vehicles and license plates
- ✅ Unedited gameplay (no overlays, text, or HUD if possible)
- ✅ Various conditions (day/night, weather)
- ✅ At least 5-10 minutes of footage

### Downloading Videos

**Method 1: youtube-dl or yt-dlp (Recommended)**
```powershell
# Install yt-dlp (if not installed)
pip install yt-dlp

# Download video
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" -o "outputs/raw_footage/day_clear/%(title)s.%(ext)s" [VIDEO_URL]
```

**Method 2: Online Downloader**
- Use services like savefrom.net or y2mate.com
- Download in MP4 format, 1080p quality
- Save to appropriate condition folder

### Attribution
Create a file `outputs/raw_footage/sources.txt` listing:
- Video URL
- Creator name
- License/usage terms
- Date downloaded

Example:
```
day_clear_01.mp4
- Source: https://www.youtube.com/watch?v=XXXXX
- Creator: John Doe Gaming
- License: Creative Commons / Fair Use for educational purposes
- Downloaded: 2025-10-05
```

---

## Frame Extraction (For Both Options)

Once you have video files in `outputs/raw_footage/`, extract frames using the provided script.

### Method 1: Extract from Single Video

```powershell
# Navigate to project root
cd "d:\Felix's_stuff\ALPR_GTA5"

# Extract frames from one video (5 FPS = 1 frame every 200ms)
python scripts/data_ingestion/extract_frames.py --input "outputs/raw_footage/day_clear/day_clear_01.mp4" --output "outputs/test_images/" --fps 5 --quality 95
```

### Method 2: Batch Extract from All Videos

```powershell
# Extract from all videos in day_clear folder
python scripts/data_ingestion/extract_frames.py --batch --input_dir "outputs/raw_footage/day_clear/" --output_dir "outputs/test_images/" --fps 5

# Repeat for each condition
python scripts/data_ingestion/extract_frames.py --batch --input_dir "outputs/raw_footage/day_rain/" --output_dir "outputs/test_images/" --fps 5
python scripts/data_ingestion/extract_frames.py --batch --input_dir "outputs/raw_footage/night_clear/" --output_dir "outputs/test_images/" --fps 5
python scripts/data_ingestion/extract_frames.py --batch --input_dir "outputs/raw_footage/night_rain/" --output_dir "outputs/test_images/" --fps 5
```

### Method 3: Use the Helper Script (Easiest)

A helper script will be created to automate batch processing. Stay tuned!

---

## Metadata Creation

After extracting frames, you need to fill in metadata for quality tracking.

### Manual Metadata Entry

Open `outputs/test_images/metadata.txt` and add entries for each image:

**Format:**
```
filename,condition,time_of_day,weather,angle,notes
```

**Example:**
```
day_clear_01_00042.jpg,day_clear,day,clear,front,good_lighting_close_range_sedan
day_clear_01_00087.jpg,day_clear,day,clear,rear,slight_shadow_medium_distance
night_clear_02_00123.jpg,night_clear,night,clear,front_45,street_lamp_lighting
day_rain_01_00034.jpg,day_rain,day,rain,side_90,wet_surface_reflections
```

### Semi-Automated Metadata (Helper Script Coming)

A Python script will be created to generate template entries that you can review and edit.

---

## Quality Check

Before considering the task complete, review extracted images:

### Checklist
- [ ] Total image count: 50-100 images
- [ ] Condition diversity:
  - [ ] Day + Clear: 10-15 images
  - [ ] Day + Rain: 5-10 images
  - [ ] Night + Clear: 10-15 images
  - [ ] Night + Rain: 5-10 images
- [ ] Angle diversity:
  - [ ] Front angles: ~30%
  - [ ] Rear angles: ~30%
  - [ ] Side/angled: ~40%
- [ ] Image quality:
  - [ ] License plates visible (even if small)
  - [ ] Not excessively blurry
  - [ ] Adequate lighting
  - [ ] No corrupted files

### Remove Bad Images

If you find images that don't meet quality standards:
1. Delete the image file
2. Remove the corresponding line from metadata.txt

### Target Quality Over Quantity

✅ Better to have 50 high-quality diverse images  
❌ Than 100 low-quality or repetitive images

---

## Troubleshooting

### "Video file not found" error
- Check file paths (use absolute paths or run from project root)
- Ensure video files are in the correct directories
- Check file extensions (.mp4, .avi, .mov)

### "Failed to open video" error
- Video file may be corrupted
- Try re-encoding with: `ffmpeg -i input.mp4 -c copy output.mp4`
- Ensure OpenCV supports the codec

### "No frames extracted" error
- Video may be too short
- Check FPS setting (try lower FPS like 2-3)
- Verify video plays correctly in media player

### "Permission denied" error
- Close any programs that might have files open
- Run PowerShell as Administrator
- Check folder permissions

---

## Completion Checklist

Before marking Task 5 as complete:

- [ ] Directory structure created (`outputs/raw_footage/`, `outputs/test_images/`)
- [ ] Videos recorded/downloaded and organized by condition
- [ ] Frames extracted (50-100 images in `outputs/test_images/`)
- [ ] Metadata file completed (`outputs/test_images/metadata.txt`)
- [ ] Quality check performed
- [ ] Bad images removed
- [ ] Dataset diversity verified (conditions, angles, lighting)
- [ ] If using public footage: Attribution documented in `sources.txt`

---

## Next Steps (Task 6+)

After completing this task, you'll:
1. Set up Label Studio for annotation (Task 7)
2. Annotate bounding boxes around license plates
3. Train/fine-tune detection model
4. Evaluate detection performance

---

## Quick Reference Card

```
┌─────────────────────────────────────────┐
│ GTA V In-Game Controls                  │
├─────────────────────────────────────────┤
│ F1               - Toggle HUD on/off    │
│ M                - Open map             │
│ Right Mouse Hold - Free look camera     │
│ Mouse Movement   - Aim camera           │
│ V                - Cinematic camera     │
│ ~                - Console (for cheats) │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ OBS Recording Controls                  │
├─────────────────────────────────────────┤
│ F9 - Start/Stop Recording (toggle)      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ GTA V Console Cheats (Press ~ first)    │
├─────────────────────────────────────────┤
│ TIME 12     - Set time to noon          │
│ TIME 22     - Set time to 10 PM         │
│ MAKEITRAIN  - Start rain                │
│ CLEAR       - Clear weather             │
│ FOGGY       - Foggy conditions          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Target Dataset Composition              │
├─────────────────────────────────────────┤
│ Day + Clear:   10-15 images             │
│ Day + Rain:    5-10 images              │
│ Night + Clear: 10-15 images             │
│ Night + Rain:  5-10 images              │
│ Various angles: Mix of front/rear/side  │
│ Total target:  50-100 high-quality      │
└─────────────────────────────────────────┘
```

---

## Need Help?

- Review detailed strategy: `docs/data_collection_strategy.md`
- Check frame extraction script: `scripts/data_ingestion/extract_frames.py`
- Project structure: `docs/project_structure.md`

**Ready to start? Follow the guide above and begin collecting data! Good luck! 🎮📹**
