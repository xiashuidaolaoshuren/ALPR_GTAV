# GTA V License Plate Data Capture Guide
## Step-by-Step Manual Collection Instructions

---

## 🎯 Goal
Capture 100-200 high-quality images of license plates from GTA V vehicles for training our ALPR system.

## ⏱️ Time Required
**Approximately 4-5 hours** split across 3 sessions

---

## 📋 Before You Start

### Required Software:
1. **GTA V** (PC version, installed and updated)
2. **OBS Studio** (download from https://obsproject.com/) - FREE
3. **At least 50 GB free disk space**

### Optional But Helpful:
- Second monitor to monitor recording status
- Notepad for tracking session notes

---

## 🎬 Part 1: Setting Up OBS Studio (15 minutes)

### Step 1: Install OBS Studio
1. Download OBS from https://obsproject.com/download
2. Run installer (click Next → Next → Install)
3. Launch OBS Studio

### Step 2: Configure OBS for GTA V Recording

**2.1 Add Game Capture Source:**
1. In OBS main window, click **"+"** under "Sources" box
2. Select **"Game Capture"**
3. Name it "GTA V" and click OK
4. In properties:
   - Mode: **"Capture specific window"**
   - Window: **Select "Grand Theft Auto V"** (you'll need to launch GTA V first)
   - Check **"Capture mouse cursor"** → OFF
   - Click **OK**

**2.2 Configure Output Settings:**
1. Click **"Settings"** (bottom right)
2. Go to **"Output"** tab
3. Change to **"Advanced"** mode (dropdown at top)
4. **Recording** sub-tab settings:
   ```
   Recording Format: mp4
   Encoder: NVIDIA NVENC H.264 (if you have NVIDIA GPU)
           OR x264 (if AMD/Intel GPU)
   Rate Control: CBR
   Bitrate: 15000 Kbps
   Keyframe Interval: 2
   Preset: Quality (NVENC) OR veryfast (x264)
   ```
5. Go to **"Video"** tab:
   ```
   Base Resolution: 1920x1080
   Output Resolution: 1920x1080
   FPS: 60
   ```
6. Click **"Apply"** then **"OK"**

**2.3 Set Recording Hotkey:**
1. Go back to **Settings → Hotkeys**
2. Find **"Start Recording"** → Set to **F9**
3. Find **"Stop Recording"** → Set to **F9** (same key toggles)
4. Click **Apply** and **OK**

**2.4 Choose Save Location:**
1. Settings → Output → Recording
2. **Recording Path** → Browse to folder with plenty of space
   - Example: `D:\GTA5_Recordings\`
3. Create folder structure:
   ```
   GTA5_Recordings/
   ├── day_clear/
   ├── night_clear/
   ├── day_rain/
   └── night_rain/
   ```

✅ **OBS is now configured!** Keep OBS running in background.

---

## 🎮 Part 2: Setting Up GTA V (10 minutes)

### Step 1: Launch GTA V
1. Start GTA V through Steam/Epic/Rockstar Launcher
2. Load into **Story Mode** (Michael, Franklin, or Trevor)
3. Wait until fully loaded into game world

### Step 2: Configure Graphics Settings
1. Press **Esc** → **Settings** → **Graphics**
2. Set these for best plate readability:
   ```
   Resolution: 1920x1080 (or your native resolution)
   Texture Quality: High
   Anti-Aliasing: FXAA + MSAA x2
   VSync: OFF (for better performance)
   ```
3. **Important:** Press **F1** to hide HUD (minimap, wanted stars, etc.)
   - This gives cleaner footage for processing

### Step 3: Choose Starting Location
**Recommended locations for beginners:**

**Option A: Los Santos International Airport**
- Open area, good lighting
- Press **M** to open map
- Set waypoint to airport (icon: plane)
- Drive or taxi there

**Option B: Del Perro Freeway**
- Highway with consistent traffic
- Press **M**, find freeway near beach
- Park on emergency lane shoulder

---

## 📹 Part 3: Recording Session 1 - Day / Clear Weather (2 hours)

### Target: 50 vehicles captured

### Step-by-Step Capture Process:

**1. Start Recording**
- Make sure GTA V window is active
- Press **F9** (OBS starts recording)
- You'll see a red dot in OBS window (if visible)

**2. Find Your First Vehicle**
- Walk to parked car OR wait for traffic to approach
- Position yourself 5-10 meters away from vehicle

**3. Frame the Shot**
- **Hold Right Mouse Button** to enter "free look" mode
- **Move mouse** to aim camera at license plate
- **Goal:** Plate should be clearly visible in center of screen
- **Don't zoom too close** - keep some context around plate

**4. Capture Multiple Angles**

**FRONT VIEW (Most Important):**
- Stand directly in front of vehicle
- Frame: Hood + windshield + front plate visible
- Hold position for **3-5 seconds**

**REAR VIEW:**
- Walk around to back of vehicle
- Stand directly behind
- Frame: Trunk + rear window + rear plate visible
- Hold position for **3-5 seconds**

**SIDE ANGLE (45°):**
- Stand at 45-degree angle from front corner
- Frame: Front quarter of vehicle + plate visible
- Hold position for **3-5 seconds**

**5. Move to Next Vehicle**
- Walk to next parked car or wait for next traffic vehicle
- Repeat steps 3-4

**6. Tips for Good Footage:**
- **Keep camera steady** - don't shake or move quickly
- **Wait for vehicle to stop** if capturing moving traffic
- **Check lighting** - avoid standing with sun directly behind you
- **Capture variety** - different car types, colors

**7. End of Session**
- After 30-40 minutes, press **F9** to stop recording
- Take a break!
- Repeat for another 30-40 minutes

**Quality Check:**
- Review footage in VLC or Windows Media Player
- Can you read plate numbers? ✓
- Is plate in focus? ✓
- Good lighting? ✓

---

## 🌃 Part 4: Recording Session 2 - Night / Clear Weather (1.5 hours)

### Target: 30 vehicles captured

### Change Time to Night:
**Method 1: Cheat Code (Fast):**
1. Press **~** key (above Tab) to open console
2. Type: `TIME 22` (sets time to 10 PM)
3. Press Enter

**Method 2: Wait In-Game (Realistic):**
- Find a bench or chair
- Hold **Enter** to sit
- Press **=** key repeatedly to advance time
- Watch clock until 10 PM - 2 AM

### Night Capture Tips:
- **Stay in well-lit areas:**
  - Gas stations (bright lights)
  - Main streets (street lamps)
  - Commercial districts
  - Avoid dark alleys
  
- **Use vehicle headlights:**
  - Park car with headlights shining on target vehicle
  - Creates good lighting on plates

- **Capture near street lamps:**
  - Position yourself so lamp lights plate area
  
**Follow same capture process as Day session:**
1. Press F9 to start recording
2. Find vehicle
3. Capture Front, Rear, 45° angles (3-5 seconds each)
4. Move to next vehicle
5. Stop recording after 30-40 minutes

---

## 🌧️ Part 5: Recording Session 3 - Rain & Varied Conditions (1 hour)

### Target: 20-30 vehicles captured

### Change Weather to Rain:
**Method 1: Cheat Code:**
1. Press **~** key
2. Type: `MAKEITRAIN`
3. Press Enter

**Method 2: Natural Weather:**
- Wait for rain to start naturally (random)
- Can take 10-30 minutes in-game time

### Rain Capture Challenges:
- **Wet surfaces create reflections** - good for testing model robustness
- **Reduced visibility** - focus on close-range vehicles (5-10m)
- **Capture variety:**
  - Vehicles with water droplets on plates
  - Reflections on wet pavement
  - Rain-streaked plates

### Bonus Conditions to Capture:
**Fog:**
- Console command: `FOGGY` (creates misty atmosphere)

**Overcast:**
- Wait for cloudy weather naturally
- Softer lighting, less shadows

---

## 💾 Part 6: Organizing Your Recordings (30 minutes)

### Step 1: Review Recordings
1. Open your recording folder (e.g., `D:\GTA5_Recordings\`)
2. You should have 6-10 video files (*.mp4)
3. Watch each briefly - note which are good quality

### Step 2: Rename Files Descriptively
**Naming Pattern:** `{condition}_{location}_{number}.mp4`

**Examples:**
- `day_clear_01.mp4`
- `night_clear_01.mp4`
- `day_rain_01.mp4`

**How to rename:**
1. Right-click file → **Rename**
2. Type new name
3. Press Enter

### Step 3: Move to Organized Folders
- Move files to appropriate subfolders:
  ```
  GTA5_Recordings/
  ├── day_clear/
  │   ├── day_clear_01.mp4
  │   └── day_clear_02.mp4
  ├── night_clear/
  │   └── night_clear_01.mp4
  └── day_rain/
      └── day_rain_01.mp4
  ```

### Step 4: Create Metadata Notes
- Create a text file: `metadata.txt`
- For each video, note:
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

## 🖼️ Part 7: Extracting Frames (Automated - 30 minutes)

**You'll use our Python script to extract individual images from videos.**

### Step 1: Activate Virtual Environment
Open PowerShell in project directory:
```powershell
.\.venv\Scripts\activate
```

### Step 2: Run Extraction Script
```powershell
python scripts/extract_frames.py --input "D:\GTA5_Recordings\day_clear\day_clear_01.mp4" --output "datasets\lpr\train\images" --fps 5
```

**What this does:**
- Extracts 1 frame every 12 frames (5 FPS from 60 FPS source)
- Saves as JPEG images in `datasets/lpr/train/images/`
- Names files automatically: `day_clear_01_00001.jpg`, `day_clear_01_00002.jpg`, etc.

### Step 3: Repeat for All Videos
**Batch process** (run this to extract from all videos):
```powershell
python scripts/extract_frames.py --batch --input_dir "D:\GTA5_Recordings" --output_dir "datasets\lpr\train\images" --fps 5
```

**Result:** You should have 500-1000 individual images extracted!

---

## ✅ Quality Check Checklist

### Review Extracted Images:
Open `datasets/lpr/train/images/` folder and check:

**GOOD Images (Keep):**
- ✅ License plate clearly visible
- ✅ Text readable (at least 4-5 characters)
- ✅ Plate at least 20x50 pixels in size
- ✅ Minimal blur
- ✅ Good lighting contrast

**BAD Images (Delete):**
- ❌ Extremely blurry
- ❌ Plate too small (< 15x30 pixels)
- ❌ More than 50% of plate hidden
- ❌ Completely dark or overexposed
- ❌ Duplicate frames (back-to-back identical images)

**Manual cleaning:**
- Open folder in File Explorer
- Use **Large Icons** view to preview
- Delete bad images by selecting and pressing **Delete**

---

## 🏆 Success Criteria

### You're done when you have:
- ✅ **100-200 good quality images** in `datasets/lpr/train/images/`
- ✅ **Images from multiple conditions:**
  - At least 50 from Day/Clear
  - At least 20 from Night/Clear
  - At least 15 from Rain/Any angle
  - At least 15 from various angles (45°, rear)
- ✅ **Organized metadata notes** documenting what you captured

---

## 🆘 Troubleshooting

### Problem: OBS recording is laggy or choppy
**Solution:**
- Lower GTA V graphics settings (Medium instead of High)
- Reduce OBS bitrate to 10000 Kbps
- Close other programs while recording

### Problem: Can't see license plates clearly in recordings
**Solution:**
- Get closer to vehicles (5-8 meters)
- Ensure GTA V texture quality is set to High
- Avoid extreme zoom in/out

### Problem: GTA V crashes or freezes
**Solution:**
- Verify game files through launcher
- Update graphics drivers
- Restart game and OBS

### Problem: Recording hotkey (F9) doesn't work
**Solution:**
- Make sure GTA V window is active (click on game)
- Check OBS Settings → Hotkeys - verify F9 is set
- Try different key if F9 conflicts with game controls

### Problem: Extracted images are blurry
**Solution:**
- Keep camera steady during recording (don't move rapidly)
- Capture vehicles when stopped or slow-moving
- Ensure plate is at least 30-40 pixels wide in frame

---

## 📧 Need Help?

If you run into issues not covered here:
1. Check project README.md for additional troubleshooting
2. Review the full strategy document: `docs/data_collection_strategy.md`
3. Open a GitHub issue with details of your problem

---

## 🎉 Next Steps

After completing data collection:
1. **Annotation**: Use Label Studio to draw bounding boxes around plates
2. **Training**: Fine-tune YOLOv8 model on your dataset
3. **Evaluation**: Test detection accuracy

**Congratulations on completing data collection! 🎊**

---

**Quick Reference Card:**

```
┌─────────────────────────────────────────┐
│ GTA V Controls                          │
├─────────────────────────────────────────┤
│ F1  - Toggle HUD on/off                 │
│ M   - Open map                          │
│ Right Mouse - Free look (hold)          │
│ Mouse Wheel - Zoom (if modded)          │
│ ~   - Console (for cheats)              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ OBS Controls                            │
├─────────────────────────────────────────┤
│ F9  - Start/Stop Recording              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Useful GTA V Cheats (Press ~ first)    │
├─────────────────────────────────────────┤
│ TIME 12  - Set to noon                  │
│ TIME 22  - Set to 10 PM                 │
│ MAKEITRAIN - Start rain                 │
│ CLEAR - Clear weather                   │
│ FOGGY - Foggy weather                   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Target Captures                         │
├─────────────────────────────────────────┤
│ Session 1: 50 vehicles (day/clear)     │
│ Session 2: 30 vehicles (night/clear)   │
│ Session 3: 20 vehicles (rain/varied)   │
│ Total: 100+ vehicles                    │
└─────────────────────────────────────────┘
```

---

**Document Version:** 1.0  
**Difficulty:** Beginner-Friendly  
**Estimated Completion Time:** 4-5 hours  
**Last Updated:** Week 1, Task 4
