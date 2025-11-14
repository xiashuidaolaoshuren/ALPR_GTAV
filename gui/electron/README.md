# GTA V ALPR Desktop Application

This directory contains the Electron wrapper for the GTA V ALPR System, packaging it as a standalone desktop application.

## Structure

```
gui/electron/
├── package.json          # Electron configuration and dependencies
├── main.js               # Electron main process (app entry point)
├── preload.js            # Security context bridge
├── build/                # Build resources
│   └── icon.ico          # Application icon
└── dist/                 # Output directory (created after build)
```

## Development

### Prerequisites

- Node.js 18+ and npm
- Python virtual environment with all dependencies (in `.venv`)
- All project models and configs in place

### Run in Development Mode

```bash
cd gui/electron
npm install
npm start
```

This will:
1. Start Streamlit server on port 8501
2. Open Electron window displaying the GUI
3. Enable DevTools for debugging

## Building

### Quick Build

Use the automated build script from project root:

```bash
scripts\build_desktop_app.bat
```

### Manual Build

```bash
cd gui/electron
npm install
npm run build
```

The installer will be created in `gui/electron/dist/`

## Build Configuration

The build is configured in `package.json` under the `build` section:

- **Target**: NSIS installer for Windows
- **Bundled Resources**:
  - Python virtual environment (`.venv`)
  - YOLOv8 models (`models/`)
  - Configuration files (`configs/`)
  - GUI source code (`gui/`)
  - Core ALPR library (`src/`)

## How It Works

1. **main.js**: Electron's main process
   - Spawns Streamlit server as subprocess
   - Waits for server to be ready (using `wait-on`)
   - Creates BrowserWindow pointing to `http://localhost:8501`
   - Handles cleanup on exit

2. **Resource Path Resolution**:
   - Development: Uses relative paths from project root
   - Production: Uses `process.resourcesPath` for bundled files

3. **Streamlit Integration**:
   - Runs headless (no browser auto-open)
   - Communicates via localhost websocket
   - Fully functional with all GUI features

## Troubleshooting

### Build Issues

**Error: Python not found in bundled app**
- Check `extraResources` in `package.json`
- Verify `.venv` exists and is complete

**Error: Models not loading**
- Ensure models are in `models/` directory
- Check path resolution in `main.js`

**Large bundle size (>1 GB)**
- This is normal due to Python + dependencies
- Consider excluding dev dependencies
- Use asar compression (enabled by default)

### Runtime Issues

**Streamlit won't start**
- Check console logs in DevTools
- Verify Python path in `main.js`
- Ensure all dependencies are in `.venv`

**Port 8501 already in use**
- Close any other Streamlit instances
- Change port in `main.js` if needed

**Video processing not working**
- Verify FFmpeg is in PATH (for H.264 conversion)
- Check model files are accessible

## Distribution

The built installer (`GTA V ALPR System Setup.exe`) can be distributed as a standalone package:

- No Python installation required
- No manual dependency installation
- Self-contained with all models and libraries
- Standard Windows installer experience

## Icon Customization

Replace `build/icon.ico` with your custom icon:
- Recommended sizes: 16x16, 32x32, 48x48, 256x256
- Format: ICO file with multiple resolutions
- Tools: ImageMagick, GIMP, or online converters

## Security

- `contextIsolation`: Enabled (renderer process isolated)
- `nodeIntegration`: Disabled (no direct Node.js in renderer)
- `preload.js`: Sandboxed bridge for safe API exposure

## Performance

- **Startup Time**: ~5-10 seconds (Streamlit initialization)
- **Bundle Size**: ~700 MB - 1 GB
- **Memory Usage**: ~500-800 MB (Streamlit + models loaded)
- **Install Size**: ~1-1.5 GB

## Future Enhancements

- [ ] Add loading screen during Streamlit startup
- [ ] Bundle FFmpeg for guaranteed H.264 support
- [ ] Implement auto-update mechanism
- [ ] Add crash reporting
- [ ] Optimize bundle size (exclude unnecessary files)
- [ ] macOS and Linux builds
