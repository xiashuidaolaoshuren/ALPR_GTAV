const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const waitOn = require('wait-on');

let mainWindow;
let streamlitProcess;

// Determine if running in development or production
const isDev = !app.isPackaged;

// Get resource paths
function getResourcePath(relativePath) {
  if (isDev) {
    // In development, use paths relative to project root
    return path.join(__dirname, '../..', relativePath);
  } else {
    // In production, use resources path
    return path.join(process.resourcesPath, relativePath);
  }
}

// Start Streamlit server
function startStreamlit() {
  return new Promise((resolve, reject) => {
    const pythonPath = isDev
      ? path.join(__dirname, '../..', '.venv', 'Scripts', 'python.exe')
      : path.join(process.resourcesPath, 'python', 'Scripts', 'python.exe');

    const streamlitScript = getResourcePath(path.join('gui', 'app.py'));
    
    console.log('Starting Streamlit...');
    console.log('Python:', pythonPath);
    console.log('Script:', streamlitScript);

    // Spawn Streamlit process
    streamlitProcess = spawn(pythonPath, [
      '-m', 'streamlit', 'run',
      streamlitScript,
      '--server.port=8501',
      '--server.headless=true',
      '--browser.gatherUsageStats=false'
    ], {
      cwd: getResourcePath(''),
      env: {
        ...process.env,
        PYTHONPATH: getResourcePath('')
      }
    });

    streamlitProcess.stdout.on('data', (data) => {
      console.log(`Streamlit: ${data}`);
    });

    streamlitProcess.stderr.on('data', (data) => {
      console.error(`Streamlit Error: ${data}`);
    });

    streamlitProcess.on('error', (error) => {
      console.error('Failed to start Streamlit:', error);
      reject(error);
    });

    streamlitProcess.on('close', (code) => {
      console.log(`Streamlit process exited with code ${code}`);
    });

    // Wait for Streamlit to be ready
    const opts = {
      resources: ['http://localhost:8501'],
      timeout: 30000,
      interval: 500
    };

    waitOn(opts)
      .then(() => {
        console.log('Streamlit is ready!');
        resolve();
      })
      .catch((err) => {
        console.error('Error waiting for Streamlit:', err);
        reject(err);
      });
  });
}

// Create main window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'build', 'icon.ico')
  });

  // Load Streamlit app
  mainWindow.loadURL('http://localhost:8501');

  // Open DevTools in development
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App lifecycle
app.whenReady().then(async () => {
  try {
    await startStreamlit();
    createWindow();
  } catch (error) {
    console.error('Failed to start application:', error);
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  // Kill Streamlit process
  if (streamlitProcess) {
    console.log('Terminating Streamlit process...');
    streamlitProcess.kill();
  }
});
