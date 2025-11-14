// Preload script for security
// This file runs in a sandboxed context with access to Node.js APIs
// but isolated from the renderer process

const { contextBridge } = require('electron');

// Expose protected methods that allow the renderer process to use
// Node.js functionality safely
contextBridge.exposeInMainWorld('electronAPI', {
  // Add any APIs you need to expose to the renderer
  // For now, we don't need any since Streamlit handles everything
});
