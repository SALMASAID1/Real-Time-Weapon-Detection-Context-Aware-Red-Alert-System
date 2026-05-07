import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy API and WebSocket requests to the FastAPI backend during development.
    // This avoids CORS issues entirely — the browser thinks all requests are same-origin.
    proxy: {
      '/api':      { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':       { target: 'ws://localhost:8000',   changeOrigin: true, ws: true },
      '/health':   { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
