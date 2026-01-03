import { defineConfig } from 'vite'

export default defineConfig({
  // Use relative base for GitHub Pages deployment
  base: './',
  publicDir: 'public',
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['wasm_exec.js']
  },
  build: {
    target: 'esnext',
    // Ensure assets are correctly referenced
    assetsDir: 'assets',
  },
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
})
