import { defineConfig } from 'vite'

export default defineConfig({
  root: 'frontend',
  publicDir: 'public',
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
  build: { outDir: '../app/dist', emptyOutDir: true },
})
