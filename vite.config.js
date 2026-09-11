import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

const frontendRoot = fileURLToPath(new URL('./frontend', import.meta.url))

export default defineConfig({
  root: frontendRoot,
  publicDir: false,
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
  build: { outDir: fileURLToPath(new URL('./app/dist', import.meta.url)), emptyOutDir: true },
})
