import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
    ...devices['Desktop Chrome'],
    launchOptions: { args: ['--use-gl=swiftshader'] },
  },
  webServer: {
    command: 'C:/Users/kumar/AppData/Local/Programs/Python/Python311/python.exe -m app.server --port 4173 --detector opencv',
    cwd: '.',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: true,
    timeout: 30_000,
  },
})
