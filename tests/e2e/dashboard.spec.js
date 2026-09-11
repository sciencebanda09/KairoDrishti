import { expect, test } from '@playwright/test'

const mission = {
  mission_id: 'SAR-E2E-001',
  home: { lat: 30.362, lon: 78.082, altitude: 1200 },
  cells: [{ id: 'RIDGE-01', lat: 30.363, lon: 78.083, altitude: 1180, priority: 1, likelihood: .9 }],
  waypoints: [
    { sequence: 1, kind: 'search', lat: 30.363, lon: 78.083, altitude: 1180, cell_id: 'RIDGE-01', rationale: 'test route' },
    { sequence: 2, kind: 'return_home', lat: 30.362, lon: 78.082, altitude: 1200, rationale: 'return' },
  ],
  detections: [],
  no_fly: [{ id: 'NFZ-A', lat: 30.3665, lon: 78.0835, radius: .0007, height: 70 }],
  truth: { lat: 30.366, lon: 78.084, confidence: .84 },
  coordinate_system: 'WGS84',
}

async function mockApi(page) {
  await page.route('**/api/mission', async route => route.fulfill({ json: mission }))
  await page.route('**/api/terrain', async route => route.fulfill({ json: { source: 'synthetic', grid: null, bounds: null } }))
  await page.route('**/api/readiness', async route => route.fulfill({ json: { detector_mode: 'opencv', model_loaded: false, terrain_detector_ready: false, planner_ready: false, opensky_enabled: false } }))
  await page.route('**/api/airspace', async route => route.fulfill({ json: { source: 'disabled', available: false, aircraft: [] } }))
  await page.route('**/api/frames/detect', async route => route.fulfill({ json: mission }))
  await page.route('**/api/change-detection', async route => route.fulfill({ json: { ...mission, change_registration: { registered: true, score: .91, method: 'orb-ransac-homography' } } }))
  await page.route('**/api/terrain/load', async route => route.fulfill({ json: {
    terrain: { source: 'dem', format: 'srtm', resolution_m: [30, 30], grid: [[100]], bounds: { south: 30, west: 78, north: 30.01, east: 78.01 } },
    mission,
  } }))
  await page.route('**/api/satellite/context', async route => route.fulfill({ json: {
    context: { products: ['rgb', 'false_colour', 'ndvi'], quality: 'ready', resolution_m: 10 },
    mission: { ...mission, satellite_context: { products: ['rgb', 'false_colour', 'ndvi'], quality: 'ready' } },
  } }))
  await page.route('**/api/mission/export*', async route => route.fulfill({
    status: 200,
    contentType: 'text/csv',
    headers: { 'Content-Disposition': 'attachment; filename="kairodristi-field-packet.csv"' },
    body: 'mission_id,record_type\nSAR-E2E-001,waypoint\n',
  }))
}

test.beforeEach(async ({ page }) => {
  await mockApi(page)
  await page.goto('/')
  await expect(page.locator('#missionState')).toHaveText('LIVE')
})

test('synthetic flight advances through route coordinates', async ({ page }) => {
  const first = await page.locator('#cameraReadout').textContent()
  await expect.poll(async () => page.locator('#cameraReadout').textContent(), { timeout: 5_000 }).not.toBe(first)
  await expect(page.locator('#cameraReadout')).toContainText('AIRBORNE')
})

test('analysis bands change the active camera presentation', async ({ page }) => {
  await page.getByRole('tab', { name: 'THERMAL' }).click()
  await expect(page.locator('.camera-card')).toHaveClass(/band-thermal/)
  await expect(page.locator('#cameraBandOverlay')).toContainText('HEAT SIGNATURE')
  await page.getByRole('tab', { name: 'CHANGE' }).click()
  await expect(page.locator('.camera-card')).toHaveClass(/band-change/)
  await expect(page.locator('#cameraBandOverlay')).toHaveText('SUCCESSIVE PASS MODE')
  await expect(page.locator('#loadChange')).toBeVisible()
})

test('frame and successive-pass uploads call their workflows', async ({ page }) => {
  await page.getByRole('tab', { name: 'THERMAL' }).click()
  await page.locator('#frameInput').setInputFiles({ name: 'thermal.png', mimeType: 'image/png', buffer: Buffer.from('fixture') })
  await expect(page.locator('#sensorState')).toHaveText('THERMAL ANALYSIS COMPLETE')
  await page.getByRole('tab', { name: 'CHANGE' }).click()
  await page.locator('#previousChangeInput').setInputFiles({ name: 'before.png', mimeType: 'image/png', buffer: Buffer.from('before') })
  await page.locator('#currentChangeInput').setInputFiles({ name: 'after.png', mimeType: 'image/png', buffer: Buffer.from('after') })
  await expect(page.locator('#sensorState')).toContainText('REGISTERED')
})

test('field packet export supports the selected format', async ({ page }) => {
  await page.locator('#exportFormat').selectOption('csv')
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'EXPORT FIELD PACKET' }).click()
  await expect((await download).suggestedFilename()).toBe('kairodristi-field-packet.csv')
})

test('local DEM and satellite context uploads update source status', async ({ page }) => {
  await page.locator('#demInput').setInputFiles({ name: 'N30E078.hgt', mimeType: 'application/octet-stream', buffer: Buffer.from('dem') })
  await expect(page.locator('#dataSourceStatus')).toContainText('SRTM 30M')
  await page.locator('#satelliteInput').setInputFiles({ name: 'sentinel-stack.tif', mimeType: 'image/tiff', buffer: Buffer.from('raster') })
  await expect(page.locator('#dataSourceStatus')).toContainText('RGB/FALSE_COLOUR/NDVI READY')
})
