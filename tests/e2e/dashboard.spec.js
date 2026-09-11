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
  let swarmPasses = 0
  const swarmDrones = [1, 2, 3, 4].map((index) => ({ drone_id: `DRONE-${String(index).padStart(2, '0')}`, status: 'searching', battery_percent: 80 - index * 4, lat: 30.362 + index * .0002, lon: 78.082 + index * .0002, assigned_sector_id: `SECTOR-${String.fromCharCode(64 + index)}`, communication_ok: true }))
  const swarmPacket = (initialized = true) => ({ mission_id: 'SWARM-E2E-001', offline: true, mode: 'swarm', initialized, simulation_only: true, planning_only: true, framing: 'SIMULATION / PLANNING ONLY', tracking_mode: 'nearest_decay', home: mission.home, cells: [1, 2, 3, 4].map((index) => ({ id: `SECTOR-${String.fromCharCode(64 + index)}`, lat: 30.363 + index * .001, lon: 78.083 + index * .001, priority: index, likelihood: .8 })), drones: initialized ? swarmDrones : [], sectors: initialized ? swarmDrones.map((drone) => ({ cell_id: drone.assigned_sector_id, owning_drone_id: drone.drone_id, owner_status: 'searching', coverage_fraction: .25, priority: 1, likelihood: .8 })) : [], detections: swarmPasses ? [{ id: 'simulated-person', track_id: 'TRK-001', lat: 30.364, lon: 78.084, confidence: .72, label: 'simulated candidate', band: 'rgb' }] : [], tracks: swarmPasses ? [{ track_id: 'TRK-001', lat: 30.364, lon: 78.084, confidence: .72, detection_count: swarmPasses, tracking_mode: 'nearest_decay' }] : [], waypoints: Object.fromEntries((initialized ? swarmDrones : []).map((drone) => [drone.drone_id, [{ sequence: 1, kind: 'search', lat: 30.363, lon: 78.083, altitude: 1180, cell_id: drone.assigned_sector_id }, { sequence: 2, kind: 'return_home', lat: mission.home.lat, lon: mission.home.lon, altitude: mission.home.altitude }]])), swarm_coverage_fraction: .25, no_fly: mission.no_fly })
  await page.route('**/api/mission', async route => route.fulfill({ json: mission }))
  await page.route('**/api/swarm**', async route => {
    if (route.request().method() === 'POST' && route.request().url().endsWith('/api/swarm/init')) return route.fulfill({ json: swarmPacket(true) })
    if (route.request().method() === 'POST' && route.request().url().endsWith('/api/swarm/telemetry')) { swarmPasses += 1; return route.fulfill({ json: swarmPacket(true) }) }
    return route.fulfill({ json: swarmPacket(false) })
  })
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

test('swarm control shows four drones, hatch, and an associated track after two passes', async ({ page }) => {
  await page.getByRole('button', { name: 'ENTER SWARM MODE' }).click()
  await expect(page.locator('#swarmControlPanel')).toBeVisible()
  await page.getByRole('button', { name: 'INIT SWARM' }).click()
  await expect(page.locator('.swarm-row')).toHaveCount(4)
  await expect(page.locator('#noFlyHatch')).toBeAttached()
  await page.getByRole('button', { name: 'SIMULATE PASS' }).click()
  await page.getByRole('button', { name: 'SIMULATE PASS' }).click()
  await expect(page.locator('.track-row')).toHaveCount(1)
  await expect(page.locator('.track-row')).toContainText('TRK-001')
  await expect(page.locator('#airspaceBadge')).toHaveText('SWARM · PLANNING ONLY')
})
