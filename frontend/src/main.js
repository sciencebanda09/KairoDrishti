import * as THREE from 'three'
import './styles.css'

const app = document.querySelector('#app')
app.innerHTML = `
  <header class="topbar">
    <div class="wordmark">KAIRODRISHTI <span>/ FIELD CONSOLE</span></div>
    <div class="topmeta"><span class="live-dot"></span><span id="missionState">STANDBY</span><span class="divider"></span><span id="missionId">SAR-DEMO-001</span></div>
    <div class="topstats"><div><small>BATTERY</small><strong id="battery">100%</strong></div><div><small>COVERAGE</small><strong id="coverage">0%</strong></div><div><small>ELAPSED</small><strong id="elapsed">00:00</strong></div></div>
  </header>
  <main class="workspace">
    <section class="viewport-panel">
      <div id="viewport"></div>
      <div class="viewport-toolbar"><span class="section-kicker">OPERATOR VIEW</span><span class="coordinate-badge">LOCAL PROJECTION · WGS84 OUTPUT</span></div>
      <div class="view-help">DRAG orbit&nbsp;&nbsp; · &nbsp;&nbsp;SCROLL zoom&nbsp;&nbsp; · &nbsp;&nbsp;R reset camera</div>
      <div class="scene-legend"><span><i class="swatch high"></i>high</span><span><i class="swatch medium"></i>medium</span><span><i class="swatch low"></i>low</span></div>
      <div class="scene-controls"><button id="resetView">RESET VIEW</button><button id="toggleHeat" class="selected">PRIORITY</button><button id="toggleCoverage" class="selected">COVERAGE</button><button id="toggleNfz" class="selected">AIRSPACE</button></div>
    </section>
    <aside class="right-rail">
      <section class="camera-card"><div class="card-head"><span class="section-kicker">DRONE CAMERA</span><span id="sensorMode">RGB / THERMAL</span></div><div id="cameraViewport"></div><div class="camera-readout" id="cameraReadout">FRAME 0000 · ALT 35 M</div></section>
      <section class="mission-card"><div class="card-head"><span class="section-kicker">MISSION STATUS</span><span class="status-chip" id="statusChip">READY</span></div><h1 id="action">TAKEOFF / CLIMB</h1><p id="actionDetail">Acquiring search altitude and first sector.</p><div class="metric-grid"><div><small>ALTITUDE</small><strong id="altitude">35 m</strong></div><div><small>GROUND SPEED</small><strong id="speed">0.0 m/s</strong></div><div><small>ACTIVE LEG</small><strong id="activeLeg">HOME → RIDGE</strong></div><div><small>RESERVE</small><strong id="reserve">18 min</strong></div></div><div class="battery-line"><span>BATTERY RESERVE</span><span id="batteryLine">100%</span></div><div class="meter"><i id="batteryMeter"></i></div></section>
      <section class="event-card"><div class="card-head"><span class="section-kicker">EVENT LOG</span><button class="quiet-button" id="triggerDetection">SIMULATE DETECTION</button></div><div id="eventLog" class="event-log"><div class="event"><time>00:00</time><span class="event-marker neutral"></span><p><b>Mission initialized</b><small>Planner route loaded · return reserve protected</small></p></div></div></section>
    </aside>
    <footer class="footer-bar"><div class="timeline"><div class="timeline-labels"><span>MISSION TIMELINE</span><span id="phaseLabel">ROUTE LOADED</span><span>RTH RESERVE</span></div><div class="timeline-track"><i id="timelineProgress"></i></div></div><div class="footer-actions"><button id="toggleMission" class="primary">START MISSION</button><button id="restartMission">RESTART</button><button id="exportPacket">EXPORT PACKET</button></div></footer>
  </main>
`

const $ = (id) => document.querySelector(`#${id}`)
const state = { data: null, route: [], running: false, time: 0, speedMultiplier: 1, detected: false, layers: { heat: true, coverage: true, nfz: true }, frame: 0 }
const scene = new THREE.Scene()
scene.background = new THREE.Color(0x1d251c)
scene.fog = new THREE.Fog(0x1d251c, 150, 480)
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 1200)
camera.position.set(155, 150, 180)
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFSoftShadowMap
renderer.outputColorSpace = THREE.SRGBColorSpace
document.querySelector('#viewport').appendChild(renderer.domElement)

const cameraScene = new THREE.Scene()
cameraScene.background = new THREE.Color(0x1d2b22)
const droneCamera = new THREE.PerspectiveCamera(62, 1, 0.1, 500)
droneCamera.position.set(0, 60, 0)
droneCamera.lookAt(0, 0, 0)
const cameraRenderer = new THREE.WebGLRenderer({ antialias: true })
cameraRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
cameraRenderer.outputColorSpace = THREE.SRGBColorSpace
document.querySelector('#cameraViewport').appendChild(cameraRenderer.domElement)

let orbitTarget = new THREE.Vector3(0, 0, 0)
let orbit = { theta: 0.7, phi: 1.05, radius: 260, down: false, x: 0, y: 0 }
const world = new THREE.Group(); scene.add(world)
const worldObjects = { priority: [], coverage: [], noFly: null, route: null, drone: null, cone: null, detection: null }

scene.add(new THREE.HemisphereLight(0xd4e1d2, 0x263127, 2.1))
const sun = new THREE.DirectionalLight(0xfff1d7, 3.2); sun.position.set(-130, 240, 120); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); scene.add(sun)
cameraScene.add(new THREE.HemisphereLight(0xb7d7c1, 0x14241b, 2.4))

function local(point) { return new THREE.Vector3((point.lon - state.data.home.lon) * 95, (point.altitude || 0) - 1160, (point.lat - state.data.home.lat) * 111) }
function terrainHeight(x, z) { return 6 + 14 * Math.sin(x * 0.035) + 11 * Math.cos(z * 0.041) + 7 * Math.sin((x + z) * 0.018) }
function terrainY(x, z) { return terrainHeight(x, z) }

function buildTerrain() {
  const size = 420, segments = 72
  const geometry = new THREE.PlaneGeometry(size, size, segments, segments)
  geometry.rotateX(-Math.PI / 2)
  const pos = geometry.attributes.position
  for (let i = 0; i < pos.count; i++) pos.setY(i, terrainY(pos.getX(i), pos.getZ(i)))
  geometry.computeVertexNormals()
  const material = new THREE.MeshStandardMaterial({ color: 0x51634b, roughness: 1, metalness: 0, flatShading: false })
  const mesh = new THREE.Mesh(geometry, material); mesh.receiveShadow = true; world.add(mesh)
  const grid = new THREE.GridHelper(420, 21, 0x9aae8f, 0x82927c); grid.position.y = -0.4; grid.material.opacity = 0.15; grid.material.transparent = true; world.add(grid)
}

function makeTrees() {
  const trunk = new THREE.CylinderGeometry(0.7, 1.1, 8, 6), crown = new THREE.ConeGeometry(5, 18, 7)
  const trunkMesh = new THREE.InstancedMesh(trunk, new THREE.MeshStandardMaterial({ color: 0x584a38 }), 110)
  const crownMesh = new THREE.InstancedMesh(crown, new THREE.MeshStandardMaterial({ color: 0x334e37, roughness: 1 }), 110)
  const dummy = new THREE.Object3D(); let n = 0
  for (let x = -190; x < 190 && n < 110; x += 23) for (let z = -185; z < 185 && n < 110; z += 29) {
    if (Math.abs(x - z) < 35 || (x > 55 && x < 145 && z > -50 && z < 50)) continue
    const y = terrainY(x, z); dummy.position.set(x, y + 4, z); dummy.scale.setScalar(0.65 + ((n * 17) % 9) / 10); dummy.updateMatrix(); trunkMesh.setMatrixAt(n, dummy.matrix); dummy.position.y = y + 15; dummy.updateMatrix(); crownMesh.setMatrixAt(n, dummy.matrix); n++
  }
  world.add(trunkMesh, crownMesh)
}

function makeLine(points, color, width = 1.5, dashed = false) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 })
  if (dashed) { material.dashSize = 7; material.gapSize = 5 }
  return dashed ? new THREE.LineDashedMaterial({ color, dashSize: 7, gapSize: 5, transparent: true, opacity: 0.7 }) && new THREE.Line(geometry, material) : new THREE.Line(geometry, material)
}

function buildDrone() {
  const group = new THREE.Group()
  const body = new THREE.Mesh(new THREE.BoxGeometry(9, 2.8, 5), new THREE.MeshStandardMaterial({ color: 0x26352f, roughness: 0.65 })); body.castShadow = true; group.add(body)
  const armMaterial = new THREE.MeshStandardMaterial({ color: 0x9da99b, roughness: 0.6 })
  for (const [x, z] of [[-8, -6], [8, -6], [-8, 6], [8, 6]]) { const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 11, 8), armMaterial); arm.rotation.z = Math.PI / 2; arm.position.set(x / 2, 0, z / 2); group.add(arm); const rotor = new THREE.Mesh(new THREE.TorusGeometry(4.3, 0.23, 6, 24), new THREE.MeshBasicMaterial({ color: 0xc7d2bb })); rotor.rotation.x = Math.PI / 2; rotor.position.set(x, 2, z); group.add(rotor) }
  const cameraBody = new THREE.Mesh(new THREE.SphereGeometry(1.7, 12, 8), new THREE.MeshStandardMaterial({ color: 0xd6b25c, metalness: 0.4 })); cameraBody.position.set(0, -2, 0); group.add(cameraBody)
  group.scale.setScalar(0.9); world.add(group); worldObjects.drone = group
  const cone = new THREE.Mesh(new THREE.ConeGeometry(30, 65, 32, 1, true), new THREE.MeshBasicMaterial({ color: 0xb8d7b1, transparent: true, opacity: 0.11, side: THREE.DoubleSide, depthWrite: false })); cone.rotation.x = Math.PI; cone.position.y = -33; group.add(cone); worldObjects.cone = cone
}

function buildScene() { buildTerrain(); makeTrees(); buildDrone() }

function buildMissionLayers() {
  state.data.cells.forEach((cell, index) => {
    const p = local(cell), color = [0xb95045, 0xb8874b, 0x6e916d, 0x8a9c70][index % 4]
    const disc = new THREE.Mesh(new THREE.CircleGeometry(30, 32), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.2, depthWrite: false })); disc.rotation.x = -Math.PI / 2; disc.position.set(p.x, terrainY(p.x, p.z) + 0.5, p.z); world.add(disc); worldObjects.priority.push(disc)
    const label = makeLabel(cell.id, color); label.position.set(p.x - 10, terrainY(p.x, p.z) + 2, p.z - 20); world.add(label)
  })
  const home = local(state.data.home); const nfz = local(state.data.no_fly[0]); const nfzShape = new THREE.Mesh(new THREE.CylinderGeometry(29, 29, 60, 32, 1, true), new THREE.MeshBasicMaterial({ color: 0xa44d45, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false })); nfzShape.position.set(nfz.x, 30, nfz.z); world.add(nfzShape); worldObjects.noFly = nfzShape
  const base = new THREE.Mesh(new THREE.CylinderGeometry(11, 14, 2, 6), new THREE.MeshStandardMaterial({ color: 0xd4bb7a })); base.position.set(home.x, terrainY(home.x, home.z) + 1, home.z); base.castShadow = true; world.add(base); world.add(makeLabel('BASE', 0xd4bb7a)).position.set(home.x - 10, terrainY(home.x, home.z) + 4, home.z)
  updateRoute()
}

function makeLabel(text, color) {
  const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 48; const ctx = canvas.getContext('2d'); ctx.font = '600 22px Arial'; ctx.fillStyle = `#${new THREE.Color(color).getHexString()}`; ctx.fillText(text, 4, 30)
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false })); sprite.scale.set(30, 5.6, 1); return sprite
}

function updateRoute() {
  if (worldObjects.route) world.remove(worldObjects.route)
  const pts = state.route.map((w) => { const p = local(w); p.y = terrainY(p.x, p.z) + 7; return p })
  worldObjects.route = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xd9c681, transparent: true, opacity: 0.8 })); world.add(worldObjects.route)
  state.route.forEach((w, i) => { const p = local(w); const marker = new THREE.Mesh(new THREE.SphereGeometry(w.kind === 'investigate' ? 4 : 2.6, 12, 8), new THREE.MeshStandardMaterial({ color: w.kind === 'return_home' ? 0xd4bb7a : w.kind === 'investigate' ? 0xc65b4c : 0xb7ca9f })); marker.position.set(p.x, terrainY(p.x, p.z) + 7, p.z); marker.castShadow = true; world.add(marker); marker.userData.routeIndex = i })
}

function getPosition() {
  const path = state.route.map(local); const seg = Math.min(Math.floor(state.time / 18), path.length - 1); const f = (state.time % 18) / 18; if (seg === path.length - 1) return path[seg].clone(); return path[seg].clone().lerp(path[seg + 1], f)
}

function moveDrone() { if (!worldObjects.drone || !state.route.length) return; const p = getPosition(); const next = local(state.route[Math.min(Math.floor(state.time / 18) + 1, state.route.length - 1)]); const y = terrainY(p.x, p.z); worldObjects.drone.position.set(p.x, y + 48, p.z); worldObjects.drone.lookAt(next.x, y + 48, next.z); droneCamera.position.set(p.x, y + 45, p.z); droneCamera.lookAt(p.x, y, p.z); }

function renderCameraFeed() { cameraRenderer.render(scene, droneCamera) }
function updateCamera() { const rect = document.querySelector('#viewport').getBoundingClientRect(); renderer.setSize(rect.width, rect.height, false); camera.aspect = rect.width / rect.height; camera.updateProjectionMatrix(); const c = document.querySelector('#cameraViewport').getBoundingClientRect(); cameraRenderer.setSize(c.width, c.height, false); droneCamera.aspect = c.width / c.height; droneCamera.updateProjectionMatrix() }
function updateOrbit() { camera.position.setFromSphericalCoords(orbit.radius, orbit.phi, orbit.theta); camera.position.add(orbitTarget); camera.lookAt(orbitTarget) }

function uiUpdate() {
  const total = state.route.length * 18 + 12, progress = Math.min(state.time / (state.route.length * 18), 1), battery = Math.max(36, 100 - state.time / total * 58), index = Math.min(Math.floor(state.time / 18), state.route.length - 1), current = state.route[index]
  const phase = state.time < 8 ? 'TAKEOFF / CLIMB' : state.detected && state.time >= 18 * (index) && state.time < 18 * (index + 1) ? 'INVESTIGATE TARGET' : state.time > state.route.length * 18 - 18 ? 'RETURN TO BASE' : 'SEARCH SECTOR'
  $('missionState').textContent = state.running ? 'LIVE' : 'PAUSED'; $('battery').textContent = `${Math.round(battery)}%`; $('coverage').textContent = `${Math.round(progress * 100)}%`; $('batteryLine').textContent = `${Math.round(battery)}%`; $('batteryMeter').style.width = `${battery}%`; $('elapsed').textContent = `00:${String(Math.floor(state.time / 1.8) % 60).padStart(2, '0')}`
  $('action').textContent = phase; $('actionDetail').textContent = phase === 'INVESTIGATE TARGET' ? 'Thermal corroboration received · route diversion active.' : phase === 'RETURN TO BASE' ? 'Protected reserve holding for landing.' : 'Following priority route across the search sector.'; $('statusChip').textContent = state.detected ? 'REPLANNED' : state.running ? 'RUNNING' : 'READY'; $('altitude').textContent = `${Math.round(48 + Math.sin(state.time * 0.1) * 3)} m`; $('speed').textContent = state.running ? `${(8 + Math.sin(state.time) * 1.5).toFixed(1)} m/s` : '0.0 m/s'; $('activeLeg').textContent = current ? `${current.cell_id || current.kind.toUpperCase()}` : 'HOME → RIDGE'; $('reserve').textContent = `${Math.max(8, Math.round(18 - state.time / 60))} min`; $('phaseLabel').textContent = phase; $('timelineProgress').style.width = `${progress * 100}%`; $('cameraReadout').textContent = `FRAME ${String(state.frame++).padStart(4, '0')} · ALT ${$('altitude').textContent}`
}

function addEvent(title, detail, kind = 'neutral') { const node = document.createElement('div'); node.className = 'event'; node.innerHTML = `<time>00:${String(Math.floor(state.time / 1.8) % 60).padStart(2, '0')}</time><span class="event-marker ${kind}"></span><p><b>${title}</b><small>${detail}</small></p>`; $('eventLog').prepend(node) }
function triggerDetection() { if (state.detected) return; state.detected = true; const index = Math.min(Math.floor(state.time / 18), state.route.length - 1); state.route = [...state.route.slice(0, index + 1), { sequence: 99, kind: 'investigate', cell_id: 'TARGET-01', lat: state.data.truth.lat, lon: state.data.truth.lon, rationale: 'thermal corroboration · investigate' }, state.route[state.route.length - 1]]; updateRoute(); addEvent('Possible person detected', '84% confidence · RGB + thermal · route replanned', 'alert'); $('sensorMode').textContent = 'THERMAL CONFIRMED' }
function exportPacket() { const rows = state.route.map((w) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: [w.lon, w.lat] }, properties: { sequence: w.sequence, kind: w.kind, cell_id: w.cell_id || null, rationale: w.rationale || '' } })); download('kairodristi-waypoints.geojson', JSON.stringify({ type: 'FeatureCollection', features: rows }, null, 2), 'application/geo+json'); const csv = ['sequence,kind,cell_id,lat,lon,rationale', ...state.route.map((w) => [w.sequence, w.kind, w.cell_id || '', w.lat, w.lon, JSON.stringify(w.rationale || '')].join(','))].join('\n'); download('kairodristi-waypoints.csv', csv, 'text/csv') }
function download(name, content, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([content], { type })); a.download = name; a.click(); URL.revokeObjectURL(a.href) }
function resetMission() { state.time = 0; state.detected = false; state.running = false; state.route = state.data.waypoints.slice(); $('eventLog').innerHTML = '<div class="event"><time>00:00</time><span class="event-marker neutral"></span><p><b>Mission initialized</b><small>Planner route loaded · return reserve protected</small></p></div>'; $('sensorMode').textContent = 'RGB / THERMAL'; updateRoute() }

function animate() { requestAnimationFrame(animate); if (state.running) { state.time += 0.055 * state.speedMultiplier; if (state.time > state.route.length * 18 + 12) { state.running = false; addEvent('Return to base complete', 'Mission ended with reserve available', 'complete') } } moveDrone(); uiUpdate(); updateOrbit(); renderer.render(scene, camera); renderCameraFeed() }
function initInput() { const el = renderer.domElement; el.addEventListener('pointerdown', (e) => { orbit.down = true; orbit.x = e.clientX; orbit.y = e.clientY; el.setPointerCapture(e.pointerId) }); el.addEventListener('pointerup', () => { orbit.down = false }); el.addEventListener('pointermove', (e) => { if (!orbit.down) return; orbit.theta -= (e.clientX - orbit.x) * 0.006; orbit.phi = THREE.MathUtils.clamp(orbit.phi + (e.clientY - orbit.y) * 0.006, 0.35, 1.5); orbit.x = e.clientX; orbit.y = e.clientY }); el.addEventListener('wheel', (e) => { orbit.radius = THREE.MathUtils.clamp(orbit.radius + e.deltaY * 0.18, 100, 440) }); window.addEventListener('keydown', (e) => { if (e.key.toLowerCase() === 'r') { orbit = { theta: 0.7, phi: 1.05, radius: 260, down: false, x: 0, y: 0 } } }); window.addEventListener('resize', updateCamera) }
function initButtons() { $('toggleMission').onclick = () => { state.running = !state.running; $('toggleMission').textContent = state.running ? 'PAUSE MISSION' : 'RESUME MISSION' }; $('restartMission').onclick = resetMission; $('triggerDetection').onclick = triggerDetection; $('exportPacket').onclick = exportPacket; $('resetView').onclick = () => { orbit = { theta: 0.7, phi: 1.05, radius: 260, down: false, x: 0, y: 0 } }; $('toggleHeat').onclick = () => { state.layers.heat = !state.layers.heat; worldObjects.priority.forEach((x) => x.visible = state.layers.heat); $('toggleHeat').classList.toggle('selected', state.layers.heat) }; $('toggleNfz').onclick = () => { state.layers.nfz = !state.layers.nfz; worldObjects.noFly.visible = state.layers.nfz; $('toggleNfz').classList.toggle('selected', state.layers.nfz) }; $('toggleCoverage').onclick = () => { state.layers.coverage = !state.layers.coverage; $('toggleCoverage').classList.toggle('selected', state.layers.coverage) } }

async function boot() { state.data = await fetch('/api/mission').then((r) => r.json()); state.route = state.data.waypoints.slice(); $('missionId').textContent = state.data.mission_id; buildScene(); buildMissionLayers(); initInput(); initButtons(); updateCamera(); animate() }
boot().catch((error) => { app.innerHTML = `<main class="error-state"><h1>Mission console unavailable</h1><p>${error.message}</p><p>Start the Python API with <code>py -3.11 -m app.server</code>.</p></main>` })
