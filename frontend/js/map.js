/* ============================================================
   map.js — Leaflet Map Engine
   Handles: map init, risk gradient overlay, wind arrows,
            point selection, timeline updates
   ============================================================ */

// ── CONSTANTS ─────────────────────────────────────────────────
const ARALKUM = { lat: 44.2, lon: 58.8 };   // dried seabed center
const GRID = { rows: 28, cols: 45, latMin: 41.5, latMax: 48.5, lonMin: 55, lonMax: 68 };

// ── COLOR SCALE ───────────────────────────────────────────────
// Continuous gradient: deep green → light green → amber → orange → red
const COLOR_STOPS = [
  [0.00, [15,  76,  43]],
  [0.25, [74,  222, 128]],
  [0.50, [251, 191,  36]],
  [0.75, [249, 115,  22]],
  [1.00, [220,  38,  38]],
];

function riskToRGB(risk) {
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const [t0, c0] = COLOR_STOPS[i];
    const [t1, c1] = COLOR_STOPS[i + 1];
    if (risk <= t1) {
      const t = (risk - t0) / (t1 - t0);
      return c0.map((v, j) => Math.round(v + (c1[j] - v) * t));
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1][1];
}

function riskToCSS(risk, opacity = 1) {
  const [r, g, b] = riskToRGB(risk);
  return `rgba(${r},${g},${b},${opacity})`;
}

// ── STATE ──────────────────────────────────────────────────────
let map, riskLayerGroup, windLayerGroup, selectedMarker;

// ── INIT ───────────────────────────────────────────────────────
function initMap(onPointClick) {
  map = L.map('map', { center: [44.5, 60.5], zoom: 6, zoomControl: true, attributionControl: false });

  // Dark CartoDB tiles
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CARTO',
    subdomains: 'abcd',
    maxZoom: 14,
  }).addTo(map);

  riskLayerGroup = L.layerGroup().addTo(map);
  windLayerGroup = L.layerGroup().addTo(map);

  // Source label: Aralkum Desert
  L.marker([44.2, 58.8], {
    icon: L.divIcon({
      className: '',
      html: `<div style="background:rgba(220,38,38,.85);color:#fff;font-size:9px;
             padding:3px 6px;border-radius:3px;white-space:nowrap;
             border:1px solid #f87171;font-family:'Courier New',monospace">
             ▲ ARALKUM SOURCE</div>`,
      iconAnchor: [56, 10],
    }),
  }).addTo(map);

  drawRiskGrid(0);
  drawWindArrows();

  // Click handler
  map.on('click', e => {
    placeMarker(e.latlng.lat, e.latlng.lng);
    if (typeof onPointClick === 'function') {
      onPointClick(e.latlng.lat, e.latlng.lng);
    }
  });
}

// ── RISK GRID ──────────────────────────────────────────────────
// Draws 28×45 = 1260 colored rectangles showing continuous risk gradient.
// Each cell uses the scientific risk formula from RiskEngine.
function drawRiskGrid(timeOffset) {
  riskLayerGroup.clearLayers();
  const { rows, cols, latMin, latMax, lonMin, lonMax } = GRID;
  const dlat = (latMax - latMin) / rows;
  const dlon = (lonMax - lonMin) / cols;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const lat = latMin + r * dlat + dlat / 2;
      const lon = lonMin + c * dlon + dlon / 2;

      // Get risk from RiskEngine (defined in risk_engine.js or inline)
      const { risk } = typeof RiskEngine !== 'undefined'
        ? RiskEngine.spatialProxy(lat, lon, timeOffset)
        : { risk: 0.3 };

      L.rectangle(
        [[latMin + r * dlat, lonMin + c * dlon],
         [latMin + (r + 1) * dlat, lonMin + (c + 1) * dlon]],
        {
          weight: 0,
          fillColor: riskToCSS(risk),
          fillOpacity: 0.52,
          interactive: false,
        }
      ).addTo(riskLayerGroup);
    }
  }
}

// ── WIND ARROWS ───────────────────────────────────────────────
// ERA5 representative u/v wind components for the region (JJA climatology).
// Wind speed = √(u²+v²), direction = atan2(u,v)
const ERA5_WIND_VECTORS = [
  { lat: 47.0, lon: 60.0, u: -2.1, v: -5.3 },
  { lat: 47.0, lon: 63.0, u: -1.8, v: -5.8 },
  { lat: 47.0, lon: 65.5, u: -1.2, v: -4.9 },
  { lat: 45.5, lon: 59.0, u: -3.2, v: -4.8 },
  { lat: 45.5, lon: 61.5, u: -2.8, v: -5.1 },
  { lat: 45.5, lon: 64.0, u: -2.0, v: -5.4 },
  { lat: 44.0, lon: 58.0, u: -4.5, v: -3.8 },
  { lat: 44.0, lon: 61.0, u: -3.9, v: -4.2 },
  { lat: 44.0, lon: 63.5, u: -3.0, v: -4.6 },
  { lat: 42.5, lon: 59.0, u: -5.2, v: -2.9 },
  { lat: 42.5, lon: 62.0, u: -4.7, v: -3.4 },
  { lat: 42.5, lon: 65.0, u: -3.8, v: -3.9 },
];

function drawWindArrows() {
  windLayerGroup.clearLayers();
  ERA5_WIND_VECTORS.forEach(p => {
    // wind_speed = √(u² + v²)  [m/s]
    // direction  = atan2(u, v) → meteorological degrees
    const speed = Math.sqrt(p.u * p.u + p.v * p.v);
    const dir   = (Math.atan2(p.u, p.v) * 180 / Math.PI + 360) % 360;
    const sz    = Math.max(14, Math.min(26, speed * 2.2));

    const icon = L.divIcon({
      className: '',
      html: `<div style="transform:rotate(${dir}deg);font-size:${sz}px;
             color:rgba(147,197,253,.75);line-height:1;user-select:none"
             title="${speed.toFixed(1)} m/s">↑</div>`,
      iconSize: [sz, sz],
      iconAnchor: [sz / 2, sz / 2],
    });
    L.marker([p.lat, p.lon], { icon, interactive: false }).addTo(windLayerGroup);
  });
}

// ── MARKER ────────────────────────────────────────────────────
function placeMarker(lat, lon) {
  if (selectedMarker) map.removeLayer(selectedMarker);
  selectedMarker = L.circleMarker([lat, lon], {
    radius: 9, color: '#fff', weight: 2.5,
    fillColor: '#fbbf24', fillOpacity: .9,
  }).addTo(map);
}

// ── EXPORTS ───────────────────────────────────────────────────
window.MapEngine = { initMap, drawRiskGrid, placeMarker, riskToCSS };
