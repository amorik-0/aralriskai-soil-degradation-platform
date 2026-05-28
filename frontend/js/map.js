/* Leaflet map helpers for the AralRiskAI dashboard. */

const ARALKUM = { lat: 44.2, lon: 58.8 };
const GRID = { rows: 28, cols: 45, latMin: 41.5, latMax: 48.5, lonMin: 55, lonMax: 68 };

const COLOR_STOPS = [
  [0.00, [63, 143, 105]],
  [0.55, [199, 139, 32]],
  [1.00, [196, 82, 61]]
];

function riskToRGB(risk) {
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const [startValue, startColor] = COLOR_STOPS[i];
    const [endValue, endColor] = COLOR_STOPS[i + 1];
    if (risk <= endValue) {
      const t = (risk - startValue) / (endValue - startValue);
      return startColor.map((value, index) => Math.round(value + (endColor[index] - value) * t));
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1][1];
}

function riskToCSS(risk, opacity = 1) {
  const [red, green, blue] = riskToRGB(risk);
  return `rgba(${red}, ${green}, ${blue}, ${opacity})`;
}

function createSourceLabel() {
  return L.divIcon({
    className: "",
    html: '<div class="source-label">Aralkum source region</div>',
    iconAnchor: [68, 12]
  });
}

window.MapHelpers = { ARALKUM, GRID, riskToCSS, createSourceLabel };
