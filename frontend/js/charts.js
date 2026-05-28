/* ============================================================
   charts.js — Chart.js Visualizations
   Charts: NDVI trend, risk forecast, salinity, SHAP importance
   ============================================================ */

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

// Track chart instances to destroy before re-creating
const _charts = {};

function destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

// ── SHARED OPTIONS ─────────────────────────────────────────────
const BASE_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
};
const GRID_COLOR  = '#162035';
const TICK_COLOR  = '#475569';
const TICK_FONT   = { size: 9 };

// ── NDVI TREND CHART ───────────────────────────────────────────
// Shows monthly NDVI values for selected location.
// Values computed from Sentinel-2 seasonal pattern × location baseline.
function renderNDVIChart(canvasId, ndviValues) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  _charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: MONTHS,
      datasets: [{
        label: 'NDVI',
        data: ndviValues.map(v => parseFloat(v.toFixed(3))),
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74,222,128,.07)',
        fill: true,
        tension: .4,
        pointRadius: 2.5,
        pointBackgroundColor: '#4ade80',
        borderWidth: 1.5,
      }],
    },
    options: {
      ...BASE_OPTS,
      scales: {
        y: { min: 0, max: .7, ticks: { color: TICK_COLOR, font: TICK_FONT }, grid: { color: GRID_COLOR } },
        x: { ticks: { color: TICK_COLOR, font: TICK_FONT }, grid: { color: GRID_COLOR } },
      },
    },
  });
}

// ── RISK FORECAST CHART ────────────────────────────────────────
// 30-day degradation risk forecast with confidence interval.
// Production: uses ERA5 weather forecast + LSTM-projected NDVI.
function renderForecastChart(canvasId, baseRisk) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  const days = Array.from({ length: 30 }, (_, i) => i + 1);

  const forecast  = days.map(d => parseFloat(Math.min(1, Math.max(0, baseRisk + .07*Math.sin(d*.28) + d*.0018)).toFixed(3)));
  const lowerBand = days.map(d => parseFloat(Math.min(1, Math.max(0, baseRisk - .04 + .05*Math.sin(d*.28) + d*.0012)).toFixed(3)));

  _charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: days,
      datasets: [
        {
          label: 'Forecast',
          data: forecast,
          borderColor: '#f87171',
          backgroundColor: 'rgba(248,113,113,.07)',
          fill: true, tension: .3, pointRadius: 0, borderWidth: 1.5,
        },
        {
          label: 'Lower CI',
          data: lowerBand,
          borderColor: 'rgba(248,113,113,.3)',
          borderDash: [3, 3],
          fill: false, tension: .3, pointRadius: 0, borderWidth: 1,
        },
      ],
    },
    options: {
      ...BASE_OPTS,
      scales: {
        y: { min: 0, max: 1, ticks: { color: TICK_COLOR, font: TICK_FONT }, grid: { color: GRID_COLOR } },
        x: { ticks: { color: TICK_COLOR, font: TICK_FONT, maxTicksLimit: 6 }, grid: { color: GRID_COLOR } },
      },
    },
  });
}

// ── SALINITY TREND CHART ───────────────────────────────────────
// Monthly NDSI (salinity index) trend. Inverse of vegetation season.
// Source: Sentinel-2 B11 (SWIR) / B8 (NIR)
function renderSalinityChart(canvasId, ndsiBase) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  // Salinity peaks in summer (Jul-Aug): low vegetation → salt exposed
  const seasonal = [1.3, 1.2, 1.1, 1.0, .9, .85, .85, .9, .95, 1.05, 1.2, 1.35];
  const values = seasonal.map(s => parseFloat(Math.min(.7, Math.max(0, ndsiBase * s)).toFixed(3)));

  _charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: MONTHS,
      datasets: [{
        label: 'NDSI',
        data: values,
        borderColor: '#f87171',
        backgroundColor: 'rgba(248,113,113,.07)',
        fill: true, tension: .4, pointRadius: 2, borderWidth: 1.5,
      }],
    },
    options: {
      ...BASE_OPTS,
      scales: {
        y: { min: 0, max: .7, ticks: { color: TICK_COLOR, font: TICK_FONT }, grid: { color: GRID_COLOR } },
        x: { ticks: { color: TICK_COLOR, font: TICK_FONT }, grid: { color: GRID_COLOR } },
      },
    },
  });
}

// ── SHAP FEATURE IMPORTANCE CHART ──────────────────────────────
// Horizontal bar chart showing contribution of each feature to risk score.
// Values approximate SHAP (SHapley Additive exPlanations) outputs.
// Production: computed via shap.TreeExplainer(xgb_model)
function renderSHAPChart(canvasId, shapItems) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  const sorted = [...shapItems].sort((a, b) => b.val - a.val);

  _charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(i => i.name),
      datasets: [{
        data: sorted.map(i => parseFloat(i.val.toFixed(3))),
        backgroundColor: ['#f87171','#f97316','#fbbf24','#a78bfa','#60a5fa','#34d399'],
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      ...BASE_OPTS,
      plugins: {
        ...BASE_OPTS.plugins,
        tooltip: { callbacks: { label: c => `SHAP: ${c.raw.toFixed(3)}` } },
      },
      scales: {
        x: { min: 0, ticks: { color: TICK_COLOR, font: TICK_FONT }, grid: { color: GRID_COLOR } },
        y: { ticks: { color: '#94a3b8', font: TICK_FONT }, grid: { color: GRID_COLOR } },
      },
    },
  });
}

// ── PUBLIC API ─────────────────────────────────────────────────
window.ChartEngine = {
  renderNDVIChart,
  renderForecastChart,
  renderSalinityChart,
  renderSHAPChart,
};
