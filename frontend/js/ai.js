/* ============================================================
   ai.js — AI Scientific Assistant (Gemini API)
   ============================================================ */

const GEMINI_KEY = 'AIzaSyBRiSmKikeEMj8PzEsSr-wy3tnKUBUV-nU';
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_KEY}`;

const SUGGESTION_PILLS = [
  'Why is the risk so high?',
  'Is it safe to irrigate tomorrow?',
  'What does NDSI 0.42 mean?',
  'How to reduce soil salinity?',
  'Explain the SHAP values',
  'What crops survive high salinity?',
];

let _currentData = null;

function setCurrentData(data) {
  _currentData = data;
}

function buildSystemPrompt() {
  const d = _currentData;
  const dataBlock = d ? `
SELECTED LOCATION: ${d.lat.toFixed(3)}N, ${d.lon.toFixed(3)}E

RISK ASSESSMENT:
  Overall degradation risk: ${d.risk.toFixed(3)} / 1.00  [${d.riskLabel}]
  Salinity risk:            ${d.salinityRisk.toFixed(3)}
  Vegetation loss risk:     ${d.vegRisk.toFixed(3)}
  Salt-dust exposure risk:  ${d.dustRisk.toFixed(3)}

SATELLITE INDICES (Sentinel-2):
  NDVI  = ${d.features.ndvi.toFixed(3)}   [healthy: >0.3, degraded: <0.15]
  NDSI  = ${d.features.ndsi.toFixed(3)}   [high-risk: >0.25]
  SI_SWIR = ${d.features.si_swir.toFixed(3)}

CLIMATE (ERA5):
  Wind speed: ${d.features.windSpeed.toFixed(1)} m/s
  Soil moisture: ${d.features.moisture.toFixed(3)} m3/m3
  Temperature: ${d.features.temp.toFixed(1)} C
  Distance from Aralkum: ${d.features.distKm.toFixed(0)} km` 
  : 'No location selected yet. Ask user to click on the map.';

  return `You are AralRiskAI, a scientific assistant for land degradation monitoring in the Aral Sea / Aralkum Desert region of Central Asia.

You interpret Sentinel-2 satellite indices (NDVI, NDSI, SI_SWIR), ERA5 climate data, and soil properties to explain land degradation risk to farmers and scientists.

Rules:
1. Always cite specific data values in your response
2. Explain what each index means in plain language
3. Give concrete, actionable recommendations
4. Be concise: 3-5 sentences
5. Use metric units

CURRENT DATA:
${dataBlock}`;
}

function appendMessage(container, role, text) {
  const div = document.createElement('div');
  div.className = `ai-msg ${role}`;
  if (role === 'bot') {
    div.innerHTML = `<span class="src">ARAL-AI</span>${text}`;
  } else {
    div.textContent = text;
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

async function sendMessage(question) {
  if (!question.trim()) return;

  const container = document.getElementById('ai-messages');
  const sendBtn   = document.getElementById('ai-send');

  appendMessage(container, 'user', question);
  const botEl = appendMessage(container, 'bot', '<em>Analysing satellite data...</em>');
  sendBtn.disabled = true;

  const fullPrompt = buildSystemPrompt() + '\n\nQuestion: ' + question;

  try {
    const res = await fetch(GEMINI_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: fullPrompt }] }]
      })
    });
    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response received.';
    botEl.innerHTML = `<span class="src">ARAL-AI</span>${text}`;
  } catch (err) {
    botEl.innerHTML = `<span class="src">ERROR</span>Connection failed. Check network.`;
  }

  sendBtn.disabled = false;
  container.scrollTop = container.scrollHeight;
}

function initAI() {
  document.getElementById('ai-toggle').addEventListener('click', () => {
    const body   = document.getElementById('ai-body');
    const toggle = document.getElementById('ai-toggle');
    const open   = body.style.display !== 'flex';
    body.style.display = open ? 'flex' : 'none';
    toggle.classList.toggle('open', open);
  });

  document.getElementById('ai-send').addEventListener('click', () => {
    const inp = document.getElementById('ai-input');
    sendMessage(inp.value);
    inp.value = '';
  });

  document.getElementById('ai-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      sendMessage(e.target.value);
      e.target.value = '';
    }
  });

  const pillRow = document.getElementById('pill-row');
  SUGGESTION_PILLS.forEach(text => {
    const pill = document.createElement('div');
    pill.className = 'pill';
    pill.textContent = text;
    pill.addEventListener('click', () => sendMessage(text));
    pillRow.appendChild(pill);
  });

  const container = document.getElementById('ai-messages');
  appendMessage(container, 'bot',
    'Hello! I analyse Sentinel-2 and ERA5 data for the Aralkum region. Click any point on the map, then ask me about the risk, salinity, or farming recommendations.'
  );
}

window.AIAssistant = { initAI, setCurrentData };
