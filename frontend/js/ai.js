/* Interpretation Panel utilities.
   The current dashboard renders interpretation text locally from selected
   environmental indicators and does not call an external text service. */

function buildInterpretationText(data) {
  if (!data) {
    return "Select a location to view a scientific interpretation of the environmental indicators.";
  }

  const riskLabel = data.riskLabel || "unclassified";
  const features = data.features || {};
  return [
    `The selected location has a ${riskLabel.toLowerCase()} degradation risk estimate.`,
    `NDVI is ${Number(features.ndvi || 0).toFixed(3)}, NDSI is ${Number(features.ndsi || 0).toFixed(3)}, and wind speed is ${Number(features.windSpeed || 0).toFixed(1)} m/s.`,
    "The estimate combines vegetation condition, salinity, surface dryness, climate stress, and dust exposure indicators."
  ].join(" ");
}

window.InterpretationPanel = { buildInterpretationText };
