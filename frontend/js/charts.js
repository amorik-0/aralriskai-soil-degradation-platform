/* Chart.js helpers shared by AralRiskAI visualizations. */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function getThemeColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: getThemeColor("--text-muted"),
          boxWidth: 10,
          usePointStyle: true
        }
      }
    },
    scales: {
      x: {
        ticks: { color: getThemeColor("--text-muted") },
        grid: { color: getThemeColor("--border") }
      },
      y: {
        ticks: { color: getThemeColor("--text-muted") },
        grid: { color: getThemeColor("--border") }
      }
    }
  };
}

window.ChartHelpers = { MONTHS, getThemeColor, baseChartOptions };
