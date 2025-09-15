// main.js — predictions + analysis + UX polish (with destroyChart defined)

let resultsChart = null;
const chartRegistry = {};   // holds Chart.js instances by canvas id
let lastForecastCache = null;   // for CSV export
let lastHeadRowsCache = null;   // for CSV export

function $(sel) { return document.querySelector(sel); }
function $all(sel) { return Array.from(document.querySelectorAll(sel)); }

function num(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  return Number(v).toLocaleString("en-US", { maximumFractionDigits: digits });
}

function setLoading(el, isLoading, text = "Running...") {
  if (!el) return;
  if (isLoading) {
    el.dataset.originalText = el.innerHTML;
    el.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${text}`;
    el.disabled = true;
  } else {
    if (el.dataset.originalText) el.innerHTML = el.dataset.originalText;
    el.disabled = false;
  }
}

function showAlert(el, msg) {
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("d-none");
}
function hideAlert(el) {
  if (!el) return;
  el.textContent = "";
  el.classList.add("d-none");
}

function csvFromRows(headers, rows) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
  };
  const head = headers.map(esc).join(",");
  const body = rows.map(r => r.map(esc).join(",")).join("\n");
  return head + "\n" + body;
}
function downloadCSV(filename, csv) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* -------------------- Prediction -------------------- */

async function callPredict(head, ticker, days, submitBtn) {
  const alertEl = $("#resultsAlert");
  hideAlert(alertEl);

  try {
    setLoading(submitBtn, true);

    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ head, ticker, days })
    });

    let data;
    try { data = await resp.json(); } catch { throw new Error("Invalid JSON from server."); }
    if (!resp.ok) {
      const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
      throw new Error(msg);
    }

    lastForecastCache = {
      head: data.head,
      ticker: data.ticker,
      dates: data.dates,
      seriesName: data.series_name,
      predicted: data.predicted
    };

    renderResults({
      head,
      ticker: data.ticker,
      seriesName: data.series_name,
      dates: data.dates,
      predicted: data.predicted,
      n: days
    });

  } catch (err) {
    console.error(err);
    showAlert(alertEl, `Prediction failed: ${err.message || err}`);
    const panel = $("#resultsPanel");
    panel && panel.classList.remove("d-none");
  } finally {
    setLoading(submitBtn, false);
  }
}

function renderResults({ head, ticker, seriesName, dates, predicted, n }) {
  const panel = $("#resultsPanel");
  const title = $("#resultsTitle");
  const tbody = $("#resultsTable tbody");
  const ctx = $("#resultsChart");

  const label = seriesName === "pred_close" ? "Close"
              : seriesName === "pred_open"  ? "Open"
              : "High";
  title.textContent = `${ticker} — ${label} forecast (next ${n} trading days)`;

  // Table
  if (tbody) {
    tbody.innerHTML = "";
    for (let i = 0; i < dates.length; i++) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${dates[i]}</td><td>${num(predicted[i])}</td>`;
      tbody.appendChild(tr);
    }
  }

  // Chart
  if (resultsChart) { resultsChart.destroy(); resultsChart = null; }
  if (ctx && typeof Chart !== "undefined") {
    resultsChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: dates,
        datasets: [{
          label: `${ticker} ${label} (predicted)`,
          data: predicted,
          tension: 0.25,
          borderWidth: 2,
          pointRadius: 0,
          borderColor: "#0d6efd",
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: true },
          tooltip: { callbacks: { label: (ctx) => ` ${num(ctx.parsed.y)}` } }
        },
        scales: { x: { ticks: { maxTicksLimit: 12 } }, y: { beginAtZero: false } }
      }
    });
  }

  panel && panel.classList.remove("d-none");
  panel && panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function clearResults() {
  const panel = $("#resultsPanel");
  const tbody = $("#resultsTable tbody");
  const alertEl = $("#resultsAlert");
  hideAlert(alertEl);
  if (resultsChart) { resultsChart.destroy(); resultsChart = null; }
  if (tbody) tbody.innerHTML = "";
  lastForecastCache = null;
  panel && panel.classList.add("d-none");
}

/* -------------------- Analysis -------------------- */

async function callAnalysis(ticker, btn) {
  const alertEl = $("#analysisAlert");
  hideAlert(alertEl);

  try {
    setLoading(btn, true, "Loading...");
    const url = `/api/analysis?ticker=${encodeURIComponent(ticker)}`;
    const resp = await fetch(url);

    let data;
    try { data = await resp.json(); } catch { throw new Error("Empty or invalid JSON from server."); }
    if (!resp.ok) {
      const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
      throw new Error(msg);
    }
    if (!data || !data.charts) {
      showAlert(alertEl, "No analysis data returned.");
    }
    // cache first 5 rows for CSV
    lastHeadRowsCache = data.first_5_rows || [];

    renderAnalysis(data);

  } catch (err) {
    console.error(err);
    showAlert(alertEl, `Analysis failed: ${err.message || err}`);
    const panel = $("#analysisOutput");
    panel && panel.classList.remove("d-none");
  } finally {
    setLoading(btn, false);
  }
}

function renderAnalysis(data) {
  const panel = $("#analysisOutput");
  panel && panel.classList.remove("d-none");

  // 1) First 5 rows table
  const tbody = $("#analysisHeadTable tbody");
  if (tbody) {
    tbody.innerHTML = "";
    (data.first_5_rows || []).forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${r.date || ""}</td>` +
        `<td>${num(r.open)}</td>` +
        `<td>${num(r.high)}</td>` +
        `<td>${num(r.low)}</td>` +
        `<td>${num(r.close)}</td>` +
        `<td>${r.volume == null ? "" : Number(r.volume).toLocaleString("en-US")}</td>`;
      tbody.appendChild(tr);
    });
  } else {
    console.warn("#analysisHeadTable not found; skipping table render.");
  }

  // 2) Charts
  const ch = (data && data.charts) ? data.charts : {};
  drawMAChart("closeMAChart", ch.close_ma, "Close");
  drawMAChart("openMAChart",  ch.open_ma,  "Open");
  drawMAChart("highMAChart",  ch.high_ma,  "High");
  drawRawChart("closeRawChart", ch.close_raw, "Close (raw)");
  drawRawChart("openRawChart",  ch.open_raw,  "Open (raw)");

  panel && panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- THIS WAS MISSING: define destroyChart before draw functions ----
function destroyChart(id) {
  if (chartRegistry[id]) {
    try { chartRegistry[id].destroy(); } catch (e) { /* ignore */ }
    delete chartRegistry[id];
  }
}

function drawMAChart(canvasId, series, priceLabel) {
  if (!series) { destroyChart(canvasId); return; }
  const el = document.getElementById(canvasId);
  if (!el || typeof Chart === "undefined") return;

  const labels = series.dates || [];
  const price = series.price || [];
  const ma30  = series.ma30 || [];
  const ma50  = series.ma50 || [];

  destroyChart(canvasId);
  chartRegistry[canvasId] = new Chart(el, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: `${priceLabel}`, data: price, borderWidth: 2, tension: 0.25, pointRadius: 0, borderColor: "#0d6efd" },
        { label: "MA30", data: ma30, borderWidth: 1.5, tension: 0.25, pointRadius: 0, borderDash: [6,4], borderColor: "#20c997" },
        { label: "MA50", data: ma50, borderWidth: 1.5, tension: 0.25, pointRadius: 0, borderDash: [6,4], borderColor: "#6f42c1" }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,  // respect fixed container height
      animation: false,            // avoid reflow/shift on draw
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true } },
      scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { beginAtZero: false } }
    }
  });
}

function drawRawChart(canvasId, series, label) {
  if (!series) { destroyChart(canvasId); return; }
  const el = document.getElementById(canvasId);
  if (!el || typeof Chart === "undefined") return;

  const labels = series.dates || [];
  const price  = series.price || [];

  destroyChart(canvasId);
  chartRegistry[canvasId] = new Chart(el, {
    type: "line",
    data: {
      labels,
      datasets: [{ label, data: price, borderWidth: 2, tension: 0.25, pointRadius: 0, borderColor: "#fd7e14" }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,  // respect fixed container height
      animation: false,            // avoid reflow/shift on draw
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true } },
      scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { beginAtZero: false } }
    }
  });
}

/* -------------------- DOM ready & handlers -------------------- */

document.addEventListener('DOMContentLoaded', () => {
  // Smooth scroll to analysis panel
  $("#openAnalysis")?.addEventListener('click', (e) => {
    e.preventDefault();
    $("#analysisPanel")?.scrollIntoView({ behavior: 'smooth' });
  });

  // Prediction forms
  const formClose = $("#formClose");
  if (formClose) {
    formClose.addEventListener("submit", (e) => {
      e.preventDefault();
      const fd = new FormData(formClose);
      const ticker = (fd.get("ticker") || "AAPL").toString().trim().toUpperCase();
      const days = Math.min(252, Math.max(1, parseInt(fd.get("days") || "30", 10)));
      const btn = formClose.querySelector('button[type="submit"]');
      callPredict("close", ticker, days, btn);
      bootstrap.Modal.getInstance(document.getElementById("modalClose"))?.hide();
    });
  }

  const formOpen = $("#formOpen");
  if (formOpen) {
    formOpen.addEventListener("submit", (e) => {
      e.preventDefault();
      const fd = new FormData(formOpen);
      const ticker = (fd.get("ticker") || "AAPL").toString().trim().toUpperCase();
      const days = Math.min(252, Math.max(1, parseInt(fd.get("days") || "30", 10)));
      const btn = formOpen.querySelector('button[type="submit"]');
      callPredict("open", ticker, days, btn);
      bootstrap.Modal.getInstance(document.getElementById("modalOpen"))?.hide();
    });
  }

  const formHigh = $("#formHigh");
  if (formHigh) {
    formHigh.addEventListener("submit", (e) => {
      e.preventDefault();
      const fd = new FormData(formHigh);
      const ticker = (fd.get("ticker") || "AAPL").toString().trim().toUpperCase();
      const days = Math.min(252, Math.max(1, parseInt(fd.get("days") || "30", 10)));
      const btn = formHigh.querySelector('button[type="submit"]');
      callPredict("high", ticker, days, btn);
      bootstrap.Modal.getInstance(document.getElementById("modalHigh"))?.hide();
    });
  }

  // Clear Results
  $("#btnClearResults")?.addEventListener("click", clearResults);

  // Export forecast CSV
  $("#btnExportForecast")?.addEventListener("click", () => {
    if (!lastForecastCache) return;
    const { ticker, seriesName, dates, predicted } = lastForecastCache;
    const label = seriesName === "pred_close" ? "pred_close" : seriesName === "pred_open" ? "pred_open" : "pred_high";
    const rows = dates.map((d, i) => [d, predicted[i]]);
    const csv = csvFromRows(["date", label], rows);
    downloadCSV(`${ticker}_${label}.csv`, csv);
  });

  // Analysis button
  const btnRunAnalysis = $("#btnRunAnalysis");
  if (btnRunAnalysis) {
    btnRunAnalysis.addEventListener("click", () => {
      const t = ($("#analysisTicker").value || "AAPL").toString().trim().toUpperCase();
      callAnalysis(t, btnRunAnalysis);
    });
  }

  // Export analysis head CSV
  $("#btnExportHead")?.addEventListener("click", () => {
    if (!lastHeadRowsCache) return;
    const rows = lastHeadRowsCache.map(r => [r.date, r.open, r.high, r.low, r.close, r.volume]);
    const csv = csvFromRows(["date","open","high","low","close","volume"], rows);
    downloadCSV(`analysis_head.csv`, csv);
  });
});
