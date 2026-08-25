/* Renders the live scoreboard from docs/data/scoreboard.json.
   Every figure shown comes from the committed prediction ledgers — nothing is hard-coded.
   Live and backtest summaries are read from separate keys and never combined into one figure. */

const fmtMoney = (v, d = 2) => (v == null ? "—" : "$" + v.toFixed(d));
const fmtPct = (v, d = 0) => (v == null ? "—" : (v * 100).toFixed(d) + "%");
const MODEL_LABEL = { lstm: "LSTM", persistence: "Persistence", llm: "LLM" };

function el(tag, attrs = {}, text) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}

function table(node, head, rows) {
  node.innerHTML = "";
  const thead = el("thead");
  const hr = el("tr");
  head.forEach((h) => hr.appendChild(el("th", {}, h)));
  thead.appendChild(hr);
  node.appendChild(thead);

  const tbody = el("tbody");
  rows.forEach((cells) => {
    const tr = el("tr");
    cells.forEach((c, i) => {
      const td = el("td", { class: i === 0 ? "" : "num" });
      td.textContent = c;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  node.appendChild(tbody);

  if (rows.length === 0) {
    const tr = el("tr");
    const td = el("td", { colspan: String(head.length), class: "stat-empty" },
      "None right now.");
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

/* One stat card per model for a summary block (live or backtest — never both at once). */
function summaryCards(node, summary) {
  node.innerHTML = "";
  const models = Object.keys(summary).sort();
  if (models.length === 0) {
    node.appendChild(el("p", { class: "stat-empty" }, "No predictions yet."));
    return;
  }
  models.forEach((model) => {
    const s = summary[model];
    const card = el("div", { class: "card" });
    card.appendChild(el("div", { class: "k" }, MODEL_LABEL[model] || model));
    if (s.n_scored === 0) {
      card.appendChild(el("div", { class: "v stat-empty" }, "not yet scored"));
      card.appendChild(el("div", { class: "note" }, `${s.n} open`));
    } else {
      card.appendChild(el("div", { class: "v" }, fmtPct(s.coverage, 0) + " coverage"));
      card.appendChild(el("div", { class: "note" },
        `width ${fmtMoney(s.mean_interval_width, 0)} · ` +
        `score ${s.interval_score == null ? "—" : s.interval_score.toFixed(1)} · ` +
        `n=${s.n_scored}`));
    }
    node.appendChild(card);
  });
}

/* Inline SVG: actual price line (solid) with predicted bands (backtest muted, live accent). */
function tickerChart(node, records, model) {
  node.innerHTML = "";
  const rows = records.filter((r) => r.model === model).slice().sort(
    (a, b) => a.target_date.localeCompare(b.target_date)
  );

  if (rows.length === 0) {
    node.appendChild(el("p", { class: "stat-empty" }, "No predictions for this combination."));
    return;
  }

  const W = 900, H = 320, padL = 56, padR = 16, padT = 16, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const xs = rows.map((r) => new Date(r.target_date).getTime());
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const xSpan = Math.max(xMax - xMin, 1);

  /* An LLM abstention has lo/hi/point = null (no numeric claim was made) — it must not
     contribute to the price scale or draw a fabricated band around $0. */
  const values = [];
  rows.forEach((r) => {
    if (r.lo != null) values.push(r.lo);
    if (r.hi != null) values.push(r.hi);
    if (r.actual != null) values.push(r.actual);
  });
  if (values.length === 0) {
    node.appendChild(el("p", { class: "stat-empty" },
      "No numeric predictions for this combination (all abstained)."));
    return;
  }
  const yMin = Math.min(...values), yMax = Math.max(...values);
  const yPad = (yMax - yMin) * 0.08 || 1;
  const yLo = yMin - yPad, yHi = yMax + yPad;

  const x = (t) => padL + ((t - xMin) / xSpan) * plotW;
  const y = (v) => padT + plotH - ((v - yLo) / (yHi - yLo)) * plotH;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.setAttribute("aria-label",
    `Actual price versus predicted ${model} range over time for the selected ticker`);

  /* gridlines + y labels */
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = yLo + ((yHi - yLo) * i) / ticks;
    const yy = y(v);
    svg.appendChild(el("line", {
      x1: padL, y1: yy, x2: W - padR, y2: yy,
      stroke: "var(--line)", "stroke-width": "1",
    }));
    svg.appendChild(el("text", {
      x: padL - 8, y: yy + 4, "text-anchor": "end",
      "font-size": "11", "font-family": "var(--mono)", fill: "var(--muted)",
    }, "$" + v.toFixed(0)));
  }

  /* predicted bands — abstentions (lo/hi null) draw no band at all */
  const bandW = Math.max(plotW / Math.max(rows.length, 1) * 0.6, 4);
  rows.filter((r) => r.lo != null && r.hi != null).forEach((r) => {
    const cx = x(new Date(r.target_date).getTime());
    const isLive = r.mode === "live";
    svg.appendChild(el("rect", {
      x: cx - bandW / 2, y: y(r.hi), width: bandW, height: Math.max(y(r.lo) - y(r.hi), 1),
      fill: "var(--accent)", opacity: isLive ? "0.30" : "0.14",
      stroke: isLive ? "var(--accent)" : "var(--muted)",
      "stroke-width": isLive ? "1.2" : "0.75",
      "stroke-dasharray": isLive ? "none" : "2,2",
      rx: "2",
    }));
  });

  /* actual price line + points, where known */
  const known = rows.filter((r) => r.actual != null);
  if (known.length > 0) {
    const pathD = known
      .map((r, i) => `${i === 0 ? "M" : "L"} ${x(new Date(r.target_date).getTime())} ${y(r.actual)}`)
      .join(" ");
    svg.appendChild(el("path", {
      d: pathD, fill: "none", stroke: "var(--text)", "stroke-width": "1.75",
    }));
    known.forEach((r) => {
      svg.appendChild(el("circle", {
        cx: x(new Date(r.target_date).getTime()), cy: y(r.actual), r: "2.6",
        fill: "var(--text)",
      }));
    });
  }

  /* x-axis labels: first, middle, last */
  const labelRows = rows.length > 1
    ? [rows[0], rows[Math.floor(rows.length / 2)], rows[rows.length - 1]]
    : rows;
  labelRows.forEach((r) => {
    svg.appendChild(el("text", {
      x: x(new Date(r.target_date).getTime()), y: H - 10, "text-anchor": "middle",
      "font-size": "11", "font-family": "var(--mono)", fill: "var(--muted)",
    }, r.target_date));
  });

  node.appendChild(svg);

  const legend = el("div", { class: "chart-legend" });
  const item = (color, dashed, label) => {
    const span = el("span");
    const sw = el("span", { class: "swatch" });
    sw.style.background = color;
    if (dashed) sw.style.border = "1px dashed var(--muted)";
    span.appendChild(sw);
    span.appendChild(document.createTextNode(label));
    return span;
  };
  legend.appendChild(item("var(--text)", false, "actual close"));
  legend.appendChild(item("var(--accent)", false, "live predicted band"));
  legend.appendChild(item("var(--surface-2)", true, "backtest predicted band"));
  node.appendChild(legend);
}

function render(data) {
  /* ---- headline ---- */
  const liveModels = Object.values(data.live_summary);
  const anyScored = liveModels.some((s) => s.n_scored > 0);
  document.getElementById("answer-body").textContent = anyScored
    ? `Coverage and interval score for each arm are broken out below — see "Live track record".`
    : `No live prediction has matured yet (horizon is ${data.horizon_days} trading days). ` +
      `${data.open_predictions.length} open prediction(s) are already committed and waiting.`;

  document.getElementById("footer-generated").textContent =
    "generated " + new Date(data.generated_utc).toLocaleString();

  /* ---- open predictions ---- */
  table(
    document.getElementById("table-open"),
    ["Ticker", "Model", "As of", "Predicted band", "Target date", "Days left"],
    data.open_predictions.map((r) => [
      r.symbol,
      MODEL_LABEL[r.model] || r.model,
      r.as_of,
      r.lo == null ? "abstained" : `${fmtMoney(r.lo)} – ${fmtMoney(r.hi)}`,
      r.target_date,
      String(r.days_remaining),
    ])
  );

  /* ---- summaries: separate, never combined ---- */
  summaryCards(document.getElementById("live-summary"), data.live_summary);
  summaryCards(document.getElementById("backtest-summary"), data.backtest_summary);

  /* ---- per-ticker chart ---- */
  const tickerSelect = document.getElementById("ticker-select");
  const modelSelect = document.getElementById("model-select");
  tickerSelect.innerHTML = "";
  data.tickers.forEach((t) => tickerSelect.appendChild(el("option", { value: t }, t)));

  function refreshModelOptions() {
    const records = data.series[tickerSelect.value] || [];
    const models = [...new Set(records.map((r) => r.model))].sort();
    const previous = modelSelect.value;
    modelSelect.innerHTML = "";
    models.forEach((m) =>
      modelSelect.appendChild(el("option", { value: m }, MODEL_LABEL[m] || m))
    );
    if (models.includes(previous)) modelSelect.value = previous;
    else if (models.includes("lstm")) modelSelect.value = "lstm";
  }

  function draw() {
    const records = data.series[tickerSelect.value] || [];
    tickerChart(document.getElementById("chart-ticker"), records, modelSelect.value);
  }

  tickerSelect.addEventListener("change", () => { refreshModelOptions(); draw(); });
  modelSelect.addEventListener("change", draw);

  refreshModelOptions();
  draw();
}

fetch("data/scoreboard.json")
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then(render)
  .catch((err) => {
    document.getElementById("answer-body").textContent =
      "Could not load the scoreboard: " + err.message;
  });
