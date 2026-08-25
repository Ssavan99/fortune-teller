/* Renders the live scoreboard from docs/data/scoreboard.json.
   Every figure shown comes from the committed prediction ledgers — nothing is hard-coded.
   Live and backtest summaries are read from separate keys and never combined into one figure. */

const fmtMoney = (v, d = 2) => (v == null ? "—" : "$" + v.toFixed(d));
const fmtPct = (v, d = 0) => (v == null ? "—" : (v * 100).toFixed(d) + "%");
const MODEL_LABEL = { lstm: "LSTM", persistence: "Persistence", llm: "LLM" };
const MODEL_ORDER = ["persistence", "lstm", "llm"];
const MODEL_COLOR_VAR = { lstm: "--blue", persistence: "--accent", llm: "--warn" };

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

/* ---- tabs ---- */
function initTabs() {
  const tabs = [
    { btn: "tab-btn-scoreboard", panel: "tab-scoreboard" },
    { btn: "tab-btn-how", panel: "tab-how" },
  ];
  tabs.forEach(({ btn, panel }) => {
    document.getElementById(btn).addEventListener("click", () => {
      tabs.forEach(({ btn: b, panel: p }) => {
        const isActive = b === btn;
        document.getElementById(b).setAttribute("aria-selected", String(isActive));
        document.getElementById(p).classList.toggle("active", isActive);
      });
    });
  });
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
      const extra = s.n_abstained ? `${s.n} open · ${s.n_abstained} abstained` : `${s.n} open`;
      card.appendChild(el("div", { class: "note" }, extra));
    } else {
      card.appendChild(el("div", { class: "v" }, fmtPct(s.coverage, 0) + " coverage"));
      const abstainedNote = s.n_abstained ? ` · ${s.n_abstained} abstained` : "";
      card.appendChild(el("div", { class: "note" },
        `width ${fmtMoney(s.mean_interval_width, 0)} · ` +
        `score ${s.interval_score == null ? "—" : s.interval_score.toFixed(1)} · ` +
        `n=${s.n_scored}${abstainedNote}`));
    }
    node.appendChild(card);
  });
}

/* ---- open predictions: one card per ticker, all models' ranges overlaid ---- */
function tickerCards(container, openPredictions) {
  container.innerHTML = "";
  const bySymbol = {};
  openPredictions.forEach((r) => {
    (bySymbol[r.symbol] = bySymbol[r.symbol] || []).push(r);
  });

  const symbols = Object.keys(bySymbol).sort();
  if (symbols.length === 0) {
    container.appendChild(el("p", { class: "stat-empty" }, "No open predictions right now."));
    return;
  }

  symbols.forEach((symbol) => {
    const rows = bySymbol[symbol];
    const card = el("div", { class: "ticker-card" });

    const head = el("div", { class: "ticker-card-head" });
    head.appendChild(el("span", { class: "ticker-symbol" }, symbol));
    const daysRemaining = Math.min(...rows.map((r) => r.days_remaining));
    head.appendChild(el("span", { class: "ticker-days" }, `${daysRemaining}d left`));
    card.appendChild(head);
    card.appendChild(el("div", { class: "ticker-target" },
      `as of ${rows[0].as_of} · target ${rows[0].target_date}`));

    const numeric = rows.filter((r) => r.lo != null);
    const scaleLo = numeric.length ? Math.min(...numeric.map((r) => r.lo)) : 0;
    const scaleHi = numeric.length ? Math.max(...numeric.map((r) => r.hi)) : 1;
    const pad = (scaleHi - scaleLo) * 0.08 || 1;
    const domainLo = scaleLo - pad, domainHi = scaleHi + pad;
    const pct = (v) => ((v - domainLo) / (domainHi - domainLo)) * 100;

    MODEL_ORDER.forEach((model) => {
      const r = rows.find((x) => x.model === model);
      if (!r) return;

      const row = el("div", { class: "range-row" });
      row.appendChild(el("span", { class: "range-label" }, MODEL_LABEL[model]));
      const track = el("div", { class: "range-track" });

      if (r.lo == null) {
        row.appendChild(track);
        card.appendChild(row);
        card.appendChild(el("div", { class: "range-abstained" }, "abstained — no numeric range"));
        return;
      }

      const left = pct(r.lo);
      const width = Math.max(pct(r.hi) - left, 1.5);
      const fill = el("div", { class: "range-fill" });
      fill.style.left = left + "%";
      fill.style.width = width + "%";
      fill.style.background = `var(${MODEL_COLOR_VAR[model]})`;
      const point = el("div", { class: "range-point" });
      point.style.left = pct(r.point) + "%";
      track.appendChild(fill);
      track.appendChild(point);
      row.appendChild(track);
      card.appendChild(row);

      const values = el("div", { class: "range-values" });
      values.appendChild(el("span", {}, fmtMoney(r.lo, 0)));
      values.appendChild(el("span", {}, fmtMoney(r.point, 0)));
      values.appendChild(el("span", {}, fmtMoney(r.hi, 0)));
      card.appendChild(values);
    });

    container.appendChild(card);
  });
}

/* Inline SVG: actual price line (solid) with predicted bands (backtest muted, live accent).
   Every decorative child is aria-hidden — the SVG's own role="img" + aria-label is the single
   thing exposed to assistive tech and plain-text extraction, so numbers from separately
   positioned <text> nodes never get read/extracted as one run-together string. */
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

  const decor = (elem) => { elem.setAttribute("aria-hidden", "true"); return elem; };

  /* gridlines + y labels */
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = yLo + ((yHi - yLo) * i) / ticks;
    const yy = y(v);
    svg.appendChild(decor(el("line", {
      x1: padL, y1: yy, x2: W - padR, y2: yy,
      stroke: "var(--line)", "stroke-width": "1",
    })));
    svg.appendChild(decor(el("text", {
      x: padL - 8, y: yy + 4, "text-anchor": "end",
      "font-size": "11", "font-family": "var(--mono)", fill: "var(--muted)",
      // SVG <text> siblings render at absolute coordinates with no implicit whitespace
      // between them, so any tool that flattens the SVG to plain text (a screen reader
      // ignoring aria-hidden, a copy-paste, a text-extraction crawler) would otherwise run
      // "$362" straight into the next label with no separator at all. The trailing spaces
      // are invisible on screen (this text is right-aligned) but keep flattened text readable.
    }, "$" + v.toFixed(0) + "   ")));
  }

  /* predicted bands — abstentions (lo/hi null) draw no band at all */
  const bandW = Math.max(plotW / Math.max(rows.length, 1) * 0.6, 4);
  rows.filter((r) => r.lo != null && r.hi != null).forEach((r) => {
    const cx = x(new Date(r.target_date).getTime());
    const isLive = r.mode === "live";
    svg.appendChild(decor(el("rect", {
      x: cx - bandW / 2, y: y(r.hi), width: bandW, height: Math.max(y(r.lo) - y(r.hi), 1),
      fill: "var(--accent)", opacity: isLive ? "0.30" : "0.14",
      stroke: isLive ? "var(--accent)" : "var(--muted)",
      "stroke-width": isLive ? "1.2" : "0.75",
      "stroke-dasharray": isLive ? "none" : "2,2",
      rx: "2",
    })));
  });

  /* actual price line + points, where known */
  const known = rows.filter((r) => r.actual != null);
  if (known.length > 0) {
    const pathD = known
      .map((r, i) => `${i === 0 ? "M" : "L"} ${x(new Date(r.target_date).getTime())} ${y(r.actual)}`)
      .join(" ");
    svg.appendChild(decor(el("path", {
      d: pathD, fill: "none", stroke: "var(--text)", "stroke-width": "1.75",
    })));
    known.forEach((r) => {
      svg.appendChild(decor(el("circle", {
        cx: x(new Date(r.target_date).getTime()), cy: y(r.actual), r: "2.6",
        fill: "var(--text)",
      })));
    });
  }

  /* x-axis labels: first, middle, last */
  const labelRows = rows.length > 1
    ? [rows[0], rows[Math.floor(rows.length / 2)], rows[rows.length - 1]]
    : rows;
  labelRows.forEach((r) => {
    svg.appendChild(decor(el("text", {
      x: x(new Date(r.target_date).getTime()), y: H - 10, "text-anchor": "middle",
      "font-size": "11", "font-family": "var(--mono)", fill: "var(--muted)",
      // Symmetric padding so flattened text stays readable (see the y-label comment above)
      // without visibly shifting the centered date — the padding is the same on both sides.
    }, "  " + r.target_date + "  ")));
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
  legend.appendChild(item("var(--surface-3)", true, "backtest predicted band"));
  node.appendChild(legend);
}

function render(data) {
  initTabs();

  /* ---- headline ---- */
  const liveModels = Object.values(data.live_summary);
  const anyScored = liveModels.some((s) => s.n_scored > 0);
  document.getElementById("answer-body").textContent = anyScored
    ? `Coverage and interval score for each arm are broken out below — see "Live track record".`
    : `No live prediction has matured yet (horizon is ${data.horizon_days} trading days). ` +
      `${data.open_predictions.length} open prediction(s) are already committed and waiting.`;

  document.getElementById("footer-generated").textContent =
    "generated " + new Date(data.generated_utc).toLocaleString();

  /* ---- open predictions, grouped by ticker ---- */
  tickerCards(document.getElementById("ticker-grid"), data.open_predictions);

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
