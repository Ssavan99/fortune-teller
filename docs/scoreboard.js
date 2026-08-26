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

/* ---- open predictions: one card per (ticker, forecast cycle), all models overlaid ----
   Grouping by symbol ALONE is wrong: the monthly cadence with a ~21-trading-day horizon means
   a new cycle's predictions are made before the previous cycle's have matured, so more than
   one cycle is routinely open for the same ticker at once (e.g. an Aug-25 cycle targeting
   ~Sep-22 is still open when the Sep-1 run adds a new cycle targeting ~Sep-29). Grouping must
   include the cycle key (as_of) too, or a newer cycle's rows silently vanish behind an older
   one's. */
function tickerCards(container, openPredictions) {
  container.innerHTML = "";
  const byCycle = {};
  openPredictions.forEach((r) => {
    const key = r.symbol + "|" + r.as_of;
    (byCycle[key] = byCycle[key] || []).push(r);
  });

  const keys = Object.keys(byCycle).sort((a, b) => {
    const [symA, asOfA] = a.split("|");
    const [symB, asOfB] = b.split("|");
    if (symA !== symB) return symA < symB ? -1 : 1;
    return asOfB.localeCompare(asOfA); // newest cycle first within a ticker
  });
  if (keys.length === 0) {
    container.appendChild(el("p", { class: "stat-empty" }, "No open predictions right now."));
    return;
  }

  keys.forEach((key) => {
    const rows = byCycle[key];
    const symbol = rows[0].symbol;
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

/* ---- findings summary: three at-a-glance cards, price / volatility / calibration ---- */
function findingsSummary(node, findings) {
  node.innerHTML = "";
  const cal = findings.calibration;
  const vol = findings.volatility;
  const dir = findings.direction;

  const cards = [
    {
      status: "bad",
      label: "Price direction (21-day)",
      headline: "No skill",
      note: `${fmtPct(dir.directional_accuracy, 1)} directional accuracy, Brier ` +
        `${dir.brier_score.model.toFixed(4)} vs a ${dir.brier_score.base_rate_baseline.toFixed(4)} ` +
        `base-rate benchmark — the rigorous negative control this project is built to produce, ` +
        `not a bug.`,
    },
    {
      status: "good",
      label: "Volatility (21-day)",
      headline: `+${(vol.skill_vs_persistence * 100).toFixed(1)}% skill`,
      note: `Beats a persistence-of-volatility baseline on ${vol.tickers_beating_persistence}/` +
        `${vol.n_tickers} tickers — how much a price moves is far more forecastable than ` +
        `which way it moves.`,
    },
    {
      status: "good",
      label: "Interval calibration",
      headline: "80% means ~80%",
      note: `${fmtPct(cal.lstm.after.coverage, 1)} (LSTM) / ` +
        `${fmtPct(cal.persistence.after.coverage, 1)} (persistence) actual coverage at an 80% ` +
        `nominal target, up from ${fmtPct(cal.lstm.before.coverage, 1)} / ` +
        `${fmtPct(cal.persistence.before.coverage, 1)} before a real engineering fix.`,
    },
  ];

  cards.forEach((c) => {
    const card = el("div", { class: `finding-card ${c.status}` });
    card.appendChild(el("div", { class: "finding-label" }, c.label));
    card.appendChild(el("div", { class: "finding-headline" }, c.headline));
    card.appendChild(el("div", { class: "finding-note" }, c.note));
    node.appendChild(card);
  });
}

/* ---- negative control: direction scored as a probability, not bare accuracy ---- */
function negativeControlCards(node, direction) {
  node.innerHTML = "";
  const cards = [
    {
      k: "Directional accuracy",
      v: fmtPct(direction.directional_accuracy, 1),
      note: `historical base rate up: ${fmtPct(direction.base_rate_up, 1)}`,
    },
    {
      k: "Brier score (lower is better)",
      v: direction.brier_score.model.toFixed(4),
      note: `base-rate baseline: ${direction.brier_score.base_rate_baseline.toFixed(4)}`,
    },
    {
      k: "Log loss (lower is better)",
      v: direction.log_loss.model.toFixed(4),
      note: `base-rate baseline: ${direction.log_loss.base_rate_baseline.toFixed(4)}`,
    },
  ];
  cards.forEach((c) => {
    const card = el("div", { class: "card" });
    card.appendChild(el("div", { class: "k" }, c.k));
    card.appendChild(el("div", { class: "v neg" }, c.v));
    card.appendChild(el("div", { class: "note" }, c.note));
    node.appendChild(card);
  });
}

/* ---- calibration rows: nominal vs actual coverage, before/after, per model.
   Coverage is never rendered without width + interval score alongside it — see the section
   copy for why coverage alone is a gameable number. ---- */
function calibrationRows(node, calibration) {
  node.innerHTML = "";
  const nominal = calibration.nominal_level;
  const [bandLo, bandHi] = calibration.pre_registered_success_band;
  const pct = (v) => (v * 100) + "%";

  function row(phase, d) {
    const r = el("div", { class: "calib-row" });
    r.appendChild(el("span", { class: "calib-tag" }, phase === "before" ? "Before" : "After"));

    const track = el("div", { class: "calib-track" });
    const band = el("div", { class: "calib-band" });
    band.style.left = pct(bandLo);
    band.style.width = pct(bandHi - bandLo);
    track.appendChild(band);

    const nomLine = el("div", { class: "calib-nominal" });
    nomLine.style.left = pct(nominal);
    track.appendChild(nomLine);

    const dot = el("div", { class: `calib-dot ${phase}` });
    dot.style.left = pct(d.coverage);
    track.appendChild(dot);
    r.appendChild(track);

    const readout = el("div", { class: "calib-readout" });
    readout.appendChild(el("strong", {}, fmtPct(d.coverage, 1) + " coverage"));
    readout.appendChild(document.createTextNode(
      ` · width ${fmtMoney(d.mean_width, 0)} · score ${d.interval_score.toFixed(1)}`
    ));
    r.appendChild(readout);
    return r;
  }

  ["lstm", "persistence"].forEach((model) => {
    const block = el("div", { class: "calib-block" });
    block.appendChild(el("div", { class: "calib-block-label" }, MODEL_LABEL[model]));
    block.appendChild(row("before", calibration[model].before));
    block.appendChild(row("after", calibration[model].after));
    node.appendChild(block);
  });
}

/* ---- volatility: skill per ticker (bar chart), all should be positive ---- */
function volatilitySkillChart(node, perTicker) {
  node.innerHTML = "";
  const entries = Object.entries(perTicker)
    .map(([symbol, v]) => [symbol, v.skill_vs_persistence])
    .sort((a, b) => b[1] - a[1]);

  if (entries.length === 0) {
    node.appendChild(el("p", { class: "stat-empty" }, "No volatility evaluation yet."));
    return;
  }

  const W = 900, H = 300, padL = 44, padR = 16, padT = 16, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxV = Math.max(...entries.map(([, v]) => v)) * 1.2;

  const decor = (elem) => { elem.setAttribute("aria-hidden", "true"); return elem; };
  const y = (v) => padT + plotH - (v / maxV) * plotH;
  const slot = plotW / entries.length;
  const barW = Math.max(slot * 0.6, 4);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.setAttribute("aria-label",
    "Volatility forecast skill versus a persistence-of-volatility baseline, per ticker; " +
    "positive means the model beat the baseline");

  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = (maxV * i) / ticks;
    const yy = y(v);
    svg.appendChild(decor(el("line", {
      x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "var(--line)", "stroke-width": "1",
    })));
    svg.appendChild(decor(el("text", {
      x: padL - 8, y: yy + 4, "text-anchor": "end",
      "font-size": "11", "font-family": "var(--mono)", fill: "var(--muted)",
    }, (v * 100).toFixed(0) + "%   ")));
  }

  entries.forEach(([symbol, v], i) => {
    const cx = padL + slot * (i + 0.5);
    const barTop = y(v);
    svg.appendChild(decor(el("rect", {
      x: cx - barW / 2, y: barTop, width: barW, height: Math.max(y(0) - barTop, 1),
      fill: "var(--accent)", opacity: "0.85", rx: "2",
    })));
    svg.appendChild(decor(el("text", {
      x: cx, y: barTop - 6, "text-anchor": "middle",
      "font-size": "10", "font-family": "var(--mono)", fill: "var(--muted)",
    }, (v * 100).toFixed(1) + "%")));
    svg.appendChild(decor(el("text", {
      x: cx, y: H - 10, "text-anchor": "middle",
      "font-size": "10.5", "font-family": "var(--mono)", fill: "var(--muted)",
    }, "  " + symbol + "  ")));
  });

  node.appendChild(svg);
}

/* ---- volatility: predicted vs realized, over time, for one ticker ---- */
function volatilitySeriesChart(node, rows) {
  node.innerHTML = "";
  if (!rows || rows.length === 0) {
    node.appendChild(el("p", { class: "stat-empty" }, "No volatility series for this ticker."));
    return;
  }

  const W = 900, H = 300, padL = 48, padR = 16, padT = 16, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const xs = rows.map((r) => new Date(r.date).getTime());
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const xSpan = Math.max(xMax - xMin, 1);

  const values = rows.flatMap((r) => [r.baseline_forecast, r.model_forecast, r.actual]);
  const yMin = 0, yMax = Math.max(...values) * 1.08;

  const x = (t) => padL + ((t - xMin) / xSpan) * plotW;
  const y = (v) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const decor = (elem) => { elem.setAttribute("aria-hidden", "true"); return elem; };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.setAttribute("aria-label",
    "Predicted versus realized 21-day forward volatility over time for the selected ticker");

  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = (yMax * i) / ticks;
    const yy = y(v);
    svg.appendChild(decor(el("line", {
      x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "var(--line)", "stroke-width": "1",
    })));
    svg.appendChild(decor(el("text", {
      x: padL - 8, y: yy + 4, "text-anchor": "end",
      "font-size": "11", "font-family": "var(--mono)", fill: "var(--muted)",
    }, (v * 100).toFixed(0) + "%   ")));
  }

  function line(key, color, dashed, width) {
    const d = rows.map((r, i) => `${i === 0 ? "M" : "L"} ${x(new Date(r.date).getTime())} ${y(r[key])}`).join(" ");
    const attrs = { d, fill: "none", stroke: color, "stroke-width": String(width) };
    if (dashed) attrs["stroke-dasharray"] = "4,3";
    svg.appendChild(decor(el("path", attrs)));
  }
  line("baseline_forecast", "var(--muted-2)", true, 1.5);
  line("model_forecast", "var(--accent)", false, 2);
  line("actual", "var(--text)", false, 1.75);

  const labelRows = rows.length > 1
    ? [rows[0], rows[Math.floor(rows.length / 2)], rows[rows.length - 1]]
    : rows;
  labelRows.forEach((r) => {
    svg.appendChild(decor(el("text", {
      x: x(new Date(r.date).getTime()), y: H - 10, "text-anchor": "middle",
      "font-size": "11", "font-family": "var(--mono)", fill: "var(--muted)",
    }, "  " + r.date + "  ")));
  });

  node.appendChild(svg);

  const legend = el("div", { class: "chart-legend" });
  const item = (color, dashed, label) => {
    const span = el("span");
    const sw = el("span", { class: "swatch" });
    sw.style.background = color;
    if (dashed) sw.style.border = "1px dashed var(--muted-2)";
    span.appendChild(sw);
    span.appendChild(document.createTextNode(label));
    return span;
  };
  legend.appendChild(item("var(--text)", false, "realized (actual)"));
  legend.appendChild(item("var(--accent)", false, "EWMA forecast"));
  legend.appendChild(item("var(--muted-2)", true, "persistence-of-volatility baseline"));
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

  /* ---- findings summary + negative control + calibration: static-shaped, driven by data.findings ---- */
  findingsSummary(document.getElementById("findings-cards"), data.findings);
  negativeControlCards(document.getElementById("negative-control-cards"), data.findings.direction);
  calibrationRows(document.getElementById("calibration-rows"), data.findings.calibration);

  /* ---- volatility: skill by ticker (static) + predicted-vs-realized (ticker-selectable) ---- */
  volatilitySkillChart(document.getElementById("chart-vol-skill"), data.findings.volatility.per_ticker);
  const volTickerSelect = document.getElementById("vol-ticker-select");
  volTickerSelect.innerHTML = "";
  data.tickers.forEach((t) => volTickerSelect.appendChild(el("option", { value: t }, t)));
  function drawVol() {
    volatilitySeriesChart(
      document.getElementById("chart-vol-series"),
      data.findings.volatility.series[volTickerSelect.value] || []
    );
  }
  volTickerSelect.addEventListener("change", drawVol);
  drawVol();

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
