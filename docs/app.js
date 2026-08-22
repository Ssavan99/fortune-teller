/* Renders the results page from docs/data/report.json.
   Every figure shown comes from the committed experiment output — nothing is hard-coded. */

const fmt = (v, d = 4) => (v >= 0 ? "+" : "") + v.toFixed(d);
const pct = (v, d = 2) => v.toFixed(d) + "%";
const sign = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");

function el(tag, attrs = {}, text) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}

function table(node, head, rows, foot) {
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
      const isObj = c && typeof c === "object";
      const td = el("td", { class: i === 0 ? "" : "num " + (isObj ? sign(c.v) : "") });
      td.textContent = isObj ? c.t : c;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  node.appendChild(tbody);

  if (foot) {
    const tfoot = el("tfoot");
    const tr = el("tr");
    foot.forEach((c, i) => {
      const td = el("td", { class: i === 0 ? "" : "num" });
      td.textContent = c;
      tr.appendChild(td);
    });
    tfoot.appendChild(tr);
    node.appendChild(tfoot);
  }
}

/* Horizontal bar chart of per-ticker skill scores, drawn as inline SVG. */
function skillChart(node, entries) {
  const W = 900, rowH = 26, padL = 62, padR = 70, padT = 28, padB = 8;
  const H = padT + entries.length * rowH + padB;
  const max = Math.max(...entries.map((e) => Math.abs(e.value)), 0.005);
  const mid = padL + (W - padL - padR) / 2;
  const half = (W - padL - padR) / 2;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.setAttribute("aria-label", "Per-ticker skill score against persistence");

  svg.appendChild(
    el("text", { x: mid, y: 16, "text-anchor": "middle", "font-size": "12",
                 fill: "var(--muted)" }, "skill score vs persistence  (right of line = better)")
  );
  const axis = el("line", { x1: mid, y1: padT - 6, x2: mid, y2: H - padB,
                            stroke: "var(--line)", "stroke-width": "1" });
  svg.appendChild(axis);

  entries.forEach((e, i) => {
    const y = padT + i * rowH;
    const w = (Math.abs(e.value) / max) * half * 0.92;
    const x = e.value >= 0 ? mid : mid - w;
    svg.appendChild(el("rect", {
      x, y: y + 5, width: Math.max(w, 1), height: rowH - 12, rx: 2,
      fill: e.value >= 0 ? "var(--good)" : "var(--bad)", opacity: "0.85",
    }));
    svg.appendChild(el("text", {
      x: padL - 10, y: y + rowH / 2 + 4, "text-anchor": "end",
      "font-size": "12.5", "font-family": "var(--mono)", fill: "var(--text)",
    }, e.label));
    svg.appendChild(el("text", {
      x: W - padR + 8, y: y + rowH / 2 + 4, "font-size": "12",
      "font-family": "var(--mono)", fill: "var(--muted)",
    }, fmt(e.value)));
  });

  node.innerHTML = "";
  node.appendChild(svg);
}

function render(data) {
  const s = data.summary;
  const a = data.experiment_a;
  const ret = a.arms.return;
  const lvl = a.arms.level;

  /* ---- headline ---- */
  const verdict = s.headline_skill > 0 ? "beats" : "does not beat";
  document.getElementById("answer-body").innerHTML =
    `Over <strong>${s.period.start} to ${s.period.end}</strong> the LSTM ` +
    `<strong>${verdict}</strong> persistence. Its best parameterisation scores a mean skill of ` +
    `<strong>${fmt(s.headline_skill)}</strong> and wins on ` +
    `<strong>${s.headline_beats} of ${s.headline_n}</strong> tickers — ` +
    `${pct(s.lstm_mape, 3)} mean error against the baseline's ${pct(s.persistence_mape, 3)}. ` +
    `Predicting price levels directly is far worse, at ${fmt(s.level_skill, 2)}.`;

  document.getElementById("footer-period").textContent =
    `held-out ${s.period.start} → ${s.period.end}`;

  /* ---- method cards ---- */
  const cfg = ret.config;
  const cards = [
    ["Held-out period", `${s.period.start} → ${s.period.end}`],
    ["Scored rows", a.period.rows.toLocaleString()],
    ["Tickers", String(s.headline_n)],
    ["Architecture", `Bi-LSTM ${cfg.hidden}×${cfg.layers}`],
    ["Lookback", `${cfg.lookback} days`],
    ["Baseline", "close[t−1]"],
  ];
  const method = document.getElementById("method");
  method.innerHTML = "";
  cards.forEach(([k, v]) => {
    const c = el("div", { class: "card" });
    c.appendChild(el("div", { class: "k" }, k));
    c.appendChild(el("div", { class: "v" }, v));
    method.appendChild(c);
  });

  /* ---- experiment A ---- */
  const tickers = Object.keys(ret.per_ticker);
  skillChart(
    document.getElementById("chart-a"),
    tickers.map((t) => ({ label: t, value: ret.per_ticker[t].skill_vs_persistence }))
  );

  table(
    document.getElementById("table-a"),
    ["Ticker", "Persistence RMSE", "LSTM (return)", "Skill", "LSTM (level)", "Skill"],
    tickers.map((t) => [
      t,
      "$" + ret.per_ticker[t].persistence.rmse.toFixed(2),
      "$" + ret.per_ticker[t].lstm.rmse.toFixed(2),
      { t: fmt(ret.per_ticker[t].skill_vs_persistence), v: ret.per_ticker[t].skill_vs_persistence },
      "$" + lvl.per_ticker[t].lstm.rmse.toFixed(2),
      { t: fmt(lvl.per_ticker[t].skill_vs_persistence, 2), v: lvl.per_ticker[t].skill_vs_persistence },
    ]),
    ["Mean", "—", "—", fmt(ret.pooled.mean_skill_vs_persistence), "—",
     fmt(lvl.pooled.mean_skill_vs_persistence, 2)]
  );

  /* ---- range table ---- */
  const dl = lvl.diagnostics;
  table(
    document.getElementById("table-range"),
    ["Ticker", "Train max", "Held-out max", "Largest prediction", "Pred ÷ train max"],
    tickers.map((t) => [
      t,
      "$" + dl[t].train_close_max.toFixed(2),
      "$" + dl[t].test_close_max.toFixed(2),
      "$" + dl[t].pred_max.toFixed(2),
      (dl[t].pred_max / dl[t].train_close_max).toFixed(2) + "×",
    ])
  );

  /* ---- move table ---- */
  const dr = ret.diagnostics;
  table(
    document.getElementById("table-moves"),
    ["Ticker", "Predicted move sd", "Actual move sd", "Ratio", "Direction correlation"],
    tickers.map((t) => [
      t,
      "$" + dr[t].pred_move_std.toFixed(2),
      "$" + dr[t].actual_move_std.toFixed(2),
      (dr[t].pred_move_std / dr[t].actual_move_std).toFixed(3),
      { t: dr[t].move_corr === null ? "n/a" : fmt(dr[t].move_corr, 3), v: dr[t].move_corr || 0 },
    ])
  );

  /* ---- experiment B ---- */
  const b = data.experiment_b;
  const bo = b.arms.price_only.per_ticker;
  const bs = b.arms.price_sentiment.per_ticker;
  document.getElementById("callout-b").innerHTML =
    `Adding sentiment changes mean skill by <strong>${fmt(s.sentiment_delta_mean)}</strong>, ` +
    `with a standard deviation across tickers of <strong>${s.sentiment_delta_sd.toFixed(4)}</strong>. ` +
    `The effect is smaller than its own spread, so this is <strong>no detectable effect</strong> — ` +
    `not evidence that sentiment hurts. It helped on ${s.sentiment_helped} of ${s.headline_n} tickers. ` +
    `Neither arm beats persistence.`;

  table(
    document.getElementById("table-b"),
    ["Ticker", "Coverage", "Price only", "+ Sentiment", "Delta"],
    Object.keys(bo).map((t) => {
      const d = bs[t].skill_vs_persistence - bo[t].skill_vs_persistence;
      return [
        t,
        pct(b.coverage[t].coverage * 100, 1),
        { t: fmt(bo[t].skill_vs_persistence), v: bo[t].skill_vs_persistence },
        { t: fmt(bs[t].skill_vs_persistence), v: bs[t].skill_vs_persistence },
        { t: fmt(d), v: d },
      ];
    })
  );

  /* ---- experiment C ---- */
  const c = data.experiment_c;
  const rows = [];
  ["level", "return"].forEach((target) => {
    ["clean", "leaky"].forEach((arm) => {
      const x = c.arms[target][arm];
      rows.push([
        `${target} / ${arm}`,
        x.scaled_rmse.toFixed(5),
        "$" + x.dollar_rmse_mean.toFixed(2),
        { t: fmt(x.mean_skill_vs_persistence, 4), v: x.mean_skill_vs_persistence },
        `${x.tickers_beating_persistence}/${x.n_tickers}`,
      ]);
    });
  });
  table(
    document.getElementById("table-c"),
    ["Target / scaler", "Scaled-unit RMSE", "Mean $ RMSE", "Mean skill", "Beats baseline"],
    rows
  );

  const lc = c.arms.level;
  const rc = c.arms.return;
  const dScaledLevel =
    ((lc.leaky.scaled_rmse - lc.clean.scaled_rmse) / lc.clean.scaled_rmse) * 100;
  const flips =
    rc.clean.mean_skill_vs_persistence < 0 && rc.leaky.mean_skill_vs_persistence > 0;
  document.getElementById("callout-c").innerHTML =
    `The leak alone moves the return arm's mean skill from ` +
    `<strong>${fmt(rc.clean.mean_skill_vs_persistence)}</strong> to ` +
    `<strong>${fmt(rc.leaky.mean_skill_vs_persistence)}</strong>` +
    (flips
      ? ` — <strong>from losing to the baseline to beating it</strong>. A result reported at ` +
        `${fmt(rc.leaky.mean_skill_vs_persistence)} would be reporting the leak, not the model.`
      : `.`) +
    ` On the level arm it cuts scaled-unit RMSE by ` +
    `<strong>${Math.abs(dScaledLevel).toFixed(1)}%</strong> and lifts mean skill from ` +
    `<strong>${fmt(lc.clean.mean_skill_vs_persistence, 2)}</strong> to ` +
    `<strong>${fmt(lc.leaky.mean_skill_vs_persistence, 2)}</strong>. ` +
    `No test row entered training in either case — only the normalisation constants did.`;
}

fetch("data/report.json")
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then(render)
  .catch((err) => {
    document.getElementById("answer-body").textContent =
      "Could not load results: " + err.message;
  });
