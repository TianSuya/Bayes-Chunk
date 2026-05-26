const state = {
  cases: [],
  activeCase: 0,
  hoverChunk: null,
};

const els = {
  caseSelect: document.getElementById("caseSelect"),
  caseQuestion: document.getElementById("caseQuestion"),
  curveCaption: document.getElementById("curveCaption"),
  chart: document.getElementById("surprisalChart"),
  bayesCaption: document.getElementById("bayesCaption"),
  fixedCaption: document.getElementById("fixedCaption"),
  bayesChunks: document.getElementById("bayesChunks"),
  fixedChunks: document.getElementById("fixedChunks"),
};

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function cleanText(value) {
  return String(value)
    .replace(/\uFFFD/g, "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?%)\]])/g, "$1")
    .replace(/([(\[])\s+/g, "$1")
    .trim();
}

function renderText(value) {
  return escapeHtml(cleanText(value));
}

function currentCase() {
  return state.cases[state.activeCase];
}

function boundaryForChunk(item, chunkIndex) {
  if (chunkIndex <= 0) return 0;
  return item.bayes_boundaries[chunkIndex - 1] ?? null;
}

function renderSelector() {
  els.caseSelect.innerHTML = state.cases
    .map(
      (item, index) =>
        `<button class="case-option${index === state.activeCase ? " active" : ""}" type="button" data-case="${index}">
          <strong>${escapeHtml(item.title)}</strong>
        </button>`,
    )
    .join("");

  els.caseSelect.querySelectorAll("[data-case]").forEach((node) => {
    node.addEventListener("click", () => {
      state.activeCase = Number(node.dataset.case);
      state.hoverChunk = null;
      renderAll();
    });
  });
}

function renderMeta() {
  const item = currentCase();
  els.caseQuestion.textContent = cleanText(item.question);
  els.bayesCaption.textContent = `${item.bayes_segments.length} adaptive chunks`;
  els.fixedCaption.textContent = `${item.fixed_segments.length} fixed-length chunks`;
}

function renderBayesChunks() {
  const item = currentCase();
  els.bayesChunks.innerHTML = item.bayes_segments
    .map((segment, index) => {
      const active = state.hoverChunk === index ? " active-step" : "";
      return `<span class="chunk bayes${active}" data-bayes-chunk="${index}">${renderText(segment)}</span>`;
    })
    .join(" ");

  els.bayesChunks.querySelectorAll("[data-bayes-chunk]").forEach((node) => {
    node.addEventListener("mouseenter", () => {
      state.hoverChunk = Number(node.dataset.bayesChunk);
      renderChart();
      renderBayesChunks();
    });
    node.addEventListener("mouseleave", () => {
      state.hoverChunk = null;
      renderChart();
      renderBayesChunks();
    });
  });
}

function renderFixedChunks() {
  const item = currentCase();
  els.fixedChunks.innerHTML = item.fixed_segments
    .map((segment) => `<span class="chunk fixed">${renderText(segment)}</span>`)
    .join(" ");
}

function pointPath(points) {
  return points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
}

function renderChart() {
  const item = currentCase();
  const values = item.surprisal;
  const width = 1040;
  const height = 360;
  const pad = { left: 56, right: 30, top: 28, bottom: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const max = Math.max(...values, 1);
  const xFor = (i) => pad.left + (i / Math.max(values.length - 1, 1)) * plotW;
  const yFor = (v) => pad.top + plotH - (v / max) * plotH;
  const points = values.map((v, i) => ({ x: xFor(i), y: yFor(v), value: v }));
  const line = pointPath(points);
  const area = `${line} L ${xFor(values.length - 1)} ${pad.top + plotH} L ${xFor(0)} ${pad.top + plotH} Z`;
  const hoverBoundary = state.hoverChunk === null ? null : boundaryForChunk(item, state.hoverChunk);

  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => {
      const y = pad.top + ratio * plotH;
      return `<line class="chart-grid" x1="${pad.left}" x2="${width - pad.right}" y1="${y}" y2="${y}" />`;
    })
    .join("");

  const fixedLines = item.fixed_boundaries
    .map((boundary) => {
      const x = xFor(Math.min(boundary, values.length - 1));
      return `<line class="boundary-fixed boundary-muted" x1="${x}" x2="${x}" y1="${pad.top}" y2="${pad.top + plotH}" />`;
    })
    .join("");

  const bayesLines = item.bayes_boundaries
    .map((boundary, index) => {
      const x = xFor(Math.min(boundary, values.length - 1));
      const isHover = hoverBoundary === boundary;
      const cls = isHover ? "boundary-bayes boundary-hover" : "boundary-bayes";
      const opacity = hoverBoundary === null || isHover ? 0.95 : 0.2;
      return `<line class="${cls}" x1="${x}" x2="${x}" y1="${pad.top}" y2="${pad.top + plotH}" opacity="${opacity}" data-boundary="${index}" />`;
    })
    .join("");

  const boundaryDots = item.bayes_boundaries
    .map((boundary) => {
      const idx = Math.min(boundary, values.length - 1);
      const isHover = hoverBoundary === boundary;
      const r = isHover ? 8 : 4.5;
      const opacity = hoverBoundary === null || isHover ? 1 : 0.25;
      return `<circle class="point-highlight" cx="${xFor(idx)}" cy="${yFor(values[idx])}" r="${r}" opacity="${opacity}" />`;
    })
    .join("");

  let hoverLabel = "";
  if (hoverBoundary !== null) {
    const idx = Math.min(hoverBoundary, values.length - 1);
    const labelX = Math.min(xFor(idx) + 12, width - 220);
    const labelY = Math.max(yFor(values[idx]) - 16, 24);
    hoverLabel = `
      <text class="boundary-label" x="${labelX}" y="${labelY}">
        boundary token ${hoverBoundary}
      </text>
    `;
    els.curveCaption.textContent = `Chunk B${state.hoverChunk + 1} starts from boundary token ${hoverBoundary}.`;
  } else {
    els.curveCaption.textContent = "Hover a Bayes chunk to reveal the boundary that produced it.";
  }

  const labels = [
    `<text x="${pad.left}" y="${height - 17}" fill="#5c6962" font-size="14">0</text>`,
    `<text x="${width - pad.right}" y="${height - 17}" text-anchor="end" fill="#5c6962" font-size="14">${values.length - 1}</text>`,
    `<text x="18" y="30" fill="#5c6962" font-size="14">surprisal</text>`,
  ].join("");

  els.chart.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    ${grid}
    <path class="chart-area" d="${area}"></path>
    <path class="chart-line" d="${line}"></path>
    ${fixedLines}
    ${bayesLines}
    ${boundaryDots}
    ${hoverLabel}
    ${labels}
  `;
}

function renderAll() {
  renderSelector();
  renderMeta();
  renderChart();
  renderBayesChunks();
  renderFixedChunks();
}

async function init() {
  let payload = window.ANYEDIT_CASES;
  console.info("AnyEdit++ demo init", {
    embeddedCases: Boolean(payload),
    caseCount: payload && payload.cases ? payload.cases.length : 0,
  });
  if (!payload) {
    const response = await fetch("static/data/cases.json");
    if (!response.ok) {
      throw new Error(`Could not load cases.json: ${response.status}`);
    }
    payload = await response.json();
  }
  if (!payload.cases || payload.cases.length === 0) {
    throw new Error("Demo data has no cases.");
  }
  state.cases = payload.cases;
  renderAll();
}

init().catch((error) => {
  console.error(error);
  els.bayesChunks.textContent = `Could not load demo data: ${error.message}`;
});
