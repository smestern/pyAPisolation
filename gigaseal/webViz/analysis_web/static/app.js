// gigaseal analysis web — frontend controller.
// Plain ES module; no build step.

const $ = (sel) => document.querySelector(sel);
const api = (path, opts = {}) =>
  fetch(path, { credentials: "same-origin", ...opts }).then(async (r) => {
    const text = await r.text();
    let body = {};
    if (text) {
      try {
        body = JSON.parse(text);
      } catch (err) {
        // Server returned non-JSON (or non-spec JSON like NaN/Infinity).
        // Surface it instead of pretending the request succeeded.
        if (r.ok) throw new Error(`invalid JSON from ${path}: ${err.message}`);
      }
    }
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
  });

const state = {
  files: [],
  modules: {},
  currentFile: null,
  currentTrace: null,
  selectedSweeps: new Set(),
  module: null,
  job: null,
  jobPoll: null,
  table: null,
};

// ---------------------------------------------------------------- init
async function init() {
  const cfg = await api("/api/config");
  state.config = cfg;
  await refreshFiles();
  await loadModules();
}

// ---------------------------------------------------------------- files
async function refreshFiles() {
  const body = await api("/api/files");
  state.files = body.files;
  renderFileList(body);
}

function renderFileList(body) {
  const list = $("#file-list");
  list.innerHTML = "";
  for (const f of body.files) {
    const li = document.createElement("li");
    li.dataset.name = f.name;
    li.innerHTML = `<span>${f.name}</span>
      <span>
        <span class="size">${f.size_mb} MB</span>
        <button class="del" title="remove">×</button>
      </span>`;
    if (state.currentFile === f.name) li.classList.add("active");
    li.addEventListener("click", (ev) => {
      if (ev.target.classList.contains("del")) return;
      selectFile(f.name);
    });
    li.querySelector(".del").addEventListener("click", async (ev) => {
      ev.stopPropagation();
      try {
        await api(`/api/files/${encodeURIComponent(f.name)}`, { method: "DELETE" });
        if (state.currentFile === f.name) {
          state.currentFile = null;
          state.currentTrace = null;
          renderPlot();
        }
        await refreshFiles();
      } catch (e) { alert(e.message); }
    });
    list.appendChild(li);
  }
  renderQuota(body.quota);
  updateRunButtons();
}

function renderQuota(q) {
  const el = $("#quota");
  if (!q || q.files_max == null) {
    el.textContent = `${q?.files_used ?? 0} files loaded`;
    return;
  }
  const mbUsed = (q.bytes_used / 1024 / 1024).toFixed(1);
  el.textContent =
    `${q.files_used} / ${q.files_max} files · ${mbUsed} MB used · ` +
    `${q.max_file_size_mb} MB per file`;
}

$("#file-input")?.addEventListener("change", async (ev) => {
  const files = ev.target.files;
  if (!files || !files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  try {
    await api("/api/files", { method: "POST", body: form });
    ev.target.value = "";
    await refreshFiles();
  } catch (e) {
    alert(e.message);
    ev.target.value = "";
  }
});

$("#demo-btn")?.addEventListener("click", async () => {
  try {
    await api("/api/files/demo", { method: "POST" });
    await refreshFiles();
  } catch (e) { alert(e.message); }
});

// ---------------------------------------------------------------- trace
async function selectFile(name) {
  state.currentFile = name;
  state.currentTrace = null;
  document.querySelectorAll("#file-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.name === name));
  try {
    const payload = await api(`/api/trace/${encodeURIComponent(name)}`);
    state.currentTrace = payload;
    state.selectedSweeps = new Set(
      payload.sweeps.slice(0, Math.min(10, payload.sweep_count)).map((s) => s.index)
    );
    renderSweepPicker();
    renderPlot();
  } catch (e) {
    $("#plot").innerHTML = `<p class="muted">${e.message}</p>`;
  }
  updateRunButtons();
}

function renderSweepPicker() {
  const el = $("#sweep-picker");
  el.innerHTML = "";
  if (!state.currentTrace) return;
  for (const sw of state.currentTrace.sweeps) {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selectedSweeps.has(sw.index);
    cb.addEventListener("change", () => {
      if (cb.checked) state.selectedSweeps.add(sw.index);
      else state.selectedSweeps.delete(sw.index);
      renderPlot();
    });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(` ${sw.index}`));
    el.appendChild(lab);
  }
}

function renderPlot() {
  if (typeof Plotly === "undefined") {
    setTimeout(renderPlot, 200);
    return;
  }
  const div = $("#plot");
  if (!state.currentTrace) {
    Plotly.purge(div);
    return;
  }
  const showCmd = $("#show-command").checked;
  const traces = [];
  const t = state.currentTrace.time;
  for (const sw of state.currentTrace.sweeps) {
    if (!state.selectedSweeps.has(sw.index)) continue;
    traces.push({
      x: t, y: sw.voltage, mode: "lines", name: `sweep ${sw.index}`,
      line: { width: 1 },
    });
    if (showCmd) {
      traces.push({
        x: t, y: sw.command, mode: "lines", name: `cmd ${sw.index}`,
        yaxis: "y2", line: { width: 1, dash: "dot" },
      });
    }
  }
  // best-effort spike overlay from current job
  if ($("#show-spikes").checked && state.job?.status === "done") {
    appendSpikeMarkers(traces);
  }
  const layout = {
    margin: { l: 40, r: 40, t: 10, b: 30 },
    paper_bgcolor: "#181b22", plot_bgcolor: "#181b22",
    font: { color: "#e5e7eb", size: 11 },
    xaxis: { title: "time (s)", gridcolor: "#2a2f3a" },
    yaxis: { title: "mV", gridcolor: "#2a2f3a" },
    showlegend: false,
  };
  if (showCmd) {
    layout.yaxis2 = { overlaying: "y", side: "right", showgrid: false, title: "cmd" };
  }
  Plotly.react(div, traces, layout, { responsive: true, displaylogo: false });
}

function appendSpikeMarkers(traces) {
  if (!state.job?.preview) return;
  // Heuristic: look for columns named threshold_time / peak_t / spike_t.
  const candidates = ["threshold_t", "threshold_time", "peak_t", "peak_time", "spike_time"];
  const matchesCurrent = (r) => {
    const file = typeof r.file === "string" ? r.file : "";
    const filename = typeof r.filename === "string" ? r.filename : "";
    return (
      file === state.currentFile ||
      file.endsWith(state.currentFile) ||
      filename === state.currentFile ||
      filename.endsWith(state.currentFile)
    );
  };
  const rows = state.job.preview.filter(matchesCurrent);
  if (!rows.length) return;
  const col = candidates.find((c) => c in rows[0]);
  if (!col) return;
  const xs = [];
  for (const r of rows) {
    const v = r[col];
    if (v != null && !isNaN(v)) xs.push(Number(v));
  }
  if (!xs.length) return;
  traces.push({
    x: xs, y: xs.map(() => 0), mode: "markers",
    marker: { color: "#34d399", size: 6, symbol: "triangle-up" },
    name: "spikes",
  });
}

$("#show-command")?.addEventListener("change", renderPlot);
$("#show-spikes")?.addEventListener("change", renderPlot);

// ---------------------------------------------------------------- modules
async function loadModules() {
  state.modules = await api("/api/modules");
  const sel = $("#module-select");
  sel.innerHTML = "";
  for (const name of Object.keys(state.modules).sort()) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = state.modules[name].display_name || name;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => selectModule(sel.value));
  if (sel.options.length) selectModule(sel.value);
}

function selectModule(name) {
  state.module = state.modules[name];
  $("#module-doc").textContent = state.module?.doc || "";
  const form = $("#param-form");
  form.innerHTML = "";
  for (const p of state.module.parameters) {
    const label = document.createElement("label");
    label.textContent = p.name.replace(/_/g, " ");
    const input = document.createElement("input");
    input.name = p.name;
    input.dataset.type = p.type;
    if (p.type === "bool") {
      input.type = "checkbox";
      input.checked = Boolean(p.value);
    } else if (p.type === "int" || p.type === "float") {
      input.type = "number";
      input.step = p.type === "float" ? "any" : "1";
      if (p.value != null) input.value = p.value;
    } else {
      input.type = "text";
      if (p.value != null) input.value = p.value;
    }
    label.appendChild(input);
    form.appendChild(label);
  }
  updateRunButtons();
}

function collectParams() {
  const params = {};
  for (const input of $("#param-form").querySelectorAll("input")) {
    const t = input.dataset.type;
    if (t === "bool") params[input.name] = input.checked;
    else if (t === "int") params[input.name] = input.value === "" ? null : parseInt(input.value, 10);
    else if (t === "float") params[input.name] = input.value === "" ? null : parseFloat(input.value);
    else params[input.name] = input.value;
    if (params[input.name] == null || Number.isNaN(params[input.name])) delete params[input.name];
  }
  return params;
}

// ---------------------------------------------------------------- jobs
function updateRunButtons() {
  const hasFiles = state.files.length > 0;
  $("#run-batch-btn").disabled = !hasFiles || !state.module;
  $("#run-current-btn").disabled = !state.currentFile || !state.module;
}

$("#run-current-btn")?.addEventListener("click", () => {
  if (!state.currentFile) return;
  submitJob([state.currentFile]);
});
$("#run-batch-btn")?.addEventListener("click", () => {
  submitJob(state.files.map((f) => f.name));
});

async function submitJob(files) {
  const payload = {
    module: state.module.name,
    params: collectParams(),
    files,
  };
  setJobStatus("queued…", "");
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.job = job;
    pollJob(job.job_id);
  } catch (e) {
    setJobStatus(e.message, "error");
  }
}

function pollJob(jobId) {
  if (state.jobPoll) clearInterval(state.jobPoll);
  state.jobPoll = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      state.job = job;
      const pct = Math.round((job.progress || 0) * 100);
      setJobStatus(
        `${job.status} — ${job.completed}/${job.total} (${pct}%)`,
        job.status === "error" ? "error" : (job.status === "done" ? "ok" : ""),
        job.progress
      );
      if (job.status === "done" || job.status === "error") {
        clearInterval(state.jobPoll);
        state.jobPoll = null;
        if (job.status === "done") {
          renderResults(job);
          renderPlot();  // re-render to pick up spike overlay
        } else if (job.error) {
          setJobStatus(job.error.split("\n")[0], "error");
        }
      }
    } catch (e) {
      clearInterval(state.jobPoll);
      state.jobPoll = null;
      setJobStatus(e.message, "error");
    }
  }, 500);
}

function setJobStatus(text, cls, progress) {
  const el = $("#job-status");
  el.className = "job-status " + (cls || "");
  el.innerHTML = `<div>${text}</div>` +
    (progress != null
      ? `<div class="progress"><div style="width:${Math.round(progress*100)}%"></div></div>`
      : "");
}

// ---------------------------------------------------------------- results
function renderResults(job) {
  $("#row-count").textContent = `${job.row_count} rows`;
  $("#export-csv").disabled = false;
  $("#export-xlsx").disabled = false;
  $("#export-csv").onclick = () =>
    window.location.href = `/api/jobs/${job.job_id}/export.csv`;
  $("#export-xlsx").onclick = () =>
    window.location.href = `/api/jobs/${job.job_id}/export.xlsx`;

  // Some analysis modules return nested dict-of-dict cells (e.g. spike
  // module emits ``spike_df.to_dict()`` → {col: {row_idx: val}}). Render
  // those as compact JSON so the table is readable instead of showing
  // "[object Object]".
  const cellFormatter = (cell) => {
    const v = cell.getValue();
    if (v == null) return "";
    if (typeof v === "object") {
      // Unwrap single-key dicts (very common — pandas {0: value}).
      const keys = Object.keys(v);
      if (!Array.isArray(v) && keys.length === 1) {
        const inner = v[keys[0]];
        return inner == null ? "" : String(inner);
      }
      try { return JSON.stringify(v); } catch { return String(v); }
    }
    return v;
  };
  const cols = (job.columns || []).map((c) => ({
    title: c, field: c, headerSort: true, resizable: true,
    formatter: cellFormatter,
  }));
  if (state.table) { state.table.destroy(); state.table = null; }
  if (typeof Tabulator === "undefined") {
    setTimeout(() => renderResults(job), 200);
    return;
  }
  state.table = new Tabulator("#results-table", {
    data: job.preview,
    columns: cols,
    layout: "fitDataStretch",
    height: "100%",
    placeholder: "no rows",
  });
}

init().catch((e) => {
  console.error(e);
  alert("startup failed: " + e.message);
});
