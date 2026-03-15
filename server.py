#!/usr/bin/env python3
"""AI File Discovery — Web Dashboard Server.

Zero-dependency web UI for the AI file scanner. Uses Python's stdlib
http.server with an inline single-page app (HTML/CSS/JS).

Usage:
    python3 server.py                    # serve on port 8505
    python3 server.py --port 9000        # custom port
    python3 server.py --scan-path /data  # pre-fill scan path
"""

import argparse
import json
import os
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from aifiles import (
    AIFileScanner,
    FileClassifier,
    ScanConfig,
    format_size,
    format_timestamp,
    _build_json_payload,
    _ensure_report_dir,
    run_integrity_check,
    append_integrity_log,
)

# ---------------------------------------------------------------------------
# HTML Dashboard (inline SPA)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI File Discovery</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #1a1a2e; --bg2: #16213e; --card: #0f3460; --border: #1a3a5c;
    --text: #e0e0e0; --text-dim: #8892a4; --accent: #00d2ff;
    --data: #00d2ff; --model: #e94560; --config: #4ecca3;
    --vector: #7b68ee; --checkpoint: #f0c040;
    --source: #c0c0c0; --document: #8be9fd; --multimedia: #ff79c6; --skill: #ffb86c;
    --agent: #50fa7b; --secret: #ff5555;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }

  /* Header */
  .header { background: var(--bg2); border-bottom: 1px solid var(--border);
            padding: 16px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .header h1 { font-size: 20px; font-weight: 600; white-space: nowrap; }
  .header h1 span { color: var(--accent); }
  .scan-form { display: flex; gap: 8px; flex: 1; min-width: 300px; }
  .scan-form input[type="text"] {
    flex: 1; padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 14px; outline: none;
    transition: border-color .2s;
  }
  .scan-form input:focus { border-color: var(--accent); }
  .btn { padding: 8px 20px; border-radius: 6px; border: none; cursor: pointer;
         font-size: 14px; font-weight: 500; transition: opacity .2s; }
  .btn:hover { opacity: .85; }
  .btn-primary { background: var(--accent); color: #000; }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }

  /* Main layout */
  .main { max-width: 1400px; margin: 0 auto; padding: 24px; }

  /* Summary cards */
  .summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                   gap: 12px; margin-bottom: 24px; }
  .card { background: var(--card); border-radius: 8px; padding: 16px;
          border: 1px solid var(--border); }
  .card-label { font-size: 12px; text-transform: uppercase; color: var(--text-dim);
                letter-spacing: .5px; margin-bottom: 4px; }
  .card-value { font-size: 28px; font-weight: 700; }

  /* Charts row */
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
  .chart-card { background: var(--card); border-radius: 8px; padding: 16px;
                border: 1px solid var(--border); }
  .chart-card h3 { font-size: 14px; margin-bottom: 12px; color: var(--text-dim); }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .bar-label { width: 100px; font-size: 13px; text-align: right; flex-shrink: 0; }
  .bar-track { flex: 1; height: 20px; background: var(--bg); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .4s ease; min-width: 2px; }
  .bar-size { width: 80px; font-size: 13px; color: var(--text-dim); flex-shrink: 0; }

  /* Recommendation banners */
  .rec-banners { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
  .rec-banner { border-radius: 8px; padding: 12px 16px; font-size: 13px; line-height: 1.6;
                display: flex; align-items: flex-start; gap: 10px; }
  .rec-banner .rec-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
  .rec-banner .rec-dismiss { margin-left: auto; background: none; border: none;
                              color: inherit; cursor: pointer; opacity: .5; font-size: 16px; flex-shrink: 0; }
  .rec-banner .rec-dismiss:hover { opacity: 1; }
  .rec-danger { background: rgba(233,69,96,.12); border: 1px solid rgba(233,69,96,.35); color: #f0a0a0; }
  .rec-info { background: rgba(0,210,255,.08); border: 1px solid rgba(0,210,255,.25); color: #a0d8ef; }
  .rec-warn { background: rgba(255,184,108,.10); border: 1px solid rgba(255,184,108,.30); color: #f0d0a0; }

  /* Filters */
  .filters { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .chip { padding: 5px 14px; border-radius: 16px; border: 1px solid var(--border);
          font-size: 13px; cursor: pointer; transition: all .2s; background: transparent; color: var(--text); }
  .chip:hover { border-color: var(--accent); }
  .chip.active { border-color: transparent; color: #000; font-weight: 500; }
  .chip[data-cat="all"].active { background: var(--accent); }
  .chip[data-cat="data"].active { background: var(--data); }
  .chip[data-cat="model"].active { background: var(--model); }
  .chip[data-cat="config"].active { background: var(--config); }
  .chip[data-cat="vector"].active { background: var(--vector); }
  .chip[data-cat="checkpoint"].active { background: var(--checkpoint); }
  .chip[data-cat="source"].active { background: var(--source); }
  .chip[data-cat="document"].active { background: var(--document); color: #000; }
  .chip[data-cat="multimedia"].active { background: var(--multimedia); }
  .chip[data-cat="skill"].active { background: var(--skill); color: #000; }
  .chip[data-cat="agent"].active { background: var(--agent); color: #000; }
  .chip[data-cat="secret"].active { background: var(--secret); }
  .min-size-input { padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border);
                    background: var(--bg); color: var(--text); font-size: 13px; width: 100px; outline: none; }
  .min-size-input:focus { border-color: var(--accent); }

  /* Table */
  .table-wrap { background: var(--card); border-radius: 8px; border: 1px solid var(--border);
                overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th { position: sticky; top: 0; background: var(--bg2); padding: 10px 12px;
             text-align: left; font-weight: 600; cursor: pointer; user-select: none;
             border-bottom: 2px solid var(--border); white-space: nowrap;
             text-decoration: underline dotted rgba(255,255,255,0.3); text-underline-offset: 3px; }
  thead th:hover { color: var(--accent); }
  thead th .sort-arrow { margin-left: 4px; font-size: 10px; }
  tbody td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
  tbody tr:hover { background: rgba(0,210,255,.04); }
  .cat-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
               font-weight: 600; text-transform: uppercase; }
  .cat-data { background: rgba(0,210,255,.15); color: var(--data); }
  .cat-model { background: rgba(233,69,96,.15); color: var(--model); }
  .cat-config { background: rgba(78,204,163,.15); color: var(--config); }
  .cat-vector { background: rgba(123,104,238,.15); color: var(--vector); }
  .cat-checkpoint { background: rgba(240,192,64,.15); color: var(--checkpoint); }
  .cat-source { background: rgba(192,192,192,.15); color: var(--source); }
  .cat-document { background: rgba(139,233,253,.15); color: var(--document); }
  .cat-multimedia { background: rgba(255,121,198,.15); color: var(--multimedia); }
  .cat-skill { background: rgba(255,184,108,.15); color: var(--skill); }
  .cat-agent { background: rgba(80,250,123,.15); color: var(--agent); }
  .cat-secret { background: rgba(255,85,85,.15); color: var(--secret); }
  .conf-h { color: #4ecca3; } .conf-m { color: #f0c040; } .conf-l { color: #e94560; }
  .risk-danger { color: #e94560; font-weight: 700; }
  .risk-warning { color: #f0c040; font-weight: 600; }
  .path-cell { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
               color: var(--text-dim); }
  .path-cell:hover { white-space: normal; word-break: break-all; }

  /* States */
  .empty-state { text-align: center; padding: 60px 20px; color: var(--text-dim); }
  .empty-state h2 { font-size: 18px; margin-bottom: 8px; color: var(--text); }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid var(--border);
             border-top-color: var(--accent); border-radius: 50%; animation: spin .6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .scanning-msg { display: flex; align-items: center; gap: 8px; justify-content: center;
                  padding: 40px; color: var(--text-dim); }

  /* Error toast */
  .errors-bar { background: rgba(233,69,96,.1); border: 1px solid rgba(233,69,96,.3);
                border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; }
  .errors-bar summary { cursor: pointer; color: var(--model); font-weight: 500; }
  .errors-bar ul { margin-top: 8px; padding-left: 20px; color: var(--text-dim); }

  /* Options bar */
  .options-bar { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
  .toggle-btn { padding: 5px 14px; border-radius: 16px; border: 1px solid var(--border);
                font-size: 13px; cursor: pointer; background: transparent; color: var(--text);
                transition: all .2s; }
  .toggle-btn:hover { border-color: var(--accent); }
  .toggle-btn.active { background: var(--accent); color: #000; border-color: transparent; font-weight: 500; }
  .opt-check { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-dim); cursor: pointer; }
  .opt-check input { accent-color: var(--accent); cursor: pointer; }

  /* Tree view */
  .tree-container { background: var(--card); border-radius: 8px; border: 1px solid var(--border);
                    overflow-x: auto; }
  .tree-row { display: flex; align-items: center; padding: 5px 12px;
              border-bottom: 1px solid var(--border); font-size: 13px; min-height: 34px; }
  .tree-row:hover { background: rgba(0,210,255,.04); }
  .tree-folder { cursor: pointer; font-weight: 600; }
  .tree-indent { display: inline-block; width: 20px; flex-shrink: 0; }
  .tree-caret { display: inline-block; width: 16px; font-size: 10px; color: var(--text-dim);
                cursor: pointer; transition: transform .15s; flex-shrink: 0; text-align: center; }
  .tree-caret.open { transform: rotate(90deg); }
  .tree-folder-name { color: var(--accent); margin-right: 8px; }
  .tree-file-name { color: var(--text); margin-right: 8px; }
  .tree-meta { color: var(--text-dim); font-size: 12px; margin-left: auto; display: flex; gap: 12px; align-items: center; }
  .tree-count { font-size: 11px; color: var(--text-dim); background: var(--bg); padding: 1px 6px; border-radius: 8px; }

  /* Browse dialog modal */
  .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.6); z-index: 1000;
                   display: flex; align-items: center; justify-content: center; }
  .modal { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px;
           width: 500px; max-width: 90vw; max-height: 70vh; display: flex; flex-direction: column; }
  .modal-header { display: flex; align-items: center; justify-content: space-between;
                  padding: 14px 18px; border-bottom: 1px solid var(--border); }
  .modal-header h3 { font-size: 15px; font-weight: 600; }
  .modal-close { background: none; border: none; color: var(--text-dim); font-size: 18px;
                 cursor: pointer; padding: 0 4px; }
  .modal-close:hover { color: var(--text); }
  .modal-breadcrumb { padding: 10px 18px; font-size: 13px; color: var(--text-dim);
                      border-bottom: 1px solid var(--border); word-break: break-all; display: flex; align-items: center; gap: 6px; }
  .modal-breadcrumb .modal-up { background: none; border: 1px solid var(--border); color: var(--accent);
                                 font-size: 12px; padding: 2px 8px; border-radius: 4px; cursor: pointer; }
  .modal-breadcrumb .modal-up:hover { border-color: var(--accent); }
  .modal-body { overflow-y: auto; flex: 1; padding: 4px 0; min-height: 200px; }
  .browse-item { padding: 8px 18px; font-size: 13px; cursor: pointer; display: flex; align-items: center; gap: 8px; }
  .browse-item:hover { background: rgba(0,210,255,.06); }
  .browse-icon { flex-shrink: 0; }
  .modal-footer { padding: 12px 18px; border-top: 1px solid var(--border); display: flex;
                  justify-content: flex-end; gap: 8px; }
  .modal-footer .btn-secondary { background: transparent; border: 1px solid var(--border); color: var(--text);
                                  padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .modal-footer .btn-secondary:hover { border-color: var(--accent); }
  .modal-footer .btn-select { background: var(--accent); color: #000; padding: 6px 16px; border-radius: 6px;
                               border: none; cursor: pointer; font-size: 13px; font-weight: 500; }
  .browse-empty { padding: 20px 18px; color: var(--text-dim); font-size: 13px; text-align: center; }
  .btn-browse { background: transparent; border: 1px solid var(--border); color: var(--text-dim);
                padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; white-space: nowrap; }
  .btn-browse:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>

<div class="header">
  <h1><span>AI</span> File Discovery</h1>
  <div class="scan-form">
    <input type="text" id="scanPath" placeholder="/path/to/scan">
    <button class="btn-browse" onclick="openBrowseDialog()">Browse</button>
    <button class="btn btn-primary" id="scanBtn" onclick="runScan()">Scan</button>
  </div>
</div>

<div class="main" id="app">
  <div class="empty-state" id="emptyState">
    <h2>Enter a directory path and click Scan</h2>
    <p>The scanner will find AI/ML files — datasets, models, configs, vector stores, checkpoints, source code, documents, multimedia, and agent skills.</p>
  </div>
</div>

<script>
let scanData = null;
let activeCategory = 'all';
let sortCol = 'size_bytes';
let sortAsc = false;
let minSizeFilter = 0;
let dismissedBanners = {};
let viewMode = 'table';
let optPermissions = false;
let optHashes = false;
let optIntegrity = false;
let expandedPaths = new Set();

const CATEGORY_COLORS = {
  data: '#00d2ff', model: '#e94560', config: '#4ecca3',
  vector: '#7b68ee', checkpoint: '#f0c040',
  source: '#c0c0c0', document: '#8be9fd', multimedia: '#ff79c6', skill: '#ffb86c',
  agent: '#50fa7b', secret: '#ff5555'
};

const ALL_CATS = ['all','data','model','config','vector','checkpoint','source','document','multimedia','skill','agent','secret'];

fetch('/api/default-path').then(r=>r.json()).then(d=>{
  document.getElementById('scanPath').value = d.path || '';

});

function runScan() {
  const path = document.getElementById('scanPath').value.trim();
  if (!path) return;
  const btn = document.getElementById('scanBtn');
  btn.disabled = true;
  btn.textContent = 'Scanning...';
  dismissedBanners = {};
  document.getElementById('app').innerHTML =
    '<div class="scanning-msg"><div class="spinner"></div> Scanning directory\u2026</div>';

  fetch('/api/scan', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({path, compute_permissions: optPermissions, compute_hashes: optHashes, integrity: optIntegrity})
  })
  .then(r => {
    if (!r.ok) return r.json().then(e => { throw new Error(e.error || 'Scan failed'); });
    return r.json();
  })
  .then(data => {
    scanData = data;
    activeCategory = 'all';
    sortCol = 'size_bytes';
    sortAsc = false;
    minSizeFilter = 0;
    render();
  })
  .catch(err => {
    document.getElementById('app').innerHTML =
      `<div class="empty-state"><h2>Error</h2><p>${esc(err.message)}</p></div>`;
  })
  .finally(() => {
    btn.disabled = false;
    btn.textContent = 'Scan';
  });
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  for (const u of ['KB','MB','GB','TB']) {
    bytes /= 1024;
    if (bytes < 1024) return bytes.toFixed(1) + ' ' + u;
  }
  return bytes.toFixed(1) + ' PB';
}

function getFiltered() {
  let files = scanData.files;
  if (activeCategory !== 'all') files = files.filter(f => (f.categories || [f.category]).includes(activeCategory));
  if (minSizeFilter > 0) files = files.filter(f => f.size_bytes >= minSizeFilter);
  files.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });
  return files;
}

function parseMinSize(s) {
  s = s.trim().toUpperCase();
  if (!s) return 0;
  const m = s.match(/^(\d+(?:\.\d+)?)\s*(B|K|KB|M|MB|G|GB|T|TB)?$/);
  if (!m) return 0;
  const v = parseFloat(m[1]);
  const u = m[2] || 'B';
  const mul = {B:1, K:1024, KB:1024, M:1048576, MB:1048576, G:1073741824, GB:1073741824, T:1099511627776, TB:1099511627776};
  return Math.floor(v * (mul[u] || 1));
}

function setCategory(cat) { activeCategory = cat; render(); }

function setSort(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = col === 'filename' || col === 'category'; }
  render();
}

function onMinSize(val) {
  minSizeFilter = parseMinSize(val);
  render();
}

function dismissBanner(id) {
  dismissedBanners[id] = true;
  const el = document.getElementById(id);
  if (el) el.remove();
}

function render() {
  const s = scanData.summary;
  const files = getFiltered();
  const maxCatSize = Math.max(1, ...Object.values(s.by_category).map(c=>c.size));

  const extEntries = Object.entries(s.by_extension)
    .sort((a,b) => b[1].size - a[1].size).slice(0, 8);
  const maxExtSize = Math.max(1, ...extEntries.map(e=>e[1].size));

  const arrow = (col) => sortCol === col ? (sortAsc ? ' \u25B2' : ' \u25BC') : '';

  // Count risks
  const dangerCount = scanData.files.filter(f => f.risk_level === 'danger').length;
  const warnCount = scanData.files.filter(f => f.risk_level === 'warning').length;
  const sourceCount = (s.by_category['source'] || {count:0}).count;
  const skillCount = (s.by_category['skill'] || {count:0}).count;
  const secretCount = (s.by_category['secret'] || {count:0}).count;
  const embeddedSecretCount = scanData.files.filter(f =>
    f.risk_level === 'danger' && !(f.categories || []).includes('secret') &&
    f.risk_reason && f.risk_reason.includes('embedded secret')
  ).length;

  let html = '';

  // Summary cards
  html += `<div class="summary-cards">
    <div class="card"><div class="card-label">Files Found</div><div class="card-value">${s.total_files}</div></div>
    <div class="card"><div class="card-label">Total Size</div><div class="card-value">${formatSize(s.total_size_bytes)}</div></div>
    <div class="card"><div class="card-label">Scan Time</div><div class="card-value">${s.scan_duration_seconds.toFixed(1)}s</div></div>
    <div class="card"><div class="card-label">Errors</div><div class="card-value">${s.errors.length}</div></div>
  </div>`;

  // Errors
  if (s.errors.length) {
    html += `<details class="errors-bar"><summary>${s.errors.length} error(s) during scan</summary>
      <ul>${s.errors.slice(0,20).map(e=>`<li>${esc(e)}</li>`).join('')}</ul></details>`;
  }

  // Charts
  html += '<div class="charts">';
  html += '<div class="chart-card"><h3>Size by Category</h3>';
  for (const [cat, info] of Object.entries(s.by_category).sort((a,b)=>b[1].size-a[1].size)) {
    const pct = (info.size / maxCatSize * 100).toFixed(1);
    const color = CATEGORY_COLORS[cat] || '#888';
    html += `<div class="bar-row">
      <div class="bar-label">${cat}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="bar-size">${formatSize(info.size)}</div>
    </div>`;
  }
  html += '</div>';
  html += '<div class="chart-card"><h3>Size by Extension</h3>';
  for (const [ext, info] of extEntries) {
    const pct = (info.size / maxExtSize * 100).toFixed(1);
    html += `<div class="bar-row">
      <div class="bar-label">${esc(ext)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:var(--accent)"></div></div>
      <div class="bar-size">${formatSize(info.size)}</div>
    </div>`;
  }
  html += '</div></div>';

  // Recommendation banners
  html += '<div class="rec-banners">';
  if (dangerCount > 0 && !dismissedBanners['risk-danger']) {
    html += `<div class="rec-banner rec-danger" id="risk-danger">
      <span class="rec-icon">\u26A0\uFE0F</span>
      <div><strong>${dangerCount} file(s) with deserialization risk</strong> (pickle/joblib). These files can execute arbitrary code when loaded. Inspect before use.</div>
      <button class="rec-dismiss" onclick="dismissBanner('risk-danger')">\u2715</button>
    </div>`;
  }
  if (sourceCount > 0 && !dismissedBanners['rec-security']) {
    html += `<div class="rec-banner rec-info" id="rec-security">
      <span class="rec-icon">\uD83D\uDD0D</span>
      <div><strong>${sourceCount} source code file(s) found.</strong> Consider running a <strong>Code Security</strong> scan to check for vulnerabilities (eval, exec, subprocess injection, hardcoded secrets).</div>
      <button class="rec-dismiss" onclick="dismissBanner('rec-security')">\u2715</button>
    </div>`;
  }
  if (skillCount > 0 && !dismissedBanners['rec-skillguard']) {
    html += `<div class="rec-banner rec-warn" id="rec-skillguard">
      <span class="rec-icon">\uD83D\uDEE1\uFE0F</span>
      <div><strong>${skillCount} AI agent skill/tool file(s) found.</strong> Consider running a <strong>SkillGuard</strong> scan to verify permissions, input validation, and safe tool boundaries.</div>
      <button class="rec-dismiss" onclick="dismissBanner('rec-skillguard')">\u2715</button>
    </div>`;
  }
  if (secretCount > 0 && !dismissedBanners['rec-secrets']) {
    html += `<div class="rec-banner rec-danger" id="rec-secrets">
      <span class="rec-icon">\uD83D\uDD12</span>
      <div><strong>${secretCount} secret/credential file(s) found.</strong> Review and ensure these files are not committed to version control. Consider using a <strong>.gitignore</strong> or secrets manager.</div>
      <button class="rec-dismiss" onclick="dismissBanner('rec-secrets')">\u2715</button>
    </div>`;
  }
  if (embeddedSecretCount > 0 && !dismissedBanners['rec-embedded-secrets']) {
    html += `<div class="rec-banner rec-warn" id="rec-embedded-secrets">
      <span class="rec-icon">\u26A0\uFE0F</span>
      <div><strong>${embeddedSecretCount} file(s) may contain embedded secrets</strong> (API keys, passwords, tokens). Review these files and move secrets to environment variables or a secrets manager.</div>
      <button class="rec-dismiss" onclick="dismissBanner('rec-embedded-secrets')">\u2715</button>
    </div>`;
  }
  // Permission alerts
  const permAlerts = files.filter(f => f.permission_alert);
  const worldWritable = permAlerts.filter(f => f.permission_alert === 'world_writable');
  const worldReadable = permAlerts.filter(f => f.permission_alert === 'world_readable');
  const groupWritable = permAlerts.filter(f => f.permission_alert === 'group_writable');
  if (permAlerts.length > 0 && !dismissedBanners['rec-perm-alerts']) {
    let detail = '';
    if (worldWritable.length) detail += `<strong style="color:#ff5555">${worldWritable.length} world-writable</strong> `;
    if (worldReadable.length) detail += `<strong style="color:#f0c040">${worldReadable.length} world-readable</strong> `;
    if (groupWritable.length) detail += `<strong style="color:#ffb86c">${groupWritable.length} group-writable</strong> `;
    html += `<div class="rec-banner rec-danger" id="rec-perm-alerts">
      <span class="rec-icon">\uD83D\uDEE1\uFE0F</span>
      <div><strong>${permAlerts.length} file(s) have overly permissive access.</strong> ${detail}
      <br><small>Sensitive AI files (models, data, secrets, configs) should restrict access. Use <code>chmod 640</code> or <code>chmod 600</code> for secrets.</small></div>
      <button class="rec-dismiss" onclick="dismissBanner('rec-perm-alerts')">\u2715</button>
    </div>`;
  }
  html += '</div>';

  // Filters
  html += '<div class="filters">';
  for (const c of ALL_CATS) {
    const cls = activeCategory === c ? 'chip active' : 'chip';
    const label = c === 'all' ? `All (${s.total_files})` :
      `${c} (${(s.by_category[c]||{count:0}).count})`;
    html += `<button class="${cls}" data-cat="${c}" onclick="setCategory('${c}')">${label}</button>`;
  }
  html += `<span style="color:var(--text-dim);font-size:13px;margin-left:8px">Min size:</span>
    <input class="min-size-input" type="text" placeholder="e.g. 10KB"
           value="${minSizeFilter > 0 ? formatSize(minSizeFilter) : ''}"
           onchange="onMinSize(this.value)">`;
  html += '</div>';

  // Options bar
  html += '<div class="options-bar">';
  html += `<button class="toggle-btn ${viewMode==='table'?'active':''}" onclick="setViewMode('table')">Table</button>`;
  html += `<button class="toggle-btn ${viewMode==='tree'?'active':''}" onclick="setViewMode('tree')">Tree</button>`;
  html += `<label class="opt-check"><input type="checkbox" ${optPermissions?'checked':''} onchange="togglePermissions(this.checked)"> Permissions</label>`;
  html += `<label class="opt-check"><input type="checkbox" ${optHashes?'checked':''} onchange="toggleHashes(this.checked)"> Hashes</label>`;
  html += `<label class="opt-check"><input type="checkbox" ${optIntegrity?'checked':''} onchange="toggleIntegrity(this.checked)"> Integrity Check</label>`;
  html += '<span style="flex:1"></span>';
  html += `<button class="btn-browse" onclick="exportJson()" title="Export scan results as JSON (Pandas-compatible)">\u2B07 Export JSON</button>`;
  html += '</div>';

  // Dynamic columns
  const hasPermissions = files.some(f => f.owner);
  const hasHashes = files.some(f => f.hash);
  const columns = [
    {key:'category', label:'Category', tooltip:'File classification (data, model, config, etc.)'},
    {key:'extension', label:'Extension', tooltip:'File extension or [dir] for directories'},
    {key:'size_bytes', label:'Size', tooltip:'File size on disk'},
    {key:'modified', label:'Modified', tooltip:'Last modification date'},
    {key:'confidence', label:'C', tooltip:'Confidence: H(igh), M(edium), L(ow) — how certain the classification is'},
    {key:'risk_level', label:'Risk', tooltip:'Security risk flags for dangerous file types'},
    {key:'filename', label:'File', tooltip:'Filename'},
  ];
  if (hasPermissions) {
    columns.push({key:'owner', label:'Owner', tooltip:'File owner (user account)'});
    columns.push({key:'group', label:'Group', tooltip:'File group'});
    columns.push({key:'permissions', label:'Perms', tooltip:'File permissions (e.g. -rwxr-xr-x)'});
  }
  if (hasHashes) {
    columns.push({key:'hash', label:'Hash', tooltip:'SHA-256 hash for modification detection'});
  }
  columns.push({key:'path', label:'Path', tooltip:'Full filesystem path'});

  // Integrity check results
  if (scanData.integrity) {
    const ic = scanData.integrity;
    if (ic.status === 'no_baseline') {
      html += `<div class="rec-banner rec-info">
        <span class="rec-icon">\u2139\uFE0F</span>
        <div><strong>Integrity Check:</strong> ${esc(ic.message)} Future scans will compare against this baseline.</div>
      </div>`;
    } else if (ic.status === 'clean') {
      html += `<div class="rec-banner" style="background:rgba(80,250,123,.08);border-color:rgba(80,250,123,.3)">
        <span class="rec-icon">\u2705</span>
        <div><strong>Integrity Check:</strong> No changes detected since <em>${esc(ic.previous_report)}</em></div>
      </div>`;
    } else if (ic.status === 'changes_detected') {
      let detail = '';
      if (ic.summary.new > 0) detail += `<span style="color:#50fa7b">\u2795 ${ic.summary.new} new</span> &nbsp; `;
      if (ic.summary.changed > 0) detail += `<span style="color:#f0c040">\u270F\uFE0F ${ic.summary.changed} changed</span> &nbsp; `;
      if (ic.summary.deleted > 0) detail += `<span style="color:#ff5555">\uD83D\uDDD1 ${ic.summary.deleted} deleted</span>`;
      html += `<div class="rec-banner rec-warn">
        <span class="rec-icon">\uD83D\uDD0D</span>
        <div><strong>Integrity Check vs ${esc(ic.previous_report)}:</strong> ${detail}</div>
      </div>`;

      // Expandable details
      let items = '';
      if (ic.new_files && ic.new_files.length) {
        items += '<div style="margin-top:6px"><strong style="color:#50fa7b">\u2795 New files:</strong><ul style="margin:4px 0;padding-left:20px">';
        ic.new_files.forEach(f => items += `<li>${esc(f.path)}</li>`);
        items += '</ul></div>';
      }
      if (ic.changed_files && ic.changed_files.length) {
        items += '<div style="margin-top:6px"><strong style="color:#f0c040">\u270F\uFE0F Changed files:</strong><ul style="margin:4px 0;padding-left:20px">';
        ic.changed_files.forEach(f => items += `<li>${esc(f.path)}</li>`);
        items += '</ul></div>';
      }
      if (ic.deleted_files && ic.deleted_files.length) {
        items += '<div style="margin-top:6px"><strong style="color:#ff5555">\uD83D\uDDD1 Deleted files:</strong><ul style="margin:4px 0;padding-left:20px">';
        ic.deleted_files.forEach(f => items += `<li>${esc(f.path)}</li>`);
        items += '</ul></div>';
      }
      if (items) {
        html += `<details class="errors-bar" style="margin-top:0"><summary>View integrity details</summary>${items}</details>`;
      }
    }
  }
  if (scanData.report_saved) {
    let logInfo = scanData.integrity_log ? ` &nbsp;|&nbsp; \uD83D\uDCDD Log: ${esc(scanData.integrity_log)}` : '';
    html += `<div style="font-size:12px;color:var(--text-dim);padding:0 8px 8px">\uD83D\uDCBE Report saved: ${esc(scanData.report_saved)}${logInfo}</div>`;
  }

  if (viewMode === 'tree') {
    html += renderTreeView(files, s.scan_path, hasPermissions, hasHashes);
  } else {
    html += renderTableView(files, columns, arrow);
  }

  document.getElementById('app').innerHTML = html;
}

function setViewMode(mode) { viewMode = mode; render(); }
function togglePermissions(checked) { optPermissions = checked; runScan(); }
function toggleHashes(checked) { optHashes = checked; runScan(); }
function toggleIntegrity(checked) { optIntegrity = checked; runScan(); }

function renderTableView(files, columns, arrow) {
  let html = '<div class="table-wrap"><table><thead><tr>';
  for (const col of columns) {
    html += `<th onclick="setSort('${col.key}')" title="${col.tooltip}">${col.label}<span class="sort-arrow">${arrow(col.key)}</span></th>`;
  }
  html += '</tr></thead><tbody>';

  if (files.length === 0) {
    html += `<tr><td colspan="${columns.length}" style="text-align:center;padding:30px;color:var(--text-dim)">No files match current filters</td></tr>`;
  }

  for (const f of files) {
    html += renderTableRow(f, columns);
  }
  html += '</tbody></table></div>';
  return html;
}

function renderTableRow(f, columns) {
  const cats = f.categories || [f.category];
  const badges = cats.map(c => `<span class="cat-badge cat-${c}">${c.toUpperCase()}</span>`).join(' ');
  const confClass = 'conf-' + f.confidence[0].toLowerCase();
  let riskHtml = '';
  if (f.risk_level === 'danger') riskHtml = `<span class="risk-danger" title="${esc(f.risk_reason)}">DANGER</span>`;
  else if (f.risk_level === 'warning') riskHtml = `<span class="risk-warning" title="${esc(f.risk_reason)}">WARN</span>`;

  let html = '<tr>';
  for (const col of columns) {
    switch (col.key) {
      case 'category': html += `<td>${badges}</td>`; break;
      case 'extension': html += `<td>${esc(f.extension || '[dir]')}</td>`; break;
      case 'size_bytes': html += `<td style="text-align:right;white-space:nowrap">${formatSize(f.size_bytes)}</td>`; break;
      case 'modified': html += `<td style="white-space:nowrap">${esc(f.modified)}</td>`; break;
      case 'confidence': html += `<td class="${confClass}" style="font-weight:600">${f.confidence[0].toUpperCase()}</td>`; break;
      case 'risk_level': html += `<td>${riskHtml}</td>`; break;
      case 'filename': html += `<td>${esc(f.filename)}</td>`; break;
      case 'path': html += `<td class="path-cell" title="${esc(f.path)}">${esc(f.path)}</td>`; break;
      case 'owner': html += `<td>${esc(f.owner||'')}</td>`; break;
      case 'group': html += `<td>${esc(f.group||'')}</td>`; break;
      case 'permissions': {
        const pa = f.permission_alert;
        const paColor = pa === 'world_writable' ? '#ff5555' : pa === 'world_readable' ? '#f0c040' : pa === 'group_writable' ? '#ffb86c' : '';
        const paIcon = pa ? (pa === 'world_writable' ? '\u26D4' : '\u26A0\uFE0F') : '';
        const paTitle = f.permission_alert_reason || '';
        html += `<td style="font-family:monospace;${paColor ? 'color:'+paColor : ''}" title="${esc(paTitle)}">${paIcon ? paIcon+' ' : ''}${esc(f.permissions||'')}</td>`;
        break;
      }
      case 'hash': html += `<td style="font-family:monospace;font-size:11px" title="${esc(f.hash||'')}">${f.hash ? esc(f.hash.slice(0,12)) + '\u2026' : ''}</td>`; break;
      default: html += `<td>${esc(String(f[col.key]||''))}</td>`;
    }
  }
  html += '</tr>';
  return html;
}

function buildTree(files, scanRoot) {
  const root = { name: scanRoot, children: {}, files: [], path: scanRoot };
  for (const f of files) {
    let rel = f.path;
    if (rel.startsWith(scanRoot)) rel = rel.slice(scanRoot.length);
    while (rel.startsWith('/')) rel = rel.slice(1);
    const dirPart = rel.substring(0, rel.length - f.filename.length);
    const segments = dirPart.split('/').filter(s => s.length > 0);
    let node = root;
    let curPath = scanRoot;
    for (const seg of segments) {
      curPath += '/' + seg;
      if (!node.children[seg]) {
        node.children[seg] = { name: seg, children: {}, files: [], path: curPath };
      }
      node = node.children[seg];
    }
    node.files.push(f);
  }
  return root;
}

function countTreeFiles(node) {
  let count = node.files.length;
  for (const child of Object.values(node.children)) count += countTreeFiles(child);
  return count;
}

function sumTreeSize(node) {
  let size = node.files.reduce((s, f) => s + f.size_bytes, 0);
  for (const child of Object.values(node.children)) size += sumTreeSize(child);
  return size;
}

function renderTreeView(files, scanRoot, hasPermissions, hasHashes) {
  if (files.length === 0) {
    return '<div class="tree-container"><div style="text-align:center;padding:30px;color:var(--text-dim)">No files match current filters</div></div>';
  }
  const tree = buildTree(files, scanRoot);
  expandedPaths.add(tree.path);
  let html = '<div class="tree-container">';
  html += renderTreeNode(tree, 0, hasPermissions, hasHashes);
  html += '</div>';
  return html;
}

function renderTreeNode(node, depth, hasPerms, hasHash) {
  let html = '';
  const isExpanded = expandedPaths.has(node.path);
  const fileCount = countTreeFiles(node);
  const totalSize = sumTreeSize(node);
  const indent = '<span class="tree-indent"></span>'.repeat(depth);
  const caretClass = isExpanded ? 'tree-caret open' : 'tree-caret';

  html += `<div class="tree-row tree-folder" onclick="toggleTreeNode('${esc(node.path)}')">
    ${indent}<span class="${caretClass}">\u25B6</span>
    <span class="tree-folder-name">\uD83D\uDCC1 ${esc(node.name)}</span>
    <span class="tree-meta">
      <span class="tree-count">${fileCount} file${fileCount!==1?'s':''}</span>
      <span>${formatSize(totalSize)}</span>
    </span>
  </div>`;

  if (isExpanded) {
    const childKeys = Object.keys(node.children).sort();
    for (const key of childKeys) {
      html += renderTreeNode(node.children[key], depth + 1, hasPerms, hasHash);
    }
    for (const f of node.files) {
      html += renderTreeFile(f, depth + 1, hasPerms, hasHash);
    }
  }
  return html;
}

function renderTreeFile(f, depth, hasPerms, hasHash) {
  const indent = '<span class="tree-indent"></span>'.repeat(depth);
  const cats = f.categories || [f.category];
  const badges = cats.map(c => `<span class="cat-badge cat-${c}">${c.toUpperCase()}</span>`).join(' ');
  let riskHtml = '';
  if (f.risk_level === 'danger') riskHtml = `<span class="risk-danger" title="${esc(f.risk_reason)}">!</span>`;
  else if (f.risk_level === 'warning') riskHtml = `<span class="risk-warning" title="${esc(f.risk_reason)}">!</span>`;
  let extras = '';
  if (hasPerms && f.owner) {
    const tpa = f.permission_alert;
    const tpaColor = tpa === 'world_writable' ? '#ff5555' : tpa === 'world_readable' ? '#f0c040' : tpa === 'group_writable' ? '#ffb86c' : 'var(--text-dim)';
    const tpaIcon = tpa ? (tpa === 'world_writable' ? '\u26D4 ' : '\u26A0\uFE0F ') : '';
    extras += `<span style="font-size:11px;color:${tpaColor}" title="${esc(f.permission_alert_reason||'')}">${tpaIcon}${esc(f.owner)}:${esc(f.group)} ${esc(f.permissions)}</span>`;
  }
  if (hasHash && f.hash) extras += `<span style="font-family:monospace;font-size:11px;color:var(--text-dim)" title="${esc(f.hash)}">${esc(f.hash.slice(0,12))}\u2026</span>`;
  return `<div class="tree-row">
    ${indent}<span class="tree-indent"></span>
    <span class="tree-file-name">${esc(f.filename)}</span>
    ${badges} ${riskHtml}
    <span class="tree-meta">${extras}<span>${formatSize(f.size_bytes)}</span></span>
  </div>`;
}

function toggleTreeNode(path) {
  if (expandedPaths.has(path)) expandedPaths.delete(path);
  else expandedPaths.add(path);
  render();
}

document.getElementById('scanPath').addEventListener('keydown', e => {
  if (e.key === 'Enter') runScan();
});

// --- Browse dialog ---
let browseCurrent = '';

function openBrowseDialog() {
  const startPath = document.getElementById('scanPath').value.trim() || '';
  browseTo(startPath);
}

function browseTo(path) {
  fetch('/api/browse', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({path})
  })
  .then(r => {
    if (!r.ok) return r.json().then(e => { throw new Error(e.error || 'Browse failed'); });
    return r.json();
  })
  .then(data => {
    browseCurrent = data.current;
    showBrowseModal(data);
  })
  .catch(err => {
    // If path invalid, try home directory
    if (path) browseTo('');
  });
}

function showBrowseModal(data) {
  // Remove existing modal
  const existing = document.getElementById('browseModal');
  if (existing) existing.remove();

  let itemsHtml = '';
  if (data.directories.length === 0) {
    itemsHtml = '<div class="browse-empty">No subdirectories found</div>';
  } else {
    for (const dir of data.directories) {
      itemsHtml += `<div class="browse-item" ondblclick="browseTo('${esc(data.current + '/' + dir)}')" onclick="selectBrowseItem(this, '${esc(data.current + '/' + dir)}')">
        <span class="browse-icon">\uD83D\uDCC1</span> ${esc(dir)}
      </div>`;
    }
  }

  const upBtn = data.parent
    ? `<button class="modal-up" onclick="browseTo('${esc(data.parent)}')">\u2191 Up</button>`
    : '';

  const modal = document.createElement('div');
  modal.id = 'browseModal';
  modal.className = 'modal-overlay';
  modal.onclick = (e) => { if (e.target === modal) closeBrowseModal(); };
  modal.innerHTML = `
    <div class="modal">
      <div class="modal-header">
        <h3>Select Directory</h3>
        <button class="modal-close" onclick="closeBrowseModal()">\u2715</button>
      </div>
      <div class="modal-breadcrumb">
        ${upBtn}
        <span>${esc(data.current)}</span>
      </div>
      <div class="modal-body">${itemsHtml}</div>
      <div class="modal-footer">
        <button class="btn-secondary" onclick="closeBrowseModal()">Cancel</button>
        <button class="btn-select" onclick="confirmBrowse()">Select This Folder</button>
      </div>
    </div>`;
  document.body.appendChild(modal);
}

function selectBrowseItem(el, path) {
  // Highlight selected item
  document.querySelectorAll('.browse-item.selected').forEach(e => e.classList.remove('selected'));
  el.classList.add('selected');
  el.style.background = 'rgba(0,210,255,.12)';
  // Deselect others
  document.querySelectorAll('.browse-item').forEach(e => { if (e !== el) e.style.background = ''; });
  browseCurrent = path;
}

function confirmBrowse() {
  document.getElementById('scanPath').value = browseCurrent;
  closeBrowseModal();
}

function closeBrowseModal() {
  const modal = document.getElementById('browseModal');
  if (modal) modal.remove();
}

// --- Export JSON ---
function exportJson() {
  if (!scanData || !scanData.files) return;
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const ts = now.getFullYear().toString()
    + pad(now.getMonth() + 1) + pad(now.getDate())
    + '_' + pad(now.getHours()) + pad(now.getMinutes()) + pad(now.getSeconds());
  const filename = `aifiles_${ts}.json`;

  const payload = {
    scan_metadata: scanData.summary,
    exported_at: now.toISOString(),
    files: scanData.files
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class ScanHandler(BaseHTTPRequestHandler):
    default_scan_path = ""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_html(DASHBOARD_HTML)
        elif parsed.path == "/api/default-path":
            self._send_json({"path": self.default_scan_path})
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/scan":
            self._handle_scan()
        elif parsed.path == "/api/browse":
            self._handle_browse()
        else:
            self._send_error(404, "Not found")

    def _handle_browse(self) -> None:
        """List directories at a given path for the folder browser dialog."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._send_error(400, "Invalid JSON body")
            return

        browse_path = body.get("path", "").strip()
        if not browse_path:
            browse_path = os.path.expanduser("~")

        browse_path = os.path.abspath(browse_path)
        if not os.path.isdir(browse_path):
            self._send_error(400, f"Not a valid directory: {browse_path}")
            return

        dirs = []
        try:
            for entry in sorted(os.scandir(browse_path), key=lambda e: e.name.lower()):
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                except OSError:
                    pass
        except PermissionError:
            self._send_error(403, f"Permission denied: {browse_path}")
            return
        except OSError as e:
            self._send_error(500, str(e))
            return

        parent = os.path.dirname(browse_path)
        self._send_json({
            "current": browse_path,
            "parent": parent if parent != browse_path else None,
            "directories": dirs,
        })

    def _handle_scan(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._send_error(400, "Invalid JSON body")
            return

        scan_path = body.get("path", "").strip()
        if not scan_path:
            self._send_error(400, "Missing 'path' field")
            return

        scan_path = os.path.abspath(scan_path)
        if not os.path.isdir(scan_path):
            self._send_error(400, f"Not a valid directory: {scan_path}")
            return

        min_size = 0
        if body.get("min_size"):
            try:
                from aifiles import parse_size
                min_size = parse_size(body["min_size"])
            except ValueError as e:
                self._send_error(400, str(e))
                return

        categories = body.get("categories")
        compute_permissions = body.get("compute_permissions", False)
        compute_hashes = body.get("compute_hashes", False)
        run_integrity = body.get("integrity", False)

        # Force hashes + permissions for integrity checks
        if run_integrity:
            compute_permissions = True
            compute_hashes = True

        config = ScanConfig(
            root_path=scan_path,
            min_size_bytes=min_size,
            categories=categories,
            quiet=True,
            compute_permissions=compute_permissions,
            compute_hashes=compute_hashes,
        )
        classifier = FileClassifier()
        scanner = AIFileScanner(config, classifier)
        results, summary = scanner.scan()

        response = {
            "summary": {
                "scan_path": summary.scan_path,
                "scan_time": summary.scan_time,
                "total_files": summary.total_files,
                "total_size_bytes": summary.total_size,
                "total_size_human": format_size(summary.total_size),
                "scan_duration_seconds": round(summary.scan_duration, 3),
                "by_category": {
                    cat: {**info, "size_human": format_size(info["size"])}
                    for cat, info in summary.by_category.items()
                },
                "by_extension": {
                    ext: {**info, "size_human": format_size(info["size"])}
                    for ext, info in summary.by_extension.items()
                },
                "errors": summary.errors,
            },
            "files": [
                {
                    "path": r.path,
                    "filename": r.filename,
                    "extension": r.extension,
                    "category": r.categories[0].value,
                    "categories": [c.value for c in r.categories],
                    "size_bytes": r.size_bytes,
                    "size_human": format_size(r.size_bytes),
                    "modified": format_timestamp(r.modified_timestamp),
                    "type": "directory" if r.is_directory else "file",
                    "detection_method": r.detection_method,
                    "confidence": r.confidence,
                    "risk_level": r.risk_level,
                    "risk_reason": r.risk_reason,
                    "owner": r.owner,
                    "group": r.group,
                    "permissions": r.permissions,
                    "permission_alert": r.permission_alert,
                    "permission_alert_reason": r.permission_alert_reason,
                    "hash": r.hash,
                }
                for r in results
            ],
        }

        # Integrity check
        if run_integrity:
            report_dir = _ensure_report_dir()
            integrity = run_integrity_check(results, report_dir)
            if integrity is None:
                response["integrity"] = {
                    "status": "no_baseline",
                    "message": "No previous report found — this scan will serve as the baseline.",
                }
            elif not integrity.has_changes:
                response["integrity"] = {
                    "status": "clean",
                    "previous_report": integrity.previous_report,
                    "previous_time": integrity.previous_time,
                    "message": "No changes detected.",
                }
            else:
                response["integrity"] = {
                    "status": "changes_detected",
                    **integrity.to_dict(),
                }

            # Save current scan as report for future comparisons
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(report_dir, f"aifiles_{timestamp}.json")
            payload = _build_json_payload(results, summary)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            response["report_saved"] = report_path

            # Append to integrity log
            log_path = append_integrity_log(report_dir, scan_path, integrity, report_path)
            response["integrity_log"] = log_path

        self._send_json(response)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        data = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"  {args[0]}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI File Discovery — Web Dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=8505,
        help="Port to serve on (default: 8505)",
    )
    parser.add_argument(
        "--scan-path", default="",
        help="Default scan path to pre-fill in the UI",
    )
    args = parser.parse_args()

    if not args.scan_path:
        test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        if os.path.isdir(test_data):
            args.scan_path = test_data

    ScanHandler.default_scan_path = args.scan_path

    server = HTTPServer(("0.0.0.0", args.port), ScanHandler)
    print(f"  AI File Discovery Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
