#!/usr/bin/env python3
"""AI File Discovery — Central Master Dashboard.

Collects scan reports from multiple nodes, aggregates results,
tracks changes, and provides a corporate-wide overview of AI files.

Zero-dependency. Pure Python 3.8+ stdlib with SQLite storage.

Usage:
    python3 central.py                       # serve on port 8510
    python3 central.py --port 9000           # custom port
    python3 central.py --db /path/store.db   # custom database path

API Endpoints:
    POST /api/report          — Submit a scan report (JSON body)
    GET  /api/nodes           — List all known nodes
    GET  /api/node/<id>       — Get latest report for a node
    GET  /api/overview        — Corporate overview statistics
    GET  /api/alerts          — All active alerts across nodes
    GET  /api/changes         — Recent integrity changes across nodes
    GET  /api/history/<id>    — Scan history for a node
    GET  /                    — Corporate dashboard UI
"""

import argparse
import hashlib
import json
import os
import secrets
import sqlite3
import sys
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    machine_id    TEXT PRIMARY KEY,
    hostname      TEXT NOT NULL,
    fqdn          TEXT,
    ip_address    TEXT,
    mac_address   TEXT,
    os            TEXT,
    os_version    TEXT,
    os_release    TEXT,
    platform      TEXT,
    architecture  TEXT,
    username      TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id    TEXT NOT NULL,
    received_at   TEXT NOT NULL,
    scan_time     TEXT,
    scan_path     TEXT,
    total_files   INTEGER,
    total_size    INTEGER,
    scan_duration REAL,
    report_json   TEXT NOT NULL,
    FOREIGN KEY (machine_id) REFERENCES nodes(machine_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    alert_type    TEXT NOT NULL,
    severity      TEXT NOT NULL,
    message       TEXT NOT NULL,
    details       TEXT,
    acknowledged  INTEGER DEFAULT 0,
    FOREIGN KEY (machine_id) REFERENCES nodes(machine_id)
);

CREATE TABLE IF NOT EXISTS changes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id    TEXT NOT NULL,
    detected_at   TEXT NOT NULL,
    change_type   TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    filename      TEXT,
    details       TEXT,
    FOREIGN KEY (machine_id) REFERENCES nodes(machine_id)
);

CREATE INDEX IF NOT EXISTS idx_reports_machine ON reports(machine_id);
CREATE INDEX IF NOT EXISTS idx_reports_received ON reports(received_at);
CREATE INDEX IF NOT EXISTS idx_alerts_machine ON alerts(machine_id);
CREATE INDEX IF NOT EXISTS idx_changes_machine ON changes(machine_id);
CREATE INDEX IF NOT EXISTS idx_changes_detected ON changes(detected_at);
"""


class CentralDB:
    """Thread-safe SQLite database for the central dashboard."""

    def __init__(self, db_path: str = "central.db"):
        self.db_path = db_path
        self._local = threading.local()
        # Initialize schema
        conn = self._get_conn()
        conn.executescript(DB_SCHEMA)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def ingest_report(self, report: dict) -> dict:
        """Ingest a scan report from a node. Returns summary of what was stored."""
        conn = self._get_conn()
        now = datetime.now().isoformat(timespec="seconds")

        system = report.get("system", {})
        machine_id = system.get("machine_id", "unknown")
        hostname = system.get("hostname", "unknown")

        # Upsert node
        existing = conn.execute(
            "SELECT machine_id FROM nodes WHERE machine_id = ?", (machine_id,)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE nodes SET hostname=?, fqdn=?, ip_address=?, mac_address=?,
                    os=?, os_version=?, os_release=?, platform=?, architecture=?,
                    username=?, last_seen=?
                WHERE machine_id=?
            """, (
                hostname, system.get("fqdn"), system.get("ip_address"),
                system.get("mac_address"), system.get("os"), system.get("os_version"),
                system.get("os_release"), system.get("platform"),
                system.get("architecture"), system.get("username"), now, machine_id,
            ))
        else:
            conn.execute("""
                INSERT INTO nodes (machine_id, hostname, fqdn, ip_address, mac_address,
                    os, os_version, os_release, platform, architecture, username,
                    first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                machine_id, hostname, system.get("fqdn"), system.get("ip_address"),
                system.get("mac_address"), system.get("os"), system.get("os_version"),
                system.get("os_release"), system.get("platform"),
                system.get("architecture"), system.get("username"), now, now,
            ))

        # Store report
        metadata = report.get("scan_metadata", {})
        conn.execute("""
            INSERT INTO reports (machine_id, received_at, scan_time, scan_path,
                total_files, total_size, scan_duration, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            machine_id, now, metadata.get("scan_time"), metadata.get("scan_path"),
            metadata.get("total_files", 0), metadata.get("total_size_bytes", 0),
            metadata.get("scan_duration_seconds", 0), json.dumps(report),
        ))

        report_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Generate alerts from report
        alerts_generated = self._generate_alerts(conn, machine_id, hostname, report, now)

        # Detect changes via integrity data
        changes_detected = self._detect_changes(conn, machine_id, report, now)

        conn.commit()

        return {
            "status": "accepted",
            "machine_id": machine_id,
            "hostname": hostname,
            "report_id": report_id,
            "alerts_generated": alerts_generated,
            "changes_detected": changes_detected,
        }

    def _generate_alerts(self, conn, machine_id, hostname, report, now) -> int:
        """Generate alerts from report content."""
        count = 0
        files = report.get("files", [])

        # Secret files alert
        secrets = [f for f in files if "secret" in (f.get("categories") or [])]
        if secrets:
            conn.execute("""
                INSERT INTO alerts (machine_id, created_at, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (machine_id, now, "secrets_found", "critical",
                  f"{len(secrets)} secret/credential file(s) found on {hostname}",
                  json.dumps([f["path"] for f in secrets])))
            count += 1

        # Deserialization risk
        risky = [f for f in files if f.get("risk_level") == "danger" and
                 f.get("risk_reason", "").startswith("Deserialization")]
        if risky:
            conn.execute("""
                INSERT INTO alerts (machine_id, created_at, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (machine_id, now, "deserialization_risk", "warning",
                  f"{len(risky)} file(s) with deserialization risk on {hostname}",
                  json.dumps([f["path"] for f in risky])))
            count += 1

        # Permission alerts
        perm_alerts = [f for f in files if f.get("permission_alert")]
        if perm_alerts:
            conn.execute("""
                INSERT INTO alerts (machine_id, created_at, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (machine_id, now, "permissions", "warning",
                  f"{len(perm_alerts)} file(s) with overly permissive access on {hostname}",
                  json.dumps([{"path": f["path"], "perms": f.get("permissions")} for f in perm_alerts[:20]])))
            count += 1

        # Embedded secrets
        embedded = [f for f in files if f.get("risk_reason", "").startswith("Possible embedded")]
        if embedded:
            conn.execute("""
                INSERT INTO alerts (machine_id, created_at, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (machine_id, now, "embedded_secrets", "critical",
                  f"{len(embedded)} file(s) with embedded secrets on {hostname}",
                  json.dumps([f["path"] for f in embedded])))
            count += 1

        return count

    def _detect_changes(self, conn, machine_id, report, now) -> int:
        """Detect changes from integrity data in the report."""
        integrity = report.get("integrity", {})
        if not integrity or integrity.get("status") not in ("changes_detected",):
            return 0

        count = 0
        for f in integrity.get("new_files", []):
            conn.execute("""
                INSERT INTO changes (machine_id, detected_at, change_type, file_path, filename, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (machine_id, now, "new", f["path"], f.get("filename", ""), json.dumps(f)))
            count += 1

        for f in integrity.get("changed_files", []):
            conn.execute("""
                INSERT INTO changes (machine_id, detected_at, change_type, file_path, filename, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (machine_id, now, "modified", f["path"], f.get("filename", ""), json.dumps(f)))
            count += 1

        for f in integrity.get("deleted_files", []):
            conn.execute("""
                INSERT INTO changes (machine_id, detected_at, change_type, file_path, filename, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (machine_id, now, "deleted", f["path"], f.get("filename", ""), json.dumps(f)))
            count += 1

        return count

    def get_nodes(self) -> list:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT n.*,
                (SELECT COUNT(*) FROM reports WHERE machine_id = n.machine_id) as report_count,
                (SELECT total_files FROM reports WHERE machine_id = n.machine_id ORDER BY received_at DESC LIMIT 1) as latest_files,
                (SELECT total_size FROM reports WHERE machine_id = n.machine_id ORDER BY received_at DESC LIMIT 1) as latest_size,
                (SELECT scan_path FROM reports WHERE machine_id = n.machine_id ORDER BY received_at DESC LIMIT 1) as latest_scan_path
            FROM nodes n ORDER BY n.last_seen DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_node(self, machine_id: str) -> dict:
        conn = self._get_conn()
        node = conn.execute("SELECT * FROM nodes WHERE machine_id = ?", (machine_id,)).fetchone()
        if not node:
            return None
        latest = conn.execute(
            "SELECT report_json FROM reports WHERE machine_id = ? ORDER BY received_at DESC LIMIT 1",
            (machine_id,)
        ).fetchone()
        return {
            "node": dict(node),
            "latest_report": json.loads(latest["report_json"]) if latest else None,
        }

    def get_overview(self) -> dict:
        conn = self._get_conn()
        nodes = self.get_nodes()
        total_nodes = len(nodes)
        total_files = sum(n.get("latest_files") or 0 for n in nodes)
        total_size = sum(n.get("latest_size") or 0 for n in nodes)

        # Aggregate categories across all latest reports
        by_category = {}
        for node in nodes:
            latest = conn.execute(
                "SELECT report_json FROM reports WHERE machine_id = ? ORDER BY received_at DESC LIMIT 1",
                (node["machine_id"],)
            ).fetchone()
            if latest:
                report = json.loads(latest["report_json"])
                cats = report.get("scan_metadata", {}).get("by_category", {})
                for cat, info in cats.items():
                    if cat not in by_category:
                        by_category[cat] = {"count": 0, "size": 0}
                    by_category[cat]["count"] += info.get("count", 0)
                    by_category[cat]["size"] += info.get("size", 0)

        # Recent alerts
        recent_alerts = conn.execute("""
            SELECT a.*, n.hostname FROM alerts a
            JOIN nodes n ON a.machine_id = n.machine_id
            ORDER BY a.created_at DESC LIMIT 50
        """).fetchall()

        # Recent changes
        recent_changes = conn.execute("""
            SELECT c.*, n.hostname FROM changes c
            JOIN nodes n ON c.machine_id = n.machine_id
            ORDER BY c.detected_at DESC LIMIT 50
        """).fetchall()

        # Alert counts by severity
        alert_counts = {}
        for row in conn.execute(
            "SELECT severity, COUNT(*) as cnt FROM alerts WHERE acknowledged = 0 GROUP BY severity"
        ).fetchall():
            alert_counts[row["severity"]] = row["cnt"]

        return {
            "total_nodes": total_nodes,
            "total_files": total_files,
            "total_size": total_size,
            "by_category": by_category,
            "alert_counts": alert_counts,
            "nodes": nodes,
            "recent_alerts": [dict(r) for r in recent_alerts],
            "recent_changes": [dict(r) for r in recent_changes],
        }

    def get_alerts(self, limit: int = 100) -> list:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT a.*, n.hostname FROM alerts a
            JOIN nodes n ON a.machine_id = n.machine_id
            ORDER BY a.created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_changes(self, limit: int = 100) -> list:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT c.*, n.hostname FROM changes c
            JOIN nodes n ON c.machine_id = n.machine_id
            ORDER BY c.detected_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_history(self, machine_id: str, limit: int = 50) -> list:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT id, machine_id, received_at, scan_time, scan_path,
                   total_files, total_size, scan_duration
            FROM reports WHERE machine_id = ? ORDER BY received_at DESC LIMIT ?
        """, (machine_id, limit)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def format_size(size_bytes):
    if size_bytes is None:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI File Discovery — Central Dashboard</title>
<style>
  :root { --bg: #0d1117; --bg2: #161b22; --card: #1c2333; --border: #30363d;
          --text: #e6edf3; --text-dim: #8b949e; --accent: #00d2ff;
          --danger: #ff5555; --warn: #f0c040; --success: #50fa7b; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--text); }

  .header { display: flex; align-items: center; gap: 16px; padding: 12px 24px;
            background: var(--bg2); border-bottom: 1px solid var(--border); }
  .header h1 { font-size: 20px; font-weight: 600; white-space: nowrap; }
  .header h1 span { color: var(--accent); }
  .header .subtitle { color: var(--text-dim); font-size: 13px; }

  .container { max-width: 1400px; margin: 0 auto; padding: 20px 24px; }

  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 12px; margin-bottom: 24px; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .stat-label { font-size: 11px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.5px; }
  .stat-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .stat-value.danger { color: var(--danger); }
  .stat-value.warn { color: var(--warn); }
  .stat-value.success { color: var(--success); }

  .section { margin-bottom: 28px; }
  .section-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }

  .node-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 12px; }
  .node-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
               padding: 16px; cursor: pointer; transition: border-color .15s; }
  .node-card:hover { border-color: var(--accent); }
  .node-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .node-name { font-weight: 600; font-size: 15px; }
  .node-os { color: var(--text-dim); font-size: 12px; }
  .node-stats { display: flex; gap: 16px; font-size: 13px; color: var(--text-dim); }
  .node-stats span { display: flex; align-items: center; gap: 4px; }
  .node-last-seen { font-size: 11px; color: var(--text-dim); margin-top: 6px; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; background: var(--bg2); color: var(--text-dim);
       font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
       border-bottom: 1px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
  tr:hover { background: rgba(0,210,255,.03); }

  .severity { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
  .severity.critical { background: rgba(255,85,85,.15); color: var(--danger); }
  .severity.warning { background: rgba(240,192,64,.15); color: var(--warn); }
  .severity.info { background: rgba(0,210,255,.1); color: var(--accent); }

  .change-type { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .change-type.new { background: rgba(80,250,123,.15); color: var(--success); }
  .change-type.modified { background: rgba(240,192,64,.15); color: var(--warn); }
  .change-type.deleted { background: rgba(255,85,85,.15); color: var(--danger); }

  .tab-bar { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 1px solid var(--border); }
  .tab { padding: 10px 20px; cursor: pointer; font-size: 14px; color: var(--text-dim);
         border-bottom: 2px solid transparent; transition: all .15s; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .chart-box { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .chart-box h3 { font-size: 14px; margin-bottom: 12px; color: var(--text-dim); }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
  .bar-label { width: 100px; text-align: right; color: var(--text-dim); flex-shrink: 0; }
  .bar-track { flex: 1; height: 16px; background: var(--bg); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .bar-value { width: 70px; text-align: right; font-size: 12px; color: var(--text-dim); flex-shrink: 0; }

  .empty { text-align: center; padding: 40px; color: var(--text-dim); }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 11px;
           background: rgba(0,210,255,.12); color: var(--accent); }

  .cat-colors { }
</style>
</head>
<body>

<div class="header">
  <h1><span>AI</span> File Discovery — <span>Central</span></h1>
  <span class="subtitle">Corporate Overview</span>
</div>

<div class="container" id="app">
  <div class="empty">
    <h2>Central Dashboard</h2>
    <p style="margin-top:8px">Waiting for nodes to report in...</p>
    <p style="margin-top:16px;font-size:13px;color:var(--text-dim)">
      Nodes push reports via:<br>
      <code style="color:var(--accent)">curl -X POST http://THIS_HOST:PORT/api/report -H 'Content-Type: application/json' -H 'Authorization: Bearer YOUR_API_KEY' -d @report.json</code>
    </p>
  </div>
</div>

<script>
const CAT_COLORS = {
  data:'#00d2ff', model:'#e94560', config:'#4ecca3', vector:'#7b68ee',
  checkpoint:'#f0c040', source:'#c0c0c0', document:'#8be9fd', multimedia:'#ff79c6',
  skill:'#ffb86c', agent:'#50fa7b', secret:'#ff5555'
};

let currentTab = 'overview';
let overview = null;

function esc(s) { if (!s) return ''; const d=document.createElement('div'); d.textContent=String(s); return d.innerHTML; }

function formatSize(b) {
  if (!b) return '0 B';
  const units = ['B','KB','MB','GB','TB'];
  let i=0; while (b>=1024 && i<units.length-1) { b/=1024; i++; }
  return i===0 ? b+' B' : b.toFixed(1)+' '+units[i];
}

function loadOverview() {
  fetch('/api/overview').then(r=>r.json()).then(data => {
    overview = data;
    render();
  });
}

function setTab(tab) { currentTab = tab; render(); }

function render() {
  if (!overview) return;
  const o = overview;
  let html = '';

  // Stats
  const critAlerts = (o.alert_counts.critical || 0);
  const warnAlerts = (o.alert_counts.warning || 0);
  html += `<div class="stat-grid">
    <div class="stat"><div class="stat-label">Nodes</div><div class="stat-value">${o.total_nodes}</div></div>
    <div class="stat"><div class="stat-label">Total AI Files</div><div class="stat-value">${o.total_files.toLocaleString()}</div></div>
    <div class="stat"><div class="stat-label">Total Size</div><div class="stat-value">${formatSize(o.total_size)}</div></div>
    <div class="stat"><div class="stat-label">Critical Alerts</div><div class="stat-value ${critAlerts?'danger':''}">${critAlerts}</div></div>
    <div class="stat"><div class="stat-label">Warnings</div><div class="stat-value ${warnAlerts?'warn':''}">${warnAlerts}</div></div>
    <div class="stat"><div class="stat-label">Recent Changes</div><div class="stat-value">${o.recent_changes.length}</div></div>
  </div>`;

  // Tabs
  html += `<div class="tab-bar">
    <div class="tab ${currentTab==='overview'?'active':''}" onclick="setTab('overview')">Overview</div>
    <div class="tab ${currentTab==='nodes'?'active':''}" onclick="setTab('nodes')">Nodes (${o.total_nodes})</div>
    <div class="tab ${currentTab==='alerts'?'active':''}" onclick="setTab('alerts')">Alerts (${critAlerts+warnAlerts})</div>
    <div class="tab ${currentTab==='changes'?'active':''}" onclick="setTab('changes')">Changes (${o.recent_changes.length})</div>
  </div>`;

  if (currentTab === 'overview') {
    html += renderOverview(o);
  } else if (currentTab === 'nodes') {
    html += renderNodes(o);
  } else if (currentTab === 'alerts') {
    html += renderAlerts(o);
  } else if (currentTab === 'changes') {
    html += renderChanges(o);
  }

  document.getElementById('app').innerHTML = html;
}

function renderOverview(o) {
  let html = '';

  // Category chart
  const cats = Object.entries(o.by_category).sort((a,b) => b[1].size - a[1].size);
  const maxSize = cats.length ? cats[0][1].size : 1;

  html += '<div class="chart-row">';
  html += '<div class="chart-box"><h3>AI Files by Category (All Nodes)</h3>';
  for (const [cat, info] of cats) {
    const pct = (info.size / maxSize * 100).toFixed(1);
    const color = CAT_COLORS[cat] || '#888';
    html += `<div class="bar-row">
      <div class="bar-label">${esc(cat)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="bar-value">${info.count} files</div>
    </div>`;
  }
  html += '</div>';

  // Size by category
  html += '<div class="chart-box"><h3>Storage by Category (All Nodes)</h3>';
  for (const [cat, info] of cats) {
    const pct = (info.size / maxSize * 100).toFixed(1);
    const color = CAT_COLORS[cat] || '#888';
    html += `<div class="bar-row">
      <div class="bar-label">${esc(cat)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="bar-value">${formatSize(info.size)}</div>
    </div>`;
  }
  html += '</div></div>';

  // Node summary cards
  html += '<div class="section"><div class="section-title">\uD83D\uDCBB Connected Nodes</div>';
  html += '<div class="node-grid">';
  for (const n of o.nodes) {
    html += `<div class="node-card" onclick="window.open('/api/node/${esc(n.machine_id)}','_blank')">
      <div class="node-header">
        <span class="node-name">\uD83D\uDCBB ${esc(n.hostname)}</span>
        <span class="node-os">${esc(n.os)} ${esc(n.architecture)}</span>
      </div>
      <div class="node-stats">
        <span>\uD83D\uDCC1 ${(n.latest_files||0).toLocaleString()} files</span>
        <span>\uD83D\uDCBE ${formatSize(n.latest_size)}</span>
        <span>\uD83D\uDCCA ${n.report_count} scans</span>
      </div>
      <div class="node-stats" style="margin-top:4px">
        <span>\uD83C\uDF10 ${esc(n.ip_address)}</span>
        <span>\uD83D\uDC64 ${esc(n.username)}</span>
        <span>\uD83D\uDD11 ${esc(n.machine_id)}</span>
      </div>
      <div class="node-last-seen">Last seen: ${esc(n.last_seen)} · Path: ${esc(n.latest_scan_path || '—')}</div>
    </div>`;
  }
  html += '</div></div>';

  return html;
}

function renderNodes(o) {
  let html = '<div class="section"><table>';
  html += '<tr><th>Hostname</th><th>OS</th><th>IP</th><th>User</th><th>Files</th><th>Size</th><th>Scans</th><th>Last Seen</th><th>Machine ID</th></tr>';
  for (const n of o.nodes) {
    html += `<tr>
      <td><strong>${esc(n.hostname)}</strong></td>
      <td>${esc(n.os)} ${esc(n.os_release)}</td>
      <td>${esc(n.ip_address)}</td>
      <td>${esc(n.username)}</td>
      <td>${(n.latest_files||0).toLocaleString()}</td>
      <td>${formatSize(n.latest_size)}</td>
      <td>${n.report_count}</td>
      <td>${esc(n.last_seen)}</td>
      <td><code>${esc(n.machine_id)}</code></td>
    </tr>`;
  }
  html += '</table></div>';
  return html;
}

function renderAlerts(o) {
  const alerts = o.recent_alerts;
  if (!alerts.length) return '<div class="empty">No alerts yet</div>';
  let html = '<div class="section"><table>';
  html += '<tr><th>Time</th><th>Node</th><th>Severity</th><th>Type</th><th>Message</th></tr>';
  for (const a of alerts) {
    html += `<tr>
      <td>${esc(a.created_at)}</td>
      <td><strong>${esc(a.hostname)}</strong></td>
      <td><span class="severity ${a.severity}">${esc(a.severity)}</span></td>
      <td>${esc(a.alert_type)}</td>
      <td>${esc(a.message)}</td>
    </tr>`;
  }
  html += '</table></div>';
  return html;
}

function renderChanges(o) {
  const changes = o.recent_changes;
  if (!changes.length) return '<div class="empty">No file changes detected yet</div>';
  let html = '<div class="section"><table>';
  html += '<tr><th>Time</th><th>Node</th><th>Change</th><th>File</th><th>Path</th></tr>';
  for (const c of changes) {
    html += `<tr>
      <td>${esc(c.detected_at)}</td>
      <td><strong>${esc(c.hostname)}</strong></td>
      <td><span class="change-type ${c.change_type}">${esc(c.change_type)}</span></td>
      <td>${esc(c.filename)}</td>
      <td style="font-size:12px;color:var(--text-dim)">${esc(c.file_path)}</td>
    </tr>`;
  }
  html += '</table></div>';
  return html;
}

// Auto-refresh every 30 seconds
loadOverview();
setInterval(loadOverview, 30000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class CentralHandler(BaseHTTPRequestHandler):
    db: CentralDB = None
    api_key: str = None  # Set at startup; None = no auth required

    def _check_auth(self) -> bool:
        """Validate API key for protected endpoints. Returns True if authorized."""
        if not self.api_key:
            return True  # No key configured — open access
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            if token == self.api_key:
                return True
        # Also accept X-API-Key header
        if self.headers.get("X-API-Key", "").strip() == self.api_key:
            return True
        self._send_error(401, "Unauthorized — valid API key required. "
                         "Use 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header.")
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/":
            self._send_html(DASHBOARD_HTML)
        elif path == "/api/overview":
            self._send_json(self.db.get_overview())
        elif path == "/api/nodes":
            self._send_json(self.db.get_nodes())
        elif path.startswith("/api/node/"):
            machine_id = path.split("/api/node/")[1]
            result = self.db.get_node(machine_id)
            if result:
                self._send_json(result)
            else:
                self._send_error(404, f"Node {machine_id} not found")
        elif path == "/api/alerts":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", [100])[0])
            self._send_json(self.db.get_alerts(limit))
        elif path == "/api/changes":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", [100])[0])
            self._send_json(self.db.get_changes(limit))
        elif path.startswith("/api/history/"):
            machine_id = path.split("/api/history/")[1]
            self._send_json(self.db.get_history(machine_id))
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/report":
            if not self._check_auth():
                return
            self._handle_report()
        else:
            self._send_error(404, "Not found")

    def _handle_report(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._send_error(400, "Invalid JSON body")
            return

        if not body.get("scan_metadata") and not body.get("files"):
            self._send_error(400, "Not a valid scan report (missing scan_metadata or files)")
            return

        try:
            result = self.db.ingest_report(body)
            self._send_json(result)
            sys.stderr.write(
                f"  \u2714 Report from {result['hostname']} "
                f"({result['machine_id']}): "
                f"{result['alerts_generated']} alerts, "
                f"{result['changes_detected']} changes\n"
            )
        except Exception as e:
            self._send_error(500, f"Failed to ingest report: {e}")

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj) -> None:
        data = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        data = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        pass  # Suppress default logging


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI File Discovery — Central Master Dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=8510,
        help="Port to serve on (default: 8510)",
    )
    parser.add_argument(
        "--db", default="central.db",
        help="SQLite database path (default: central.db)",
    )
    parser.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="Require API key for report submission. If set to 'generate', "
             "a random key will be created. If omitted, no auth is required.",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key
    if api_key == "generate":
        api_key = secrets.token_urlsafe(32)
    elif api_key and os.path.isfile(api_key):
        # Allow reading key from a file
        api_key = open(api_key).read().strip()

    db = CentralDB(args.db)
    CentralHandler.db = db
    CentralHandler.api_key = api_key

    server = HTTPServer(("0.0.0.0", args.port), CentralHandler)
    print(f"\n  AI File Discovery — Central Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Database: {os.path.abspath(args.db)}")
    if api_key:
        print(f"\n  API Key (required for POST /api/report):")
        print(f"    {api_key}")
        print(f"\n  Nodes push reports via:")
        print(f"    curl -X POST http://HOST:{args.port}/api/report \\")
        print(f"         -H 'Content-Type: application/json' \\")
        print(f"         -H 'Authorization: Bearer {api_key}' \\")
        print(f"         -d @report.json")
        print(f"\n  Or use the scanner directly:")
        print(f"    python3 aifiles.py /path --integrity --push http://HOST:{args.port} --api-key {api_key}")
    else:
        print(f"\n  ⚠  No API key configured — report submission is open (use --api-key to secure)")
        print(f"\n  Nodes push reports via:")
        print(f"    curl -X POST http://HOST:{args.port}/api/report \\")
        print(f"         -H 'Content-Type: application/json' -d @report.json")
        print(f"\n  Or use the scanner directly:")
        print(f"    python3 aifiles.py /path --integrity --push http://HOST:{args.port}")
    print(f"\n  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
