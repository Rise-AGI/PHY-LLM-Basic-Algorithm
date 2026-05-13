"""
JSON 可视化浏览器 — 通用 JSON 文件查看器

左栏: 选择 JSON 文件 + 字段筛选
右栏: 自动排版显示（表格 / 键值卡 / LaTeX 数学渲染）

用法:
    python view_json.py                  # 默认端口 5000
    python view_json.py --port 8080      # 指定端口
"""

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(os.path.dirname(HERE), "json")

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JSON Viewer</title>
<script>
MathJax = {
  tex: { inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']] },
  startup: { typeset: false }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root {
  --bg: #f8f9fa; --card-bg: #fff; --border: #dee2e6; --text: #212529;
  --text-muted: #6c757d; --accent: #2563eb; --accent-hover: #1d4ed8;
  --danger: #dc2626; --success: #16a34a; --warn: #f59e0b;
  --sidebar-w: 260px; --topbar-h: 52px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Sidebar ── */
#sidebar { width: var(--sidebar-w); min-width: var(--sidebar-w); background: #1e293b; color: #e2e8f0; display: flex; flex-direction: column; overflow: hidden; }
#sidebar h2 { padding: 16px 16px 12px; font-size: 15px; font-weight: 600; letter-spacing: .3px; color: #f1f5f9; border-bottom: 1px solid #334155; }
#file-list { flex: 1; overflow-y: auto; padding: 8px 0; }
#file-list .file { display: block; padding: 8px 16px; cursor: pointer; font-size: 12.5px; border-left: 3px solid transparent; transition: all .15s; color: #cbd5e1; word-break: break-all; }
#file-list .file:hover { background: #334155; color: #f1f5f9; }
#file-list .file.active { background: #334155; border-left-color: #38bdf8; color: #fff; font-weight: 600; }
#file-list .file .badge { float: right; font-size: 10px; background: #475569; padding: 1px 7px; border-radius: 9px; margin-top: 1px; }
#file-list .file.active .badge { background: #38bdf8; color: #0f172a; }

/* ── Main ── */
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* Top bar */
#topbar { height: var(--topbar-h); min-height: var(--topbar-h); background: var(--card-bg); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 16px; gap: 10px; flex-wrap: wrap; overflow-y: auto; }
#topbar .label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; color: var(--text-muted); white-space: nowrap; }
#topbar .chip { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 14px; font-size: 11.5px; cursor: pointer; border: 1px solid var(--border); background: #fff; white-space: nowrap; transition: all .15s; user-select: none; }
#topbar .chip:hover { border-color: var(--accent); }
#topbar .chip.on { background: var(--accent); color: #fff; border-color: var(--accent); }
#topbar .chip .count { font-size: 9.5px; opacity: .75; }
#topbar #field-count { font-size: 11px; color: var(--text-muted); margin-left: auto; white-space: nowrap; }
#topbar #search { padding: 3px 10px; border: 1px solid var(--border); border-radius: 14px; font-size: 11.5px; width: 160px; outline: none; }
#topbar #search:focus { border-color: var(--accent); }

/* Content */
#content { flex: 1; overflow: auto; padding: 16px; }

/* Summary panel */
.summary { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.summary .stat { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 18px; min-width: 100px; }
.summary .stat .val { font-size: 22px; font-weight: 700; color: var(--accent); }
.summary .stat .key { font-size: 10.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; margin-top: 2px; }

/* Table */
.table-wrap { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { position: sticky; top: 0; background: #f1f5f9; text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--border); font-weight: 600; font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .3px; cursor: pointer; white-space: nowrap; user-select: none; }
th:hover { color: var(--text); }
th .sort-arrow { margin-left: 3px; font-size: 9px; }
td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: top; max-width: 500px; }
tr:hover td { background: #f8fafc; }
.cell-text { white-space: pre-wrap; word-break: break-word; max-height: 160px; overflow-y: auto; font-size: 11.5px; line-height: 1.55; }
.cell-bool-true { color: var(--success); font-weight: 600; }
.cell-bool-false { color: var(--danger); font-weight: 600; }
.cell-null { color: var(--text-muted); font-style: italic; }
.cell-number { font-variant-numeric: tabular-nums; text-align: right; display: block; }
.truncated { max-height: 120px; overflow: hidden; position: relative; }
.truncated::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 30px; background: linear-gradient(transparent, #fff); }
.expand-btn { color: var(--accent); cursor: pointer; font-size: 11px; margin-top: 2px; display: inline-block; }

/* JSON block (for nested objects) */
.json-block { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 5px; padding: 6px 10px; font: 11px/1.5 "SF Mono", "Fira Code", monospace; white-space: pre-wrap; word-break: break-all; max-height: 140px; overflow-y: auto; }

/* KV list */
.kv-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }
.kv-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 15px; }
.kv-card .k { font-size: 10.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .3px; margin-bottom: 4px; }
.kv-card .v { font-size: 13px; font-weight: 500; word-break: break-word; }
.kv-card .v.pre { font: 12px/1.5 "SF Mono", "Fira Code", monospace; white-space: pre-wrap; background: #f8fafc; padding: 6px 8px; border-radius: 4px; margin-top: 4px; }

/* Empty / loading */
.empty { text-align: center; color: var(--text-muted); padding: 80px 20px; font-size: 14px; }

/* Responsive */
@media (max-width: 768px) {
  #sidebar { width: 180px; min-width: 180px; }
  .kv-list { grid-template-columns: 1fr; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* Copy button */
.copy-btn { opacity: 0; font-size: 10px; margin-left: 4px; cursor: pointer; color: var(--accent); background: none; border: none; padding: 0 3px; }
tr:hover .copy-btn, .kv-card:hover .copy-btn { opacity: 1; }
</style>
</head>
<body>

<!-- ── Sidebar ── -->
<nav id="sidebar">
  <h2>JSON Viewer</h2>
  <div id="file-list"></div>
</nav>

<!-- ── Main ── -->
<div id="main">
  <div id="topbar">
    <span class="label">显示字段</span>
    <span id="field-chips"></span>
    <input id="search" type="text" placeholder="搜索过滤..." oninput="applyFilters()">
    <span id="field-count"></span>
  </div>
  <div id="content">
    <div class="empty">选择左侧 JSON 文件开始浏览</div>
  </div>
</div>

<script>
// ── State ──
let currentFile = null;
let currentData = null;       // parsed JSON root
let rows = [];                // normalized list of flat dicts
let allKeys = [];
let visibleKeys = new Set();
let sortKey = null;
let sortDir = 1;

// ── Init ──
fetch('/api/files')
  .then(r => r.json())
  .then(files => {
    const list = document.getElementById('file-list');
    if (!files.length) { list.innerHTML = '<div class="file" style="color:#94a3b8">(无 JSON 文件)</div>'; return; }
    files.forEach(f => {
      const el = document.createElement('div');
      el.className = 'file';
      el.innerHTML = `${f.name}<span class="badge">${f.label}</span>`;
      el.onclick = () => { loadFile(f.name); };
      list.appendChild(el);
    });
  });

// ── Load ──
function loadFile(name) {
  document.querySelectorAll('#file-list .file').forEach(el => el.classList.remove('active'));
  const target = [...document.querySelectorAll('#file-list .file')].find(el => el.textContent.startsWith(name));
  if (target) target.classList.add('active');
  currentFile = name;
  fetch('/api/data/' + encodeURIComponent(name))
    .then(r => r.json())
    .then(data => {
      currentData = data;
      normalize(data);
      visibleKeys = new Set(allKeys);
      renderFieldChips();
      render();
    });
}

// ── Normalize → flat rows ──
function normalize(data) {
  if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object' && data[0] !== null) {
    rows = data.map(flatten);
  } else if (data && typeof data === 'object' && !Array.isArray(data)) {
    // check for array fields
    const arrayFields = [];
    for (const [k, v] of Object.entries(data)) {
      if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'object' && v[0] !== null) {
        arrayFields.push(k);
      }
    }
    if (arrayFields.length === 1) {
      // use the largest array
      rows = data[arrayFields[0]].map(flatten);
    } else if (arrayFields.length > 1) {
      // pick the largest
      let best = arrayFields[0];
      for (const f of arrayFields) { if (data[f].length > data[best].length) best = f; }
      rows = data[best].map(flatten);
    } else {
      // no array — treat the object itself as one row
      rows = [flatten(data)];
    }
  } else {
    rows = [];
  }
  // collect all keys
  const keySet = new Set();
  rows.forEach(r => Object.keys(r).forEach(k => keySet.add(k)));
  allKeys = [...keySet];
}

function flatten(obj, prefix = '') {
  const out = {};
  if (obj === null || typeof obj !== 'object') { out[prefix || 'value'] = obj; return out; }
  for (const [k, v] of Object.entries(obj)) {
    const fullKey = prefix ? prefix + '.' + k : k;
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(out, flatten(v, fullKey));
    } else if (Array.isArray(v)) {
      if (v.length > 0 && typeof v[0] === 'object' && v[0] !== null) {
        out[fullKey] = `[Array × ${v.length}]`;
      } else {
        out[fullKey] = JSON.stringify(v);
      }
    } else {
      out[fullKey] = v;
    }
  }
  return out;
}

// ── Field chips ──
function renderFieldChips() {
  const container = document.getElementById('field-chips');
  container.innerHTML = allKeys.map(k => {
    const on = visibleKeys.has(k);
    return `<span class="chip${on ? ' on' : ''}" onclick="toggleKey('${esc(k)}')" title="${esc(k)}">${esc(k)}</span>`;
  }).join('');
  document.getElementById('field-count').textContent = `${visibleKeys.size}/${allKeys.length} 字段`;
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/'/g,'&#39;'); }

function toggleKey(key) {
  if (visibleKeys.has(key)) { visibleKeys.delete(key); } else { visibleKeys.add(key); }
  renderFieldChips();
  render();
}

// ── Search ──
function applyFilters() {
  render(); // re-render applies search + field filter
}

// ── Sort ──
function setSort(key) {
  if (sortKey === key) { sortDir = -sortDir; } else { sortKey = key; sortDir = 1; }
  render();
}

// ── Render ──
function render() {
  const container = document.getElementById('content');
  if (!rows.length) {
    // fallback: show raw JSON as KV
    if (currentData && typeof currentData === 'object' && !Array.isArray(currentData)) {
      container.innerHTML = renderKV(currentData);
    } else {
      container.innerHTML = '<div class="empty">无表格数据 — 查看原始 JSON</div>';
    }
    return;
  }

  const search = document.getElementById('search').value.toLowerCase();
  let filtered = rows;
  if (search) {
    filtered = rows.filter(r => {
      return Object.values(r).some(v => {
        if (v === null || v === undefined) return false;
        return String(v).toLowerCase().includes(search);
      });
    });
  }

  // sort
  if (sortKey) {
    filtered = [...filtered].sort((a, b) => {
      let va = a[sortKey], vb = b[sortKey];
      if (va === null || va === undefined) va = '';
      if (vb === null || vb === undefined) vb = '';
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * sortDir;
      va = String(va); vb = String(vb);
      return va.localeCompare(vb, 'zh') * sortDir;
    });
  }

  const keys = allKeys.filter(k => visibleKeys.has(k));

  let html = '';
  // Summary stats if grading info present
  if (currentData && !Array.isArray(currentData) && typeof currentData === 'object') {
    const summ = currentData.summary || currentData;
    html += '<div class="summary">';
    for (const [k, v] of Object.entries(summ)) {
      if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
        html += `<div class="stat"><div class="val">${formatVal(v)}</div><div class="key">${esc(k)}</div></div>`;
      }
    }
    html += '</div>';
  }

  html += `<div class="table-wrap"><table><thead><tr>`;
  html += `<th style="width:40px">#</th>`;
  keys.forEach(k => {
    const arrow = sortKey === k ? (sortDir === 1 ? '▲' : '▼') : '';
    html += `<th onclick="setSort('${esc(k)}')">${esc(k)}<span class="sort-arrow">${arrow}</span></th>`;
  });
  html += `</tr></thead><tbody>`;

  filtered.forEach((row, i) => {
    html += `<tr><td style="color:var(--text-muted);font-size:10px">${i + 1}</td>`;
    keys.forEach(k => {
      html += `<td>${cellHtml(row[k])}</td>`;
    });
    html += '</tr>';
  });
  html += `</tbody></table></div>`;
  html += `<div style="padding:8px;color:var(--text-muted);font-size:11px">共 ${filtered.length} 条` + (search ? ` (已过滤)` : '') + `</div>`;

  container.innerHTML = html;

  // typeset LaTeX
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise([container]).catch(console.error);
  }
}

function renderKV(obj) {
  let html = '<div class="kv-list">';
  for (const [k, v] of Object.entries(obj)) {
    html += `<div class="kv-card"><div class="k">${esc(k)}</div>`;
    html += `<div class="v${(typeof v === 'string' && v.length > 80 ? ' pre' : '')}">${formatVal(v)}</div>`;
    html += '</div>';
  }
  html += '</div>';
  return html;
}

function cellHtml(val) {
  if (val === null || val === undefined) return '<span class="cell-null">null</span>';
  if (typeof val === 'boolean') return `<span class="cell-bool-${val}">${val}</span>`;
  if (typeof val === 'number') return `<span class="cell-number">${val}</span>`;
  let s = String(val);
  if (s.length <= 60) return `<span class="cell-text">${esc(s)}</span>`;
  // Long text: truncate with expand
  const id = 'c' + Math.random().toString(36).slice(2, 9);
  return `<div class="truncated" id="${id}"><span class="cell-text">${esc(s)}</span></div><span class="expand-btn" onclick="toggleExpand('${id}')">展开全部 (${s.length} 字符)</span>`;
}

function formatVal(v) {
  if (v === null || v === undefined) return '<span class="cell-null">null</span>';
  if (typeof v === 'boolean') return `<span class="cell-bool-${v}">${v}</span>`;
  if (typeof v === 'number') return String(v);
  const s = String(v);
  if (s.length <= 120) return esc(s);
  return `<span class="cell-text">${esc(s)}</span>`;
}

function toggleExpand(id) {
  const el = document.getElementById(id);
  if (el.classList.contains('truncated')) {
    el.classList.remove('truncated');
    el.nextElementSibling.textContent = '收起';
  } else {
    el.classList.add('truncated');
    el.nextElementSibling.textContent = '展开全部 (' + el.querySelector('.cell-text').textContent.length + ' 字符)';
  }
}
</script>
</body>
</html>'''

# ── Flask app ────────────────────────────────────────────────

def create_app(json_dir: str):
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route('/')
    def index():
        return HTML

    @app.route('/api/files')
    def list_files():
        files = []
        if os.path.isdir(json_dir):
            for name in sorted(os.listdir(json_dir)):
                if name.endswith('.json'):
                    fpath = os.path.join(json_dir, name)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except Exception:
                        continue
                    label = _describe(data)
                    files.append({"name": name, "label": label})
        return jsonify(files)

    @app.route('/api/data/<path:name>')
    def get_data(name):
        safe = os.path.basename(name)
        if not safe.endswith('.json'):
            return jsonify({"error": "invalid"}), 400
        fpath = os.path.join(json_dir, safe)
        if not os.path.isfile(fpath):
            return jsonify({"error": "not found"}), 404
        with open(fpath, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))

    return app


def _describe(data) -> str:
    """简短描述 JSON 结构。"""
    if isinstance(data, list):
        return f"[{len(data)}]"
    if isinstance(data, dict):
        # check for summary + results pattern
        if "summary" in data and "results" in data:
            return f"summary + [{len(data['results'])}]"
        if "models" in data and isinstance(data.get("models"), list):
            return f"models × {len(data['models'])}"
        return f"{{{len(data)}}}"
    return "?"


def main():
    parser = argparse.ArgumentParser(description="JSON 可视化浏览器")
    parser.add_argument("--port", type=int, default=5000, help="端口 (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址 (default: 127.0.0.1)")
    parser.add_argument("--json-dir", default=JSON_DIR, help=f"JSON 目录 (default: {JSON_DIR})")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    json_dir = args.json_dir

    app = create_app(json_dir)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  JSON Viewer → {url}")
    print(f"  数据目录: {json_dir}")
    print(f"  按 Ctrl+C 退出\n")
    if not args.no_browser:
        webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
