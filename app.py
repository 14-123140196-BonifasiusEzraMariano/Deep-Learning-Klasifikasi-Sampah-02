# -*- coding: utf-8 -*-
"""Web app inferensi klasifikasi sampah.

Jalankan:
    python app.py
Lalu buka: http://localhost:5000
"""
import base64
import io
import os
import sys

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template_string, request
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission"))
try:
    from submission.model import get_model, get_transform  # noqa: E402
except ImportError:
    from model import get_model, get_transform  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
CLASS_ICONS = {
    "botol_plastik":  "🍶",
    "gelas_plastik":  "🥤",
    "kertas_kardus":  "📦",
    "saset_kemasan":  "🛍️",
    "styrofoam":      "☁️",
}
CLASS_COLORS = {
    "botol_plastik":  "#3b82f6",
    "gelas_plastik":  "#8b5cf6",
    "kertas_kardus":  "#f59e0b",
    "saset_kemasan":  "#10b981",
    "styrofoam":      "#ef4444",
}
SUBMISSION_DIR = "./submission"

# ── Load model ───────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transform(train=False)

model = get_model(pretrained=False)
state = torch.load(os.path.join(SUBMISSION_DIR, "model.pth"), map_location="cpu")
model.load_state_dict(state)
model.to(device).eval()
print(f"[OK] Model loaded | Device: {device}")

# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>WasteAI — Klasifikasi Sampah</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0a0e1a;
      --surface:   #111827;
      --card:      #1a2235;
      --border:    rgba(255,255,255,0.07);
      --accent:    #6366f1;
      --accent2:   #818cf8;
      --text:      #f1f5f9;
      --muted:     #64748b;
      --success:   #10b981;
      --radius:    20px;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px 60px;
      background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.15), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,92,246,0.08), transparent);
    }

    header {
      text-align: center;
      margin-bottom: 48px;
    }
    header .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(99,102,241,0.15);
      border: 1px solid rgba(99,102,241,0.3);
      color: var(--accent2);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 18px;
    }
    header h1 {
      font-size: clamp(2rem, 5vw, 3rem);
      font-weight: 800;
      background: linear-gradient(135deg, #f1f5f9, var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      line-height: 1.15;
      margin-bottom: 12px;
    }
    header p {
      color: var(--muted);
      font-size: 15px;
      max-width: 440px;
      margin: 0 auto;
      line-height: 1.6;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 32px;
      width: 100%;
      max-width: 600px;
      box-shadow: 0 25px 50px rgba(0,0,0,0.4);
    }

    /* ── Drop Zone ── */
    #drop-zone {
      border: 2px dashed rgba(99,102,241,0.35);
      border-radius: 14px;
      padding: 56px 20px;
      text-align: center;
      cursor: pointer;
      transition: all .25s ease;
      position: relative;
      background: rgba(99,102,241,0.04);
    }
    #drop-zone:hover, #drop-zone.drag-over {
      border-color: var(--accent);
      background: rgba(99,102,241,0.09);
      transform: scale(1.01);
    }
    #drop-zone .icon-upload {
      font-size: 48px;
      margin-bottom: 14px;
      display: block;
      filter: drop-shadow(0 0 12px rgba(99,102,241,0.5));
    }
    #drop-zone h3 {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 6px;
    }
    #drop-zone p {
      font-size: 13px;
      color: var(--muted);
    }
    #drop-zone p span {
      color: var(--accent2);
      text-decoration: underline;
      cursor: pointer;
    }
    #file-input { display: none; }

    /* ── Preview ── */
    #preview-section { display: none; margin-top: 24px; }
    #preview-wrap {
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      background: #0a0e1a;
      display: flex;
      align-items: center;
      justify-content: center;
      max-height: 280px;
    }
    #preview-img {
      max-width: 100%;
      max-height: 280px;
      object-fit: contain;
      display: block;
      border-radius: 12px;
    }
    #change-btn {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(0,0,0,0.6);
      border: 1px solid var(--border);
      color: white;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
      cursor: pointer;
      backdrop-filter: blur(8px);
      transition: background .2s;
    }
    #change-btn:hover { background: rgba(255,255,255,0.15); }

    /* ── Button ── */
    #predict-btn {
      width: 100%;
      padding: 15px;
      margin-top: 20px;
      background: linear-gradient(135deg, var(--accent), #8b5cf6);
      border: none;
      border-radius: 12px;
      color: white;
      font-family: 'Inter', sans-serif;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: all .25s ease;
      letter-spacing: 0.02em;
      position: relative;
      overflow: hidden;
    }
    #predict-btn:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(99,102,241,0.45);
    }
    #predict-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    /* ── Loader ── */
    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin .7s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Result ── */
    #result-section {
      display: none;
      margin-top: 28px;
      animation: fadeUp .4s ease;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-header {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 20px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      margin-bottom: 16px;
    }
    .result-icon {
      font-size: 42px;
      line-height: 1;
    }
    .result-info { flex: 1; }
    .result-label {
      font-size: 20px;
      font-weight: 700;
      text-transform: capitalize;
      letter-spacing: 0.01em;
    }
    .result-conf {
      font-size: 13px;
      color: var(--muted);
      margin-top: 3px;
    }
    .result-badge {
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }

    .bars-title {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 12px;
    }
    .bar-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .bar-label {
      width: 130px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex-shrink: 0;
    }
    .bar-label.active { color: var(--text); font-weight: 600; }
    .bar-track {
      flex: 1;
      height: 7px;
      background: rgba(255,255,255,0.06);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.7s cubic-bezier(.4,0,.2,1);
      width: 0%;
    }
    .bar-pct {
      width: 42px;
      font-size: 12px;
      font-weight: 600;
      text-align: right;
      flex-shrink: 0;
    }
  </style>
</head>
<body>
  <header>
    <div class="badge">♻️ AI-Powered</div>
    <h1>WasteAI</h1>
    <p>Upload foto sampah, model ResNet-50 akan mengklasifikasikannya ke dalam 5 kategori secara otomatis.</p>
  </header>

  <div class="card">
    <!-- Drop Zone -->
    <div id="drop-zone">
      <span class="icon-upload">🗂️</span>
      <h3>Drag & drop gambar di sini</h3>
      <p>atau <span onclick="document.getElementById('file-input').click()">klik untuk memilih file</span></p>
      <p style="margin-top:8px; font-size:12px;">JPG, PNG, WEBP · Maks 10MB</p>
    </div>
    <input type="file" id="file-input" accept="image/*"/>

    <!-- Preview -->
    <div id="preview-section">
      <div id="preview-wrap">
        <img id="preview-img" src="" alt="Preview"/>
        <button id="change-btn" onclick="resetUI()">✕ Ganti</button>
      </div>
      <button id="predict-btn" onclick="runPredict()">🔍 Klasifikasikan Gambar</button>
    </div>

    <!-- Result -->
    <div id="result-section">
      <div class="result-header" id="result-header">
        <div class="result-icon" id="result-icon"></div>
        <div class="result-info">
          <div class="result-label" id="result-label"></div>
          <div class="result-conf" id="result-conf"></div>
        </div>
        <div class="result-badge" id="result-badge"></div>
      </div>
      <div class="bars-title">Distribusi Probabilitas</div>
      <div id="bars-container"></div>
    </div>
  </div>

  <script>
    const CLASSES = {{ classes|tojson }};
    const COLORS  = {{ colors|tojson }};
    const ICONS   = {{ icons|tojson }};

    let selectedFile = null;

    // Drag & drop
    const dz = document.getElementById('drop-zone');
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => {
      e.preventDefault(); dz.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith('image/')) handleFile(f);
    });
    dz.addEventListener('click', () => document.getElementById('file-input').click());

    document.getElementById('file-input').addEventListener('change', e => {
      if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = ev => {
        document.getElementById('preview-img').src = ev.target.result;
        document.getElementById('drop-zone').style.display = 'none';
        document.getElementById('preview-section').style.display = 'block';
        document.getElementById('result-section').style.display = 'none';
      };
      reader.readAsDataURL(file);
    }

    function resetUI() {
      selectedFile = null;
      document.getElementById('file-input').value = '';
      document.getElementById('drop-zone').style.display = 'block';
      document.getElementById('preview-section').style.display = 'none';
      document.getElementById('result-section').style.display = 'none';
    }

    async function runPredict() {
      if (!selectedFile) return;
      const btn = document.getElementById('predict-btn');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>Menganalisis...';

      const fd = new FormData();
      fd.append('image', selectedFile);

      try {
        const res  = await fetch('/predict', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) { alert('Error: ' + data.error); return; }
        showResult(data);
      } catch(e) {
        alert('Gagal menghubungi server.');
      } finally {
        btn.disabled = false;
        btn.innerHTML = '🔍 Klasifikasikan Gambar';
      }
    }

    function showResult(data) {
      const top      = data.predictions[0];
      const color    = COLORS[top.label];
      const icon     = ICONS[top.label];
      const pct      = (top.confidence * 100).toFixed(1);
      const labelFmt = top.label.replace(/_/g, ' ');

      document.getElementById('result-icon').textContent   = icon;
      document.getElementById('result-label').textContent  = labelFmt;
      document.getElementById('result-conf').textContent   = `Confidence: ${pct}%`;
      document.getElementById('result-header').style.borderColor = color + '44';
      document.getElementById('result-header').style.background  = color + '12';

      const badge = document.getElementById('result-badge');
      badge.textContent = pct + '%';
      badge.style.background = color + '22';
      badge.style.color      = color;

      // Bars
      const container = document.getElementById('bars-container');
      container.innerHTML = '';
      data.predictions.forEach(p => {
        const isTop = p.label === top.label;
        const pPct  = (p.confidence * 100).toFixed(1);
        const lbl   = p.label.replace(/_/g, ' ');
        const c     = COLORS[p.label];
        const row   = document.createElement('div');
        row.className = 'bar-row';
        row.innerHTML = `
          <div class="bar-label ${isTop ? 'active' : ''}">${lbl}</div>
          <div class="bar-track">
            <div class="bar-fill" data-w="${pPct}" style="background:${c}${isTop ? '' : '88'}"></div>
          </div>
          <div class="bar-pct" style="color:${isTop ? c : 'var(--muted)'};">${pPct}%</div>`;
        container.appendChild(row);
      });

      document.getElementById('result-section').style.display = 'block';

      // Animate bars
      requestAnimationFrame(() => {
        document.querySelectorAll('.bar-fill').forEach(el => {
          el.style.width = el.dataset.w + '%';
        });
      });
    }
  </script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(
        HTML,
        classes=CLASSES,
        colors=CLASS_COLORS,
        icons=CLASS_ICONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar."}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "File bukan gambar yang valid."}), 400

    with torch.no_grad():
        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu().tolist()

    predictions = sorted(
        [{"label": cls, "confidence": prob} for cls, prob in zip(CLASSES, probs)],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    print("[*] Server berjalan di http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
