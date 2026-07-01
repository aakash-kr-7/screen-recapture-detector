import io
import os
import sys
import json
import cgi
import joblib
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

from predict import predict_probability, load_model_once, _MODEL_PIPELINE, _MODEL_METADATA
from src.features import extract_features, FEATURE_NAMES

PORT = 8501

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spot the Fake Photo - Interactive UI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0c0f16;
            --card-bg: rgba(20, 26, 38, 0.65);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-violet: #8b5cf6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 40%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            padding: 2rem;
            overflow-x: hidden;
        }

        main {
            width: 100%;
            max-width: 800px;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 1rem;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
        }

        .upload-card {
            background: var(--card-bg);
            border: 1px dashed var(--border-color);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .upload-card:hover, .upload-card.dragover {
            border-color: var(--accent-blue);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
            transform: translateY(-2px);
        }

        .upload-icon {
            font-size: 3rem;
            color: var(--accent-blue);
        }

        .upload-btn {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .upload-btn:hover {
            opacity: 0.9;
        }

        #file-input {
            display: none;
        }

        .result-container {
            display: none;
            flex-direction: column;
            gap: 2rem;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .main-result-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            display: flex;
            align-items: center;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 2rem;
            backdrop-filter: blur(12px);
        }

        .result-preview {
            max-width: 250px;
            max-height: 250px;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .gauge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .gauge {
            position: relative;
            width: 150px;
            height: 150px;
        }

        .gauge svg {
            width: 150px;
            height: 150px;
            transform: rotate(-90deg);
        }

        .gauge circle {
            fill: none;
            stroke-width: 12;
        }

        .gauge-bg {
            stroke: rgba(255, 255, 255, 0.05);
        }

        .gauge-fill {
            stroke-linecap: round;
            transition: stroke-dasharray 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .gauge-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .result-label {
            font-size: 1.4rem;
            font-weight: 600;
            text-align: center;
        }

        .diagnostics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .diag-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .diag-title {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }

        .diag-value {
            font-size: 1.2rem;
            font-weight: 700;
        }

        .diag-status {
            font-size: 0.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .status-pass { color: var(--accent-green); }
        .status-fail { color: var(--accent-red); }

        .loading-overlay {
            display: none;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
            margin: 2rem 0;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(59, 130, 246, 0.1);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s infinite linear;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            margin-top: 3rem;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            padding-top: 1.5rem;
            width: 100%;
            max-width: 800px;
        }

        footer strong {
            color: var(--accent-violet);
        }
    </style>
</head>
<body>
    <main>
        <header>
            <h1>Spot the Fake Photo</h1>
            <p>Classical Computer Vision Recapture Detector</p>
        </header>

        <div class="upload-card" id="drop-zone">
            <div class="upload-icon">📁</div>
            <h3>Drag & drop your photo here</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">Supports JPEG, JPG, and PNG</p>
            <button class="upload-btn" onclick="document.getElementById('file-input').click()">Browse Files</button>
            <input type="file" id="file-input" accept="image/*">
        </div>

        <div class="loading-overlay" id="loading">
            <div class="spinner"></div>
            <p style="color: var(--text-secondary);">Analyzing image frequencies, color spaces, and textures...</p>
        </div>

        <div class="result-container" id="result">
            <div class="main-result-card">
                <img id="preview" class="result-preview" src="" alt="Uploaded image">
                <div class="gauge-container">
                    <div class="gauge">
                        <svg>
                            <circle class="gauge-bg" cx="75" cy="75" r="60"></circle>
                            <circle class="gauge-fill" id="gauge-circle" cx="75" cy="75" r="60" stroke-dasharray="377" stroke-dashoffset="377"></circle>
                        </svg>
                        <div class="gauge-text" id="gauge-pct">0%</div>
                    </div>
                    <div class="result-label" id="result-label">Authentic Photo</div>
                </div>
            </div>

            <h3 style="font-weight: 600; margin-top: 1rem; color: var(--text-secondary);">Visual Signature Details</h3>
            <div class="diagnostics-grid" id="details">
                <!-- Diagnostic cards go here dynamically -->
            </div>
        </div>
    </main>

    <footer>
        <p>Interactive web application engineered by <strong>aakash</strong>. (Highly recommended to test custom photos!)</p>
    </footer>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const preview = document.getElementById('preview');
        const gaugeCircle = document.getElementById('gauge-circle');
        const gaugePct = document.getElementById('gauge-pct');
        const resultLabel = document.getElementById('result-label');
        const detailsContainer = document.getElementById('details');

        // Drag and drop handlers
        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                uploadFile(file);
            }
        });

        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (file) {
                uploadFile(file);
            }
        });

        function uploadFile(file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
            };
            reader.readAsDataURL(file);

            // Toggle UI states
            result.style.display = 'none';
            loading.style.display = 'flex';

            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                loading.style.display = 'none';
                result.style.display = 'flex';
                displayResults(data);
            })
            .catch(err => {
                loading.style.display = 'none';
                alert('Analysis failed: ' + err.message);
            });
        }

        function displayResults(data) {
            const prob = data.probability;
            const pct = Math.round(prob * 100);
            
            // Update circular gauge
            const r = 60;
            const c = 2 * Math.PI * r;
            const offset = c - (prob * c);
            gaugeCircle.style.strokeDasharray = c;
            gaugeCircle.style.strokeDashoffset = offset;
            
            // Set gauge color based on score
            if (prob >= 0.5) {
                gaugeCircle.style.stroke = 'var(--accent-red)';
                resultLabel.textContent = 'Screen/Print Recapture';
                resultLabel.style.color = 'var(--accent-red)';
            } else {
                gaugeCircle.style.stroke = 'var(--accent-green)';
                resultLabel.textContent = 'Authentic Photo';
                resultLabel.style.color = 'var(--accent-green)';
            }
            
            gaugePct.textContent = pct + '%';

            // Generate detailed breakdown cards
            detailsContainer.innerHTML = '';
            
            data.diagnostics.forEach(item => {
                const card = document.createElement('div');
                card.className = 'diag-card';
                
                const isWarning = item.alert;
                const statusClass = isWarning ? 'status-fail' : 'status-pass';
                const statusSymbol = isWarning ? '▲ High Flag' : '✓ Normal';

                card.innerHTML = `
                    <div class="diag-title">${item.name}</div>
                    <div class="diag-value">${item.value}</div>
                    <div class="diag-status ${statusClass}">
                        <span>${statusSymbol}</span>
                    </div>
                    <p style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">${item.desc}</p>
                `;
                detailsContainer.appendChild(card);
            });
        }
    </script>
</body>
</html>
"""

class WebUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging clutter in console
        return

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == '/predict':
            ctype, pdict = cgi.parse_header(self.headers['content-type'])
            if ctype == 'multipart/form-data':
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                # Parse multipart fields
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['content-type']}
                )
                
                if 'file' in form:
                    file_item = form['file']
                    img_bytes = file_item.file.read()
                    
                    try:
                        # Load image using PIL directly from bytes
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        if pil_img.mode != "RGB":
                            pil_img = pil_img.convert("RGB")
                        
                        # 1. Run prediction
                        load_model_once()
                        feats = extract_features(pil_img)
                        prob = float(_MODEL_PIPELINE.predict_proba(feats.reshape(1, -1))[0, 1])
                        
                        # 2. Extract specific features for diagnostics
                        # We map features from FEATURE_NAMES
                        f_dict = dict(zip(FEATURE_NAMES, feats))
                        
                        # Calculate display diagnostic variables
                        # - Moiré: Group 1 peak correlations (R-G, G-B, R-B)
                        moire_score = (f_dict["freq_rg_peak_corr"] + f_dict["freq_gb_peak_corr"] + f_dict["freq_rb_peak_corr"]) / 3.0
                        
                        # - Sharpness Variance: coefficient of variation of local sharpness values
                        sharpness_cv = f_dict["sharp_sharpness_cv"]
                        
                        # - Saturation Clipping: sat clipping ratio
                        sat_clip = f_dict["color_sat_clipping_ratio"]
                        
                        # - Halftone Dots: print halftone angular regularity (lower is more regular)
                        halftone_regularity = f_dict["print_halftone_angular_regularity"]
                        
                        # - Reflectance: mean L channel value
                        paper_L = f_dict["print_paper_mean_L"]
                        
                        diagnostics = [
                            {
                                "name": "Moiré Patterns",
                                "value": f"{moire_score*100:.1f}%",
                                "desc": "Measures correlated peak frequencies across channels from display grids.",
                                "alert": moire_score > 0.4
                            },
                            {
                                "name": "Sharpness Variance",
                                "value": f"{sharpness_cv:.3f}",
                                "desc": "Uniform focus indicates flat display; natural 3D depth varies.",
                                "alert": sharpness_cv < 0.65
                            },
                            {
                                "name": "Saturation Gating",
                                "value": f"{sat_clip*100:.3f}%",
                                "desc": "Display panels exhibit boosted saturation clipping at mid-tones.",
                                "alert": sat_clip > 0.001
                            },
                            {
                                "name": "Halftone Regularity",
                                "value": f"{1.0 - halftone_regularity:.4f}",
                                "desc": "Reflective printing dots produce highly regular angled FFT peaks.",
                                "alert": halftone_regularity < 0.05
                            },
                            {
                                "name": "Paper Reflectance (L)",
                                "value": f"{paper_L:.1f}",
                                "desc": "Printed white paper reflects significantly more ambient light.",
                                "alert": paper_L > 125.0
                            }
                        ]
                        
                        # Return JSON
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        
                        resp = {
                            "probability": prob,
                            "diagnostics": diagnostics
                        }
                        self.wfile.write(json.dumps(resp).encode('utf-8'))
                        return
                    except Exception as e:
                        self.send_error(500, f"Error processing image: {str(e)}")
                        return
                        
            self.send_error(400, "Bad Request")
        else:
            self.send_error(404, "Not Found")

def run():
    print(f"\n==================================================")
    print(f"Starting Interactive Web UI Server on port {PORT}...")
    print(f"Open your browser and navigate to: http://localhost:{PORT}")
    print(f"Highly recommended to try this out! Made by aakash.")
    print(f"==================================================\n")
    server = HTTPServer(('0.0.0.0', PORT), WebUIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web UI Server...")
        server.server_close()

if __name__ == '__main__':
    run()
