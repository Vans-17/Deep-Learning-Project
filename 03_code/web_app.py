"""
Flask Web UI for LDFaceNet-Lite Face Swap.
Run: python web_app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import io
import os
import sys
import base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
from PIL import Image

from src.pipeline import init_models, run_warp_swap, run_pipeline
from src.utils import load_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Global models registry loaded once at startup
MODELS = None

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "04_data" / "sample_inputs"

def get_models():
    global MODELS
    if MODELS is None:
        print("Initialising models for web app...")
        MODELS = init_models()
    return MODELS

def decode_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image)

def image_to_base64(img_array: np.ndarray) -> str:
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🎭 LDFaceNet Face Swapper</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    .dropzone {
      border: 2px dashed #4b5563;
      transition: all 0.2s ease;
    }
    .dropzone.dragover {
      border-color: #6366f1;
      background-color: rgba(99, 102, 241, 0.08);
    }
  </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen font-sans antialiased">
  <div class="max-w-6xl mx-auto px-4 py-8">
    
    <!-- Header -->
    <header class="text-center mb-8">
      <div class="inline-flex items-center justify-center p-3 bg-indigo-600/20 text-indigo-400 rounded-2xl mb-3 border border-indigo-500/30">
        <i class="fa-solid fa-masks-theater text-3xl"></i>
      </div>
      <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400 bg-clip-text text-transparent">
        LDFaceNet 3D Face Swapper
      </h1>
      <p class="text-gray-400 mt-2 max-w-xl mx-auto text-sm sm:text-base">
        Upload any two faces to swap identity with 3D BFM geometry, custom target skin tone adaptation, and fairness control.
      </p>
    </header>

    <!-- Main Card -->
    <div class="bg-gray-900 border border-gray-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl">
      
      <!-- Upload Section -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        
        <!-- Source Face (Identity Donor) -->
        <div class="flex flex-col">
          <label class="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
            <span class="w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 inline-flex items-center justify-center text-xs font-bold">1</span>
            Source Face (Identity Donor)
          </label>
          <div id="src-dropzone" class="dropzone rounded-2xl p-4 flex flex-col items-center justify-center min-h-[240px] cursor-pointer bg-gray-950/50 hover:border-indigo-500 relative overflow-hidden group">
            <input type="file" id="src-file" accept="image/*" class="hidden">
            <div id="src-placeholder" class="text-center">
              <i class="fa-regular fa-image text-4xl text-gray-500 mb-2 group-hover:text-indigo-400 transition-colors"></i>
              <p class="text-sm text-gray-300 font-medium">Click or drag & drop</p>
              <p class="text-xs text-gray-500 mt-1">Upload any face image</p>
            </div>
            <img id="src-preview" class="hidden max-h-[220px] rounded-xl object-contain shadow-lg" />
          </div>
          <!-- Quick Sample Selection -->
          <div class="flex items-center gap-2 mt-2">
            <span class="text-xs text-gray-500">Quick Samples:</span>
            <button type="button" onclick="loadSample('src', 'AJ_Cook_0001.jpg')" class="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded-lg border border-gray-700 transition-all">AJ Cook</button>
            <button type="button" onclick="loadSample('src', 'shaktiman.jpg')" class="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded-lg border border-gray-700 transition-all">Shaktiman</button>
          </div>
        </div>

        <!-- Target Face (Pose / Background Donor) -->
        <div class="flex flex-col">
          <label class="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
            <span class="w-6 h-6 rounded-full bg-purple-500/20 text-purple-400 inline-flex items-center justify-center text-xs font-bold">2</span>
            Target Face (Pose & Body Donor)
          </label>
          <div id="tgt-dropzone" class="dropzone rounded-2xl p-4 flex flex-col items-center justify-center min-h-[240px] cursor-pointer bg-gray-950/50 hover:border-purple-500 relative overflow-hidden group">
            <input type="file" id="tgt-file" accept="image/*" class="hidden">
            <div id="tgt-placeholder" class="text-center">
              <i class="fa-regular fa-image text-4xl text-gray-500 mb-2 group-hover:text-purple-400 transition-colors"></i>
              <p class="text-sm text-gray-300 font-medium">Click or drag & drop</p>
              <p class="text-xs text-gray-500 mt-1">Upload any face image</p>
            </div>
            <img id="tgt-preview" class="hidden max-h-[220px] rounded-xl object-contain shadow-lg" />
          </div>
          <!-- Quick Sample Selection -->
          <div class="flex items-center gap-2 mt-2">
            <span class="text-xs text-gray-500">Quick Samples:</span>
            <button type="button" onclick="loadSample('tgt', 'Adrien_Brody_0003.jpg')" class="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded-lg border border-gray-700 transition-all">Adrien Brody</button>
          </div>
        </div>

      </div>

      <!-- Controls & Sliders -->
      <div class="bg-gray-950/60 border border-gray-800 rounded-2xl p-5 mb-8">
        <h3 class="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
          <i class="fa-solid fa-sliders text-indigo-400"></i> Pipeline Customization
        </h3>
        
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          
          <!-- Method Selection -->
          <div>
            <label class="block text-xs font-medium text-gray-400 mb-1.5">Swap Pipeline Method</label>
            <select id="swap-method" class="w-full bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-indigo-500">
              <option value="warp" selected>Warp Swap (Fast, Exact Identity)</option>
              <option value="blend">Blend (Diffusion + Mask Composite)</option>
              <option value="hybrid">Hybrid (Warp seeded into VAE)</option>
              <option value="diffusion">Pure Diffusion Img2Img</option>
            </select>
          </div>

          <!-- Target Skin Tone Influence -->
          <div>
            <div class="flex justify-between items-center text-xs font-medium text-gray-400 mb-1.5">
              <span>Target Skin Tone Influence</span>
              <span id="val-skin-weight" class="text-indigo-400 font-mono">0.80</span>
            </div>
            <input id="target-skin-weight" type="range" min="0" max="1" step="0.05" value="0.80" 
              class="w-full accent-indigo-500 cursor-pointer"
              oninput="document.getElementById('val-skin-weight').textContent = parseFloat(this.value).toFixed(2)">
            <p class="text-[11px] text-gray-500 mt-1">0 = Keep source complexion | 1 = Full target skin tone</p>
          </div>

          <!-- Skin Lightness (Fairness) -->
          <div>
            <div class="flex justify-between items-center text-xs font-medium text-gray-400 mb-1.5">
              <span>Skin Lightness (Brighten)</span>
              <span id="val-lightness" class="text-pink-400 font-mono">0.25</span>
            </div>
            <input id="skin-lightness" type="range" min="0" max="0.8" step="0.05" value="0.25" 
              class="w-full accent-pink-500 cursor-pointer"
              oninput="document.getElementById('val-lightness').textContent = parseFloat(this.value).toFixed(2)">
            <p class="text-[11px] text-gray-500 mt-1">Lightens and clarifies the swapped face</p>
          </div>

        </div>
      </div>

      <!-- Action Button -->
      <div class="flex justify-center mb-8">
        <button id="btn-swap" onclick="executeSwap()" 
          class="w-full sm:w-auto px-10 py-3.5 bg-gradient-to-r from-indigo-500 via-purple-600 to-pink-500 hover:from-indigo-600 hover:to-pink-600 font-semibold rounded-2xl shadow-lg shadow-indigo-500/25 active:scale-95 transition-all flex items-center justify-center gap-3">
          <i class="fa-solid fa-wand-magic-sparkles"></i>
          <span>Swap Face Now</span>
        </button>
      </div>

      <!-- Loading State -->
      <div id="loading-spinner" class="hidden text-center py-10">
        <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-indigo-500 border-t-transparent mb-3"></div>
        <p class="text-sm font-medium text-gray-300">Extracting 3DMM landmarks & generating face swap...</p>
        <p class="text-xs text-gray-500 mt-1">This takes ~2-4 seconds for warp mode</p>
      </div>

      <!-- Results Section -->
      <div id="results-area" class="hidden border-t border-gray-800 pt-8">
        <div class="flex flex-col sm:flex-row justify-between items-center mb-6 gap-2">
          <h2 class="text-xl font-bold text-gray-100 flex items-center gap-2">
            <i class="fa-solid fa-circle-check text-emerald-400"></i> Swapped Result
          </h2>
          <div id="id-sim-badge" class="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs px-3 py-1.5 rounded-full font-mono font-medium">
            ArcFace ID Similarity: --
          </div>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-3 gap-6 items-center">
          
          <div class="bg-gray-950/70 border border-gray-800 rounded-2xl p-3 text-center">
            <p class="text-xs text-gray-400 mb-2 font-medium">Source (Identity)</p>
            <img id="res-src" class="w-full max-h-64 object-contain rounded-xl" />
          </div>

          <div class="bg-gray-950/70 border border-gray-800 rounded-2xl p-3 text-center">
            <p class="text-xs text-gray-400 mb-2 font-medium">Target (Pose & Context)</p>
            <img id="res-tgt" class="w-full max-h-64 object-contain rounded-xl" />
          </div>

          <div class="bg-gray-950/70 border-2 border-indigo-500/40 rounded-2xl p-3 text-center shadow-lg shadow-indigo-500/10">
            <p class="text-xs text-indigo-400 mb-2 font-semibold">Swapped Output</p>
            <img id="res-output" class="w-full max-h-64 object-contain rounded-xl" />
          </div>

        </div>

        <div class="flex justify-center mt-6">
          <a id="btn-download" download="swapped_face.png" class="px-6 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700 text-sm font-medium rounded-xl flex items-center gap-2 transition-all">
            <i class="fa-solid fa-download"></i> Download High-Res Result
          </a>
        </div>

      </div>

    </div>

  </div>

  <script>
    let srcDataUrl = null;
    let tgtDataUrl = null;

    function setupDropzone(zoneId, inputId, previewId, placeholderId, isSrc) {
      const zone = document.getElementById(zoneId);
      const input = document.getElementById(inputId);
      const preview = document.getElementById(previewId);
      const placeholder = document.getElementById(placeholderId);

      zone.onclick = () => input.click();

      zone.ondragover = (e) => { e.preventDefault(); zone.classList.add('dragover'); };
      zone.ondragleave = () => zone.classList.remove('dragover');
      zone.ondrop = (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], preview, placeholder, isSrc);
      };

      input.onchange = () => {
        if (input.files.length) handleFile(input.files[0], preview, placeholder, isSrc);
      };
    }

    function handleFile(file, preview, placeholder, isSrc) {
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        placeholder.classList.add('hidden');
        if (isSrc) srcDataUrl = e.target.result;
        else tgtDataUrl = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    setupDropzone('src-dropzone', 'src-file', 'src-preview', 'src-placeholder', true);
    setupDropzone('tgt-dropzone', 'tgt-file', 'tgt-preview', 'tgt-placeholder', false);

    async function loadSample(type, filename) {
      try {
        const res = await fetch(`/api/sample/${filename}`);
        const data = await res.json();
        if (data.image) {
          const preview = document.getElementById(`${type}-preview`);
          const placeholder = document.getElementById(`${type}-placeholder`);
          preview.src = data.image;
          preview.classList.remove('hidden');
          placeholder.classList.add('hidden');
          if (type === 'src') srcDataUrl = data.image;
          else tgtDataUrl = data.image;
        }
      } catch (err) {
        alert("Failed to load sample: " + err);
      }
    }

    async function executeSwap() {
      if (!srcDataUrl || !tgtDataUrl) {
        alert("Please select or upload both Source and Target images first!");
        return;
      }

      const btn = document.getElementById('btn-swap');
      const spinner = document.getElementById('loading-spinner');
      const resultsArea = document.getElementById('results-area');

      btn.disabled = true;
      spinner.classList.remove('hidden');
      resultsArea.classList.add('hidden');

      const payload = {
        source: srcDataUrl,
        target: tgtDataUrl,
        method: document.getElementById('swap-method').value,
        target_skin_weight: parseFloat(document.getElementById('target-skin-weight').value),
        skin_lightness: parseFloat(document.getElementById('skin-lightness').value)
      };

      try {
        const response = await fetch('/api/swap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok || result.error) {
          throw new Error(result.error || "Swap failed on server.");
        }

        document.getElementById('res-src').src = srcDataUrl;
        document.getElementById('res-tgt').src = tgtDataUrl;
        document.getElementById('res-output').src = result.output;
        document.getElementById('btn-download').href = result.output;
        
        const simStr = result.id_sim !== null ? result.id_sim.toFixed(4) : "N/A";
        document.getElementById('id-sim-badge').textContent = `ArcFace ID Similarity: ${simStr}`;

        resultsArea.classList.remove('hidden');
      } catch (err) {
        alert("Error during swap: " + err.message);
      } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
      }
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/sample/<filename>')
def get_sample(filename):
    safe_path = (SAMPLE_DIR / filename).resolve()
    if safe_path.exists() and safe_path.is_relative_to(SAMPLE_DIR):
        img = load_image(str(safe_path))
        return jsonify({"image": image_to_base64(img)})
    return jsonify({"error": "File not found"}), 404

@app.route('/api/swap', methods=['POST'])
def swap():
    data = request.get_json()
    if not data or 'source' not in data or 'target' not in data:
        return jsonify({"error": "Missing source or target image"}), 400

    try:
        # Decode base64 images
        src_b64 = data['source'].split(',')[-1]
        tgt_b64 = data['target'].split(',')[-1]
        
        src_bytes = base64.b64decode(src_b64)
        tgt_bytes = base64.b64decode(tgt_b64)

        src_img = decode_image_from_bytes(src_bytes)
        tgt_img = decode_image_from_bytes(tgt_bytes)

        method = data.get('method', 'warp')
        target_skin_weight = float(data.get('target_skin_weight', 0.8))
        skin_lightness = float(data.get('skin_lightness', 0.25))

        models = get_models()

        warp_kw = dict(
            warp_method='global',
            warp_scale=1.0,
            pw_weight=0.0,
            warp_dy=0.0,
            target_skin_weight=target_skin_weight,
            skin_lightness=skin_lightness,
        )

        if method == 'warp':
            output = run_warp_swap(src_img, tgt_img, models=models, **warp_kw)
        elif method == 'hybrid':
            swapped = run_warp_swap(src_img, tgt_img, models=models, verbose=False, **warp_kw)
            cur = run_pipeline(src_img, tgt_img, models=models, init_img=swapped['result'],
                               verbose=False, strength=0.45)['result']
            output = {'result': cur, 'identity_sim': None}
            try:
                src_id = models['arcface'].extract(src_img)
                res_id = models['arcface'].extract(cur)
                output['identity_sim'] = float(models['arcface'].cosine_sim(src_id, res_id))
            except Exception:
                pass
        elif method == 'blend':
            swapped = run_warp_swap(src_img, tgt_img, models=models, verbose=False, **warp_kw)
            diffused = run_pipeline(src_img, tgt_img, models=models, init_img=swapped['result'],
                                    verbose=False, strength=0.45)['result']
            d = diffused
            s = cv2.resize(swapped['result'], (d.shape[1], d.shape[0]))
            mask = cv2.resize(swapped['face_mask'].astype(np.float32), (d.shape[1], d.shape[0]))[..., None]
            res = (s.astype(np.float32) * mask + d.astype(np.float32) * (1 - mask))
            output = {'result': res.astype(np.uint8), 'identity_sim': None}
            try:
                src_id = models['arcface'].extract(src_img)
                res_id = models['arcface'].extract(output['result'])
                output['identity_sim'] = float(models['arcface'].cosine_sim(src_id, res_id))
            except Exception:
                pass
        else:
            output = run_pipeline(src_img, tgt_img, models=models)

        out_b64 = image_to_base64(output['result'])
        sim_val = float(output['identity_sim']) if output.get('identity_sim') is not None else None

        return jsonify({
            "output": out_b64,
            "id_sim": sim_val
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LDFaceNet Web App")
    parser.add_argument('--port', type=int, default=7860, help="Port to run server on (default: 7860)")
    args = parser.parse_args()

    # Preload models
    get_models()
    port = args.port or int(os.environ.get('PORT', 7860))
    print(f"\n🚀 Server ready! Open your browser at: http://127.0.0.1:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
