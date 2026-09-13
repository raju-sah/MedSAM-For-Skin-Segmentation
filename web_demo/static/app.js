/**
 * CG-MedSAM Web Demo Client Logic
 * Handles image rendering, interactive bounding-box drawing, API calls, and telemetry HUD.
 */

// Application State
const state = {
  currentImage: null,        // HTMLImageElement
  currentFile: null,         // File or Blob
  imageDimensions: { width: 0, height: 0 },
  bbox: [0, 0, 100, 100],    // [x1, y1, x2, y2]
  selectedModel: 'cg_adapter',
  viewMode: 'overlay',       // 'overlay', 'split', 'mask', 'original'
  showCore: true,
  showRing: true,
  isDrawing: false,
  isDraggingHandle: null,    // null or 'tl', 'tr', 'bl', 'br', 'move'
  dragStart: { x: 0, y: 0 },
  origBboxAtDrag: null,
  results: null              // Latest response from /api/segment
};

// DOM References
const elements = {
  statusPill: document.getElementById('statusPill'),
  statusLabel: document.getElementById('statusLabel'),
  sampleCardsContainer: document.getElementById('sampleCardsContainer'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  btnAutoPrompt: document.getElementById('btnAutoPrompt'),
  btnResetBox: document.getElementById('btnResetBox'),
  btnRunSegment: document.getElementById('btnRunSegment'),
  segmentBtnText: document.getElementById('segmentBtnText'),
  segmentSpinner: document.getElementById('segmentSpinner'),
  inputX1: document.getElementById('inputX1'),
  inputY1: document.getElementById('inputY1'),
  inputX2: document.getElementById('inputX2'),
  inputY2: document.getElementById('inputY2'),
  mainCanvas: document.getElementById('mainCanvas'),
  canvasHint: document.getElementById('canvasHint'),
  imgDimText: document.getElementById('imgDimText'),
  viewModeGroup: document.getElementById('viewModeGroup'),
  toggleCoreBox: document.getElementById('toggleCoreBox'),
  toggleRingBox: document.getElementById('toggleRingBox'),
  btnDownloadMask: document.getElementById('btnDownloadMask'),
  btnDownloadOverlay: document.getElementById('btnDownloadOverlay'),
  btnDownloadTelemetry: document.getElementById('btnDownloadTelemetry'),
  // Telemetry DOM
  deltaENum: document.getElementById('deltaENum'),
  deltaEBar: document.getElementById('deltaEBar'),
  deltaEFootnote: document.getElementById('deltaEFootnote'),
  gammaNum: document.getElementById('gammaNum'),
  gammaBadge: document.getElementById('gammaBadge'),
  gammaExplanation: document.getElementById('gammaExplanation'),
  skinCohortChip: document.getElementById('skinCohortChip'),
  toneCircle: document.getElementById('toneCircle'),
  toneName: document.getElementById('toneName'),
  skinLuminanceVal: document.getElementById('skinLuminanceVal'),
  itaVal: document.getElementById('itaVal'),
  lesionAreaVal: document.getElementById('lesionAreaVal'),
  lesionCoverageVal: document.getElementById('lesionCoverageVal'),
  promptTypeVal: document.getElementById('promptTypeVal'),
  backendVal: document.getElementById('backendVal')
};

const ctx = elements.mainCanvas.getContext('2d');

// Initialize
async function initApp() {
  setupEventListeners();
  await checkSystemStatus();
  await loadSampleCards();
}

// System Status Check
async function checkSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.cuda_available) {
      elements.statusLabel.textContent = `CUDA Online (${data.gpu_name})`;
    } else {
      elements.statusLabel.textContent = `CPU Execution (${data.base_medsam_checkpoint_present ? 'Real MedSAM' : 'Mock Foundation'})`;
    }
    if (data.base_medsam_checkpoint_present) {
      elements.backendVal.textContent = data.cuda_available ? 'CUDA / MedSAM-ViT' : 'CPU / MedSAM-ViT';
    } else {
      elements.backendVal.textContent = 'Mock Foundation';
    }
  } catch (err) {
    elements.statusLabel.textContent = 'Backend Offline';
  }
}

// Load Samples Gallery
async function loadSampleCards() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    elements.sampleCardsContainer.innerHTML = '';

    data.samples.forEach((sample, idx) => {
      const card = document.createElement('div');
      card.className = `sample-card ${idx === 0 ? 'active' : ''}`;
      const tagClass = sample.skin_type.includes('Dark') ? 'tag-dark' : (sample.skin_type.includes('Medium') ? 'tag-med' : 'tag-light');

      card.innerHTML = `
        <img class="sample-thumb" src="/api/samples/${sample.filename}" alt="${sample.name}">
        <div class="sample-info">
          <div class="sample-title">${sample.name}</div>
          <div class="sample-meta-row">
            <span class="sample-fst-tag ${tagClass}">${sample.skin_type}</span>
            <span style="font-size:10.5px;color:var(--text-muted);">&Delta;E* &asymp; ${sample.delta_e_expected}</span>
          </div>
        </div>
      `;

      card.addEventListener('click', async () => {
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        await loadSampleImage(sample);
      });

      elements.sampleCardsContainer.appendChild(card);

      if (idx === 0) {
        // Load first sample by default
        loadSampleImage(sample);
      }
    });
  } catch (err) {
    console.error('Failed to load sample cards:', err);
  }
}

// Load a specific sample image
async function loadSampleImage(sample) {
  try {
    const response = await fetch(`/api/samples/${sample.filename}`);
    const blob = await response.blob();
    const file = new File([blob], sample.filename, { type: 'image/jpeg' });
    await setImageSource(file, sample.suggested_bbox);
  } catch (err) {
    console.error('Error loading sample image:', err);
  }
}

// Set Image Source and reset canvas
async function setImageSource(file, presetBbox = null) {
  state.currentFile = file;
  state.results = null;

  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        state.currentImage = img;
        state.imageDimensions = { width: img.naturalWidth, height: img.naturalHeight };
        elements.imgDimText.textContent = `${img.naturalWidth} x ${img.naturalHeight} px`;

        // Configure canvas dimensions
        elements.mainCanvas.width = img.naturalWidth;
        elements.mainCanvas.height = img.naturalHeight;

        if (presetBbox) {
          state.bbox = [...presetBbox];
        } else {
          // Default center 60% crop
          const w = img.naturalWidth;
          const h = img.naturalHeight;
          state.bbox = [
            Math.round(w * 0.20),
            Math.round(h * 0.20),
            Math.round(w * 0.80),
            Math.round(h * 0.80)
          ];
        }

        updateInputsFromState();
        renderCanvas();
        resolve();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

// Event Listeners setup
function setupEventListeners() {
  // Dropzone drag-and-drop
  elements.dropzone.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      setImageSource(e.target.files[0]);
    }
  });

  elements.dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropzone.classList.add('dragover');
  });

  elements.dropzone.addEventListener('dragleave', () => {
    elements.dropzone.classList.remove('dragover');
  });

  elements.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setImageSource(e.dataTransfer.files[0]);
    }
  });

  // Model Selection
  document.querySelectorAll('.model-option').forEach(option => {
    option.addEventListener('click', () => {
      document.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
      option.classList.add('active');
      state.selectedModel = option.dataset.model;
      option.querySelector('input').checked = true;
    });
  });

  // View Mode Tabs
  elements.viewModeGroup.querySelectorAll('.tab-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      elements.viewModeGroup.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.viewMode = btn.dataset.view;
      renderCanvas();
    });
  });

  // Toggles for Core and Ring
  elements.toggleCoreBox.addEventListener('change', (e) => {
    state.showCore = e.target.checked;
    renderCanvas();
  });

  elements.toggleRingBox.addEventListener('change', (e) => {
    state.showRing = e.target.checked;
    renderCanvas();
  });

  // Auto-Detect Box
  elements.btnAutoPrompt.addEventListener('click', async () => {
    if (!state.currentFile) return;
    try {
      const formData = new FormData();
      formData.append('file', state.currentFile);
      elements.btnAutoPrompt.disabled = true;
      elements.btnAutoPrompt.textContent = 'Estimating...';

      const res = await fetch('/api/auto_prompt', { method: 'POST', body: formData });
      const data = await res.json();
      state.bbox = data.bbox;
      updateInputsFromState();
      renderCanvas();
    } catch (err) {
      console.error('Auto prompt error:', err);
    } finally {
      elements.btnAutoPrompt.disabled = false;
      elements.btnAutoPrompt.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
        Auto-Detect Box
      `;
    }
  });

  // Reset Box
  elements.btnResetBox.addEventListener('click', () => {
    if (!state.currentImage) return;
    const w = state.imageDimensions.width;
    const h = state.imageDimensions.height;
    state.bbox = [
      Math.round(w * 0.20),
      Math.round(h * 0.20),
      Math.round(w * 0.80),
      Math.round(h * 0.80)
    ];
    updateInputsFromState();
    renderCanvas();
  });

  // Coordinate Inputs
  [elements.inputX1, elements.inputY1, elements.inputX2, elements.inputY2].forEach(input => {
    input.addEventListener('change', () => {
      state.bbox = [
        parseInt(elements.inputX1.value) || 0,
        parseInt(elements.inputY1.value) || 0,
        parseInt(elements.inputX2.value) || 1,
        parseInt(elements.inputY2.value) || 1
      ];
      renderCanvas();
    });
  });

  // Run Segmentation
  elements.btnRunSegment.addEventListener('click', async () => {
    await executeSegmentation();
  });

  // Canvas Mouse Interactions for Bounding Box Drawing & Resizing
  setupCanvasInteractions();

  // Downloads
  setupDownloadButtons();
}

// Sync inputs with state
function updateInputsFromState() {
  elements.inputX1.value = state.bbox[0];
  elements.inputY1.value = state.bbox[1];
  elements.inputX2.value = state.bbox[2];
  elements.inputY2.value = state.bbox[3];
}

// Canvas Mouse Interactions
function setupCanvasInteractions() {
  const canvas = elements.mainCanvas;

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY)
    };
  }

  function getHandleUnderCursor(pt) {
    const [x1, y1, x2, y2] = state.bbox;
    const handleRadius = Math.max(8, canvas.width * 0.02);

    const dist = (px, py, hx, hy) => Math.hypot(px - hx, py - hy);

    if (dist(pt.x, pt.y, x1, y1) <= handleRadius) return 'tl';
    if (dist(pt.x, pt.y, x2, y1) <= handleRadius) return 'tr';
    if (dist(pt.x, pt.y, x1, y2) <= handleRadius) return 'bl';
    if (dist(pt.x, pt.y, x2, y2) <= handleRadius) return 'br';

    if (pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2) return 'move';
    return null;
  }

  canvas.addEventListener('mousedown', (e) => {
    const pt = getCanvasCoords(e);
    const handle = getHandleUnderCursor(pt);

    if (handle) {
      state.isDraggingHandle = handle;
      state.dragStart = pt;
      state.origBboxAtDrag = [...state.bbox];
    } else {
      state.isDrawing = true;
      state.dragStart = pt;
      state.bbox = [pt.x, pt.y, pt.x, pt.y];
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.currentImage) return;

    if (state.isDrawing) {
      const pt = getCanvasCoords(e);
      const x1 = Math.min(state.dragStart.x, pt.x);
      const y1 = Math.min(state.dragStart.y, pt.y);
      const x2 = Math.max(state.dragStart.x, pt.x);
      const y2 = Math.max(state.dragStart.y, pt.y);
      state.bbox = [
        Math.max(0, x1),
        Math.max(0, y1),
        Math.min(canvas.width, x2),
        Math.min(canvas.height, y2)
      ];
      updateInputsFromState();
      renderCanvas();
    } else if (state.isDraggingHandle) {
      const pt = getCanvasCoords(e);
      const dx = pt.x - state.dragStart.x;
      const dy = pt.y - state.dragStart.y;
      const [ox1, oy1, ox2, oy2] = state.origBboxAtDrag;

      if (state.isDraggingHandle === 'move') {
        const bw = ox2 - ox1;
        const bh = oy2 - oy1;
        let nx1 = Math.max(0, Math.min(canvas.width - bw, ox1 + dx));
        let ny1 = Math.max(0, Math.min(canvas.height - bh, oy1 + dy));
        state.bbox = [nx1, ny1, nx1 + bw, ny1 + bh];
      } else if (state.isDraggingHandle === 'tl') {
        state.bbox = [Math.min(ox2 - 10, ox1 + dx), Math.min(oy2 - 10, oy1 + dy), ox2, oy2];
      } else if (state.isDraggingHandle === 'tr') {
        state.bbox = [ox1, Math.min(oy2 - 10, oy1 + dy), Math.max(ox1 + 10, ox2 + dx), oy2];
      } else if (state.isDraggingHandle === 'bl') {
        state.bbox = [Math.min(ox2 - 10, ox1 + dx), oy1, ox2, Math.max(oy1 + 10, oy2 + dy)];
      } else if (state.isDraggingHandle === 'br') {
        state.bbox = [ox1, oy1, Math.max(ox1 + 10, ox2 + dx), Math.max(oy1 + 10, oy2 + dy)];
      }

      updateInputsFromState();
      renderCanvas();
    } else {
      // Hover cursor
      const pt = getCanvasCoords(e);
      const handle = getHandleUnderCursor(pt);
      if (handle === 'tl' || handle === 'br') canvas.style.cursor = 'nwse-resize';
      else if (handle === 'tr' || handle === 'bl') canvas.style.cursor = 'nesw-resize';
      else if (handle === 'move') canvas.style.cursor = 'move';
      else canvas.style.cursor = 'crosshair';
    }
  });

  window.addEventListener('mouseup', () => {
    state.isDrawing = false;
    state.isDraggingHandle = null;
    state.origBboxAtDrag = null;
  });
}

// Execute Segmentation API Call
async function executeSegmentation() {
  if (!state.currentFile) return;

  elements.segmentBtnText.textContent = 'Segmenting...';
  elements.segmentSpinner.style.display = 'inline-block';
  elements.btnRunSegment.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', state.currentFile);
    formData.append('x1', state.bbox[0]);
    formData.append('y1', state.bbox[1]);
    formData.append('x2', state.bbox[2]);
    formData.append('y2', state.bbox[3]);
    formData.append('model_name', state.selectedModel);

    const res = await fetch('/api/segment', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Segmentation failed.');
    const data = await res.json();
    state.results = data;

    // Update Telemetry Panel
    updateTelemetry(data.contrast_info);

    // Enable Download Buttons
    elements.btnDownloadMask.disabled = false;
    elements.btnDownloadOverlay.disabled = false;
    elements.btnDownloadTelemetry.disabled = false;

    // Render updated canvas
    renderCanvas();
  } catch (err) {
    console.error('Error during segmentation:', err);
    alert('Error running segmentation: ' + err.message);
  } finally {
    elements.segmentBtnText.textContent = 'Segment Lesion';
    elements.segmentSpinner.style.display = 'none';
    elements.btnRunSegment.disabled = false;
  }
}

// Update Telemetry Panel Numbers & Indicators
function updateTelemetry(info) {
  // Delta E
  elements.deltaENum.textContent = info.delta_e.toFixed(1);
  const fillPct = Math.min(100, Math.max(5, (info.delta_e / 60.0) * 100.0));
  elements.deltaEBar.style.width = `${fillPct}%`;

  if (info.delta_e < 18.0) {
    elements.deltaEFootnote.textContent = 'Acute boundary ambiguity: Pigment matches surrounding melanin closely.';
  } else if (info.delta_e < 35.0) {
    elements.deltaEFootnote.textContent = 'Moderate optical contrast: Typical for medium skin tones.';
  } else {
    elements.deltaEFootnote.textContent = 'High contrast boundary: Readily separable spectral profile.';
  }

  // Gamma Factor
  const gamma = info.gamma_factor || 1.0;
  elements.gammaNum.textContent = `${gamma.toFixed(2)}x`;
  if (gamma > 1.15) {
    elements.gammaBadge.textContent = 'Amplified (+ capacity)';
    elements.gammaBadge.style.background = 'rgba(245, 158, 11, 0.2)';
    elements.gammaBadge.style.color = '#FCD34D';
    elements.gammaExplanation.textContent = 'Dynamic gate scales up adapter representations to resolve subtle low-contrast edges.';
  } else if (gamma < 0.90) {
    elements.gammaBadge.textContent = 'Attenuated (stable)';
    elements.gammaBadge.style.background = 'rgba(56, 189, 248, 0.2)';
    elements.gammaBadge.style.color = '#38BDF8';
    elements.gammaExplanation.textContent = 'High contrast: Gate relies smoothly on pre-trained MedSAM foundation features.';
  } else {
    elements.gammaBadge.textContent = 'Nominal (1.0x)';
    elements.gammaBadge.style.background = 'rgba(16, 185, 129, 0.2)';
    elements.gammaBadge.style.color = '#6EE7B7';
    elements.gammaExplanation.textContent = 'Balanced adapter activation matching standard benchmark tuning.';
  }

  // Skin Cohort
  elements.toneName.textContent = info.estimated_fst;
  if (info.estimated_fst.includes('Dark')) {
    elements.toneCircle.style.background = '#78350F';
  } else if (info.estimated_fst.includes('Medium')) {
    elements.toneCircle.style.background = '#D97706';
  } else {
    elements.toneCircle.style.background = '#FDE68A';
  }

  elements.skinLuminanceVal.textContent = `${info.ring_luminance.toFixed(1)} L*`;
  elements.itaVal.textContent = `${info.ita_degrees.toFixed(1)}°`;

  // Morphometry
  elements.lesionAreaVal.textContent = `${info.lesion_area_px.toLocaleString()} px`;
  elements.lesionCoverageVal.textContent = `${info.lesion_coverage_pct}%`;
  elements.promptTypeVal.textContent = info.prompt_type || 'User Prompt';
}

// Master Render Function
function renderCanvas() {
  if (!state.currentImage) return;

  const canvas = elements.mainCanvas;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // 1. Render Base View depending on viewMode
  if (state.results && state.viewMode === 'mask') {
    // Binary mask only
    const maskImg = new Image();
    maskImg.onload = () => ctx.drawImage(maskImg, 0, 0, w, h);
    maskImg.src = state.results.mask_base64;
    return;
  } else if (state.results && state.viewMode === 'overlay') {
    // Composite overlay generated by backend
    const overImg = new Image();
    overImg.onload = () => {
      ctx.drawImage(overImg, 0, 0, w, h);
      drawPromptGeometry(w, h);
    };
    overImg.src = state.results.overlay_base64;
    return;
  } else if (state.results && state.viewMode === 'split') {
    // Left half original, right half overlay
    ctx.drawImage(state.currentImage, 0, 0, w, h);
    const splitImg = new Image();
    splitImg.onload = () => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(w / 2, 0, w / 2, h);
      ctx.clip();
      ctx.drawImage(splitImg, 0, 0, w, h);
      ctx.restore();

      // Draw divider line
      ctx.strokeStyle = '#00F0FF';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(w / 2, 0);
      ctx.lineTo(w / 2, h);
      ctx.stroke();

      drawPromptGeometry(w, h);
    };
    splitImg.src = state.results.overlay_base64;
    return;
  } else {
    // Original image
    ctx.drawImage(state.currentImage, 0, 0, w, h);
    drawPromptGeometry(w, h);
  }
}

// Draw Bounding Box, Eroded Core, Background Ring, and Drag Handles
function drawPromptGeometry(w, h) {
  const [x1, y1, x2, y2] = state.bbox;
  const bw = x2 - x1;
  const bh = y2 - y1;

  // 1. Eroded Core Box (C_core) if enabled
  if (state.showCore && bw > 10 && bh > 10) {
    const cx1 = x1 + Math.round(0.25 * bw);
    const cy1 = y1 + Math.round(0.25 * bh);
    const cx2 = x2 - Math.round(0.25 * bw);
    const cy2 = y2 - Math.round(0.25 * bh);

    ctx.save();
    ctx.strokeStyle = '#38BDF8';
    ctx.lineWidth = 1.5;
    ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
    ctx.fillRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
    ctx.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
    ctx.fillStyle = '#38BDF8';
    ctx.font = '10px Inter';
    ctx.fillText('C_core', cx1 + 4, cy1 + 12);
    ctx.restore();
  }

  // 2. Background Ring (R_ring) if enabled
  if (state.showRing && bw > 10 && bh > 10) {
    const rx1 = Math.max(0, x1 - Math.round(0.25 * bw));
    const ry1 = Math.max(0, y1 - Math.round(0.25 * bh));
    const rx2 = Math.min(w, x2 + Math.round(0.25 * bw));
    const ry2 = Math.min(h, y2 + Math.round(0.25 * bh));

    ctx.save();
    ctx.strokeStyle = '#A855F7';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
    ctx.fillStyle = '#A855F7';
    ctx.font = '10px Inter';
    ctx.fillText('R_ring', rx1 + 4, ry1 - 4 > 10 ? ry1 - 4 : ry1 + 12);
    ctx.restore();
  }

  // 3. Prompt Bounding Box (Amber Yellow)
  ctx.save();
  ctx.strokeStyle = '#F59E0B';
  ctx.lineWidth = 2.5;
  ctx.strokeRect(x1, y1, bw, bh);

  // Handles
  const handleRadius = Math.max(4, Math.min(8, w * 0.015));
  ctx.fillStyle = '#FBBF24';
  const handles = [
    [x1, y1], [x2, y1], [x1, y2], [x2, y2]
  ];
  handles.forEach(([hx, hy]) => {
    ctx.beginPath();
    ctx.arc(hx, hy, handleRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  ctx.restore();
}

// Download Buttons Handling
function setupDownloadButtons() {
  function downloadBlob(dataUrl, filename) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  elements.btnDownloadMask.addEventListener('click', () => {
    if (state.results && state.results.mask_base64) {
      downloadBlob(state.results.mask_base64, 'medsam_lesion_mask.png');
    }
  });

  elements.btnDownloadOverlay.addEventListener('click', () => {
    if (state.results && state.results.overlay_base64) {
      downloadBlob(state.results.overlay_base64, 'medsam_lesion_overlay.png');
    }
  });

  elements.btnDownloadTelemetry.addEventListener('click', () => {
    if (state.results && state.results.contrast_info) {
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.results.contrast_info, null, 2));
      downloadBlob(dataStr, 'medsam_telemetry_report.json');
    }
  });
}

// Start application on DOMContentLoaded
document.addEventListener('DOMContentLoaded', initApp);
