// ============================================================
// renderers.js — Artifact Rendering System
// Provides type-specific renderers with error boundaries.
// Exposes window.ChatApp.renderArtifact & renderRichMessage.
// Must be loaded before chat.js and websocket.js.
// ============================================================

// Initialize the shared ChatApp namespace
window.ChatApp = window.ChatApp || {};

// ============================================================
// Private: Per-type Renderers
// ============================================================

async function _renderChart(blockContainer, art, index) {
  const stableId = art.result_id
    ? `chart-${art.result_id}`
    : `chart-temp-${Math.random().toString(36).substr(2, 5)}`;
  const chartId = `${stableId}-${index}`;

  blockContainer.insertAdjacentHTML('beforeend', `
    <div class="rich-block chart-block" style="background: rgba(24, 24, 27, 0.4); backdrop-filter: blur(10px); color: white;">
      <div class="block-header" style="border-bottom: 1px solid rgba(255,255,255,0.05);">
        <div class="block-title-group">
          <i data-lucide="line-chart" class="block-icon" size="14" style="color: #8B5CF6;"></i>
          <span class="block-title" style="font-weight: 500; font-size: 13px;">${art.title || 'Data Analysis'}</span>
        </div>
      </div>
      <div id="${chartId}" class="chart-container" style="height: 320px; width: 100%; padding: 16px;">
        <div class="chart-loader" style="height: 100%; display: flex; align-items: center; justify-content: center; color: #71717a; font-size: 12px;">
          Initializing Chart Engine...
        </div>
      </div>
    </div>
  `);

  // Lazy-load chart data if not inline
  let chartData = art.metadata;
  if (!chartData && art.result_id) {
    try {
      const resp = await fetch(`/api/analysis-result/${art.result_id}/`);
      const json = await resp.json();
      chartData  = json.data;
    } catch (e) {
      console.error('[Renderer:chart] Failed to fetch chart data:', e);
    }
  }

  if (chartData && (chartData.type || art.option)) {
    const el = document.getElementById(chartId);
    if (el) {
      ChatChartManager.initChart(
        chartId, el,
        art.option ? 'RAW' : chartData.type,
        art.option || chartData
      );
    }
  }
}

function _renderPlot(blockContainer, art) {
  // Defensive extraction: new string → legacy object → legacy metadata
  let base64 = null;
  if (typeof art.content === 'string' && art.content.length > 100) {
    base64 = art.content;
  } else if (art.metadata && typeof art.metadata.base64 === 'string') {
    base64 = art.metadata.base64;
  } else if (art.content && typeof art.content === 'object' && typeof art.content.base64 === 'string') {
    base64 = art.content.base64;
  }

  if (!base64) {
    console.warn('[Renderer:plot] No base64 data found in artifact:', art);
    return;
  }

  const imgSrc = base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
  blockContainer.insertAdjacentHTML('beforeend', `
    <div class="rich-block plot-block" style="background: #18181b; border: 1px solid #27272a; border-radius: 12px; overflow: hidden; margin-top: 12px;">
      <div class="block-header" style="background: rgba(255,255,255,0.05); padding: 10px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);">
        <div class="block-title-group">
          <i data-lucide="image" class="block-icon" size="14" style="color: #8B5CF6;"></i>
          <span class="block-title" style="font-weight: 500; font-size: 13px;">${art.title || 'Static Visualization'}</span>
        </div>
      </div>
      <div class="photo-frame" style="padding: 20px; background: white; display: flex; justify-content: center;">
        <img src="${imgSrc}"
             style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);">
      </div>
    </div>
  `);
}

function _renderTable(blockContainer, art) {
  const tableData = art.data || art.rows || [];
  const headers   = art.headers || [];

  // Pre-compute separator strings to avoid nested template-literal conflicts
  const thStyle = 'padding: 10px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); color: #a1a1aa; font-weight: 600; white-space: nowrap;';
  const tdStyle = 'padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.02); color: #e4e4e7; white-space: nowrap;';
  const thSep   = `</th><th style="${thStyle}">`;
  const tdSep   = `</td><td style="${tdStyle}">`;

  const thead = headers.length
    ? `<thead><tr style="background: rgba(255,255,255,0.02);">
         <th style="${thStyle}">${headers.map(h => String(h ?? '')).join(thSep)}</th>
       </tr></thead>`
    : '';

  const tbody = tableData.map(row =>
    `<tr><td style="${tdStyle}">${row.map(cell => String(cell ?? '')).join(tdSep)}</td></tr>`
  ).join('');

  const footer = art.footer
    ? `<div style="padding: 8px 12px; font-size: 11px; color: #71717a; border-top: 1px solid rgba(255,255,255,0.02); font-style: italic;">${art.footer}</div>`
    : '';

  blockContainer.insertAdjacentHTML('beforeend', `
    <div class="rich-block table-block" style="margin-top: 12px; border: 1px solid rgba(82,82,91,0.3); border-radius: 12px; overflow: hidden;">
      <div class="block-header" style="background: rgba(255,255,255,0.03); padding: 10px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; gap: 8px;">
        <i data-lucide="table" class="block-icon" size="14" style="color: #8B5CF6;"></i>
        <span class="block-title" style="font-weight: 500; font-size: 13px;">${art.title || art.label || 'Data Insight'}</span>
      </div>
      <div class="table-container" style="overflow-x: auto; font-size: 12px; background: rgba(9,9,11,0.2);">
        <table style="width: 100%; border-collapse: collapse;">
          ${thead}
          <tbody>${tbody}</tbody>
        </table>
        ${footer}
      </div>
    </div>
  `);
}

// ============================================================
// Private: Error Boundary Fallback
// ============================================================

function _renderFallback(blockContainer, art, error) {
  console.error(
    `[Renderer] Failed — type="${art.type || 'unknown'}", title="${art.title || ''}"`,
    error
  );
  blockContainer.insertAdjacentHTML('beforeend', `
    <div style="padding: 10px 14px; background: rgba(248,113,113,0.05); border: 1px solid rgba(248,113,113,0.2); border-radius: 8px; margin-top: 8px; display: flex; align-items: center; gap: 8px;">
      <i data-lucide="alert-triangle" size="13" style="color: #f87171; flex-shrink: 0;"></i>
      <p style="font-size: 12px; color: #f87171; margin: 0;">
        Could not render <b>${art.type || 'artifact'}</b>${art.title ? ': ' + art.title : ''}.
        The data may be malformed.
      </p>
    </div>
  `);
}

// ============================================================
// Public: renderArtifact — dispatcher with error boundary
// ============================================================

async function renderArtifact(blockContainer, art, index) {
  if (!blockContainer || !art) return;
  try {
    if      (art.type === 'chart') await _renderChart(blockContainer, art, index);
    else if (art.type === 'plot')       _renderPlot(blockContainer, art);
    else if (art.type === 'table')      _renderTable(blockContainer, art);
    else    console.warn(`[Renderer] Unknown artifact type: "${art.type}"`, art);
  } catch (err) {
    _renderFallback(blockContainer, art, err);
  }
}

// ============================================================
// Public: renderRichMessage — thin orchestrator
// ============================================================

async function renderRichMessage(container, content, metadata = null) {
  if (!container) return;
  console.log(`[RichMessage] Rendering ID: ${metadata ? metadata.id : 'N/A'}`, metadata);

  // 1. Render markdown text
  const textEl = container.querySelector('.text-content');
  if (textEl && content) textEl.innerHTML = marked.parse(content);

  // 2. Render artifacts (each isolated behind its own error boundary)
  if (metadata && metadata.artifacts) {
    console.log(`[RichMessage] ${metadata.artifacts.length} artifact(s)`, metadata.artifacts);
    const blocksTop    = container.querySelector('.rich-blocks-top');
    const blocksBottom = container.querySelector('.rich-blocks-bottom') || container.querySelector('.rich-blocks-container');

    if (blocksTop)    blocksTop.innerHTML    = '';
    if (blocksBottom) blocksBottom.innerHTML = '';

    for (const [index, art] of metadata.artifacts.entries()) {
      // Tables above text (narrative order), visuals below
      const target = (art.type === 'table' && blocksTop) ? blocksTop : blocksBottom;
      if (!target) continue;
      await renderArtifact(target, art, index);
    }
  }

  lucide.createIcons();
}

// ============================================================
// Export to window.ChatApp
// ============================================================

window.ChatApp.renderArtifact   = renderArtifact;
window.ChatApp.renderRichMessage = renderRichMessage;
