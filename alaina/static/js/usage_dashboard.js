// ============================================================
// usage_dashboard.js — Usage Modal, Charts & Billing History
// ============================================================

let trendChart    = null;
let sessionsChart = null;

function openUsageModal() {
  document.getElementById('usage-modal-backdrop').classList.add('open');
  lucide.createIcons();
  fetchUsageData();
}

function closeUsageModal() {
  document.getElementById('usage-modal-backdrop').style.display = 'none';
  switchUsageTab('overview'); // Reset for next open
}

function switchUsageTab(tab) {
  const overview    = document.getElementById('usage-overview-pane');
  const history     = document.getElementById('usage-history-pane');
  const overviewBtn = document.getElementById('usage-tab-overview');
  const historyBtn  = document.getElementById('usage-tab-history');

  if (tab === 'overview') {
    overview.style.display = 'block';
    history.style.display  = 'none';
    overviewBtn.classList.add('active');
    historyBtn.classList.remove('active');
  } else {
    overview.style.display = 'none';
    history.style.display  = 'block';
    overviewBtn.classList.remove('active');
    historyBtn.classList.add('active');
    fetchRetailBillingHistory();
  }
}

function fetchRetailBillingHistory() {
  const container = document.getElementById('retail-invoice-list');
  fetch('/api/billing/history/')
    .then(r => r.json())
    .then(d => {
      if (d.invoices.length === 0) {
        container.innerHTML = `<p style="text-align:center; padding:40px; color:var(--zinc-500); border:1px dashed var(--zinc-800); border-radius:8px; font-size:13px;">${window.PageConfig.emptyMsg}</p>`;
        return;
      }

      let html = `<div style="display:flex; flex-direction:column; gap:12px;">`;
      d.invoices.forEach(inv => {
        html += `
          <div style="background:rgba(39,39,42,0.4); border:1px solid var(--zinc-800); border-radius:12px; padding:16px; display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div style="font-size:12px; color:var(--zinc-500); font-weight:700; margin-bottom:4px;">${inv.number}</div>
              <div style="font-size:14px; font-weight:500; color:white;">${inv.description}</div>
              <div style="font-size:12px; color:var(--zinc-500); margin-top:4px;">${inv.date}</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:16px; font-weight:700; color:white; margin-bottom:8px;">${inv.amount}</div>
              <a href="${inv.url}" target="_blank" style="padding:4px 10px; font-size:11px; text-decoration:none; display:inline-flex; align-items:center; gap:6px; color:var(--zinc-400); border:1px solid var(--zinc-800); border-radius:6px; background:transparent;">
                <i data-lucide="external-link" size="12"></i> View
              </a>
              <a href="${inv.url}" target="_blank" style="padding:4px 10px; font-size:11px; text-decoration:none; display:inline-flex; align-items:center; gap:6px; color:var(--zinc-400); border:1px solid var(--zinc-800); border-radius:6px; background:transparent;" title="View & Download PDF">
                <i data-lucide="download" size="12"></i>
              </a>
            </div>
          </div>
        `;
      });
      html += `</div>`;
      container.innerHTML = html;
      lucide.createIcons();
    });
}

function resendInvoiceEmail(invoiceId) {
  const formData = new FormData();
  formData.append('invoice_id',          invoiceId);
  formData.append('csrfmiddlewaretoken', window.PageConfig.csrfToken);
  fetch('/api/billing/resend/', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(d => alert(d.message));
}

function requestCorporateCredits() {
  const amount = prompt('Enter the number of credits you wish to request from your administrator:', '10');
  if (amount && !isNaN(amount) && parseFloat(amount) > 0) {
    const formData = new FormData();
    formData.append('amount',  amount);
    formData.append('message', 'Request from dashboard');
    fetch('/api/corporate/request-credits/', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(d => { alert(d.message); });
  } else if (amount !== null) {
    alert('Please enter a valid amount.');
  }
}

async function fetchUsageData() {
  try {
    const response = await fetch(window.PageConfig.usageDataUrl);
    const data     = await response.json();

    // 1. Plan name
    document.getElementById('usage-plan-name').innerText = data.plan_name + ' Plan';

    // 2. Credits (10,000 tokens = 1 Credit)
    const rate        = 10000;
    const creditsUsed = (data.used_tokens / rate).toFixed(2);
    const creditsMax  = (data.max_tokens  / rate).toFixed(1);
    document.getElementById('usage-credits-used').innerText = creditsUsed;
    document.getElementById('usage-credits-max').innerText  = creditsMax;

    // 3. Progress bar
    const percent = data.max_tokens > 0 ? (data.used_tokens / data.max_tokens) * 100 : 0;
    document.getElementById('usage-quota-bar').style.width = percent + '%';

    // 4. KPI cards
    document.getElementById('usage-kpi-today').innerText = (data.used_tokens / rate).toFixed(3);
    document.getElementById('usage-kpi-total').innerText = creditsUsed;

    // 5. Rescue pool notice
    const rescuePool = data.rescue_tokens || 0;
    const rescueEl   = document.getElementById('usage-rescue-notice');
    if (rescueEl) {
      if (rescuePool > 0) {
        rescueEl.style.display = 'block';
        document.getElementById('usage-rescue-amount').innerText = rescuePool.toLocaleString();
      } else {
        rescueEl.style.display = 'none';
      }
    }

    // 6. Expiry notice
    const expiryEl = document.getElementById('usage-expiry-notice');
    if (expiryEl) {
      if (data.expiry_date && !data.is_expired) {
        expiryEl.style.display = 'block';
        document.getElementById('usage-expiry-date').innerText = new Date(data.expiry_date).toLocaleDateString();
      } else {
        expiryEl.style.display = 'none';
      }
    }

    // 7. Charts
    const reportResp = await fetch(`/usage/stats/?t=${new Date().getTime()}`);
    const reportData = await reportResp.json();
    if (reportData.status === 'success') {
      loadUsageCharts(reportData.charts);
    }
  } catch (err) {
    console.error('Failed to fetch usage stats:', err);
  }
}

function loadUsageCharts(charts) {
  // Trend Chart
  if (!trendChart) trendChart = echarts.init(document.getElementById('usage-trend-chart'));
  trendChart.setOption({
    grid:   { top: 10, left: 40, right: 10, bottom: 30 },
    xAxis:  { type: 'category', data: charts.daily_dates, axisLine: { lineStyle: { color: '#3f3f46' } } },
    yAxis:  { type: 'value', splitLine: { lineStyle: { color: '#27272a' } } },
    series: [{
      data:       charts.daily_values,
      type:       'line',
      smooth:     true,
      areaStyle:  { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(139, 92, 246, 0.3)' }, { offset: 1, color: 'transparent' }]) },
      lineStyle:  { color: '#8B5CF6', width: 3 },
      showSymbol: false,
    }]
  });

  // Sessions Chart
  if (!sessionsChart) sessionsChart = echarts.init(document.getElementById('usage-sessions-chart'));
  sessionsChart.setOption({
    grid:   { top: 10, left: 100, right: 30, bottom: 20 },
    xAxis:  { type: 'value', show: false },
    yAxis:  {
      type:      'category',
      data:      charts.sessions.map(s => s.name),
      axisLine:  { show: false },
      axisTick:  { show: false },
      axisLabel: { color: '#a1a1aa', fontSize: 11 }
    },
    series: [{
      type:      'bar',
      data:      charts.sessions.map(s => s.value),
      itemStyle: { borderRadius: [0, 4, 4, 0], color: '#6366F1' },
      barPadding: '20%'
    }]
  });
}
