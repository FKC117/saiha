// ============================================================
// modals.js — Modal Open/Close/Tab & Settings/Retail Logic
// ============================================================

// --- Dataset Modal ---
function openDatasetModal()  { document.getElementById('datasetModal').classList.add('open'); }
function closeDatasetModal() { document.getElementById('datasetModal').classList.remove('open'); }

function switchModalTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.modal-tab').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  document.getElementById(tabId === 'upload-tab' ? 'tab-btn-upload' : 'tab-btn-select').classList.add('active');
}

// --- Settings Modal ---
function openSettingsModal()  { document.getElementById('settingsModal').classList.add('open'); }
function closeSettingsModal() { document.getElementById('settingsModal').classList.remove('open'); }

function switchSettingsTab(tabId) {
  const modal = document.getElementById('settingsModal');
  modal.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  modal.querySelectorAll('.modal-tab').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  const btnId = 'tab-btn-' + tabId.replace('-tab', '');
  const btn   = document.getElementById(btnId);
  if (btn) btn.classList.add('active');
}

async function handleProfileSubmit(event) {
  event.preventDefault();
  const form     = event.target;
  const statusEl = document.getElementById('profile-status');
  const btn      = document.getElementById('save-profile-btn');

  btn.disabled  = true;
  btn.innerText = 'Saving...';
  statusEl.style.display = 'none';

  try {
    const response = await fetch(form.getAttribute('hx-post'), {
      method: 'POST',
      body: new FormData(form),
      headers: { 'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value }
    });
    const data = await response.json();

    if (data.status === 'success') {
      statusEl.innerText     = data.message;
      statusEl.style.color   = '#34d399';
      statusEl.style.display = 'block';
      document.getElementById('sidebar-user-name').innerText     = data.display_name;
      document.getElementById('sidebar-user-initials').innerText = data.initials;
      setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
    } else {
      statusEl.innerText     = data.message || 'Error updating profile';
      statusEl.style.color   = '#f87171';
      statusEl.style.display = 'block';
    }
  } catch (err) {
    console.error(err);
    statusEl.innerText     = 'Server error. Please try again.';
    statusEl.style.color   = '#f87171';
    statusEl.style.display = 'block';
  } finally {
    btn.disabled  = false;
    btn.innerText = 'Save Profile';
  }
}

// --- Generic Modal Close ---
function closeModal(id) { document.getElementById(id).style.display = 'none'; }

// --- Retail Recharge Modal ---
function openRetailRechargeModal() {
  closeUsageModal();
  document.getElementById('retailRechargeModal').style.display = 'flex';
}

let activeRetailPackage = { id: '', amount: 0, cost: 0 };

function handleRetailPackageClick(el) {
  activeRetailPackage = {
    id:     el.dataset.id,
    amount: el.dataset.credits,
    cost:   parseFloat(el.dataset.priceUsd),
    tier:   el.dataset.name,
    bdt:    el.dataset.priceBdt
  };
  document.getElementById('retail-checkout-tier').innerText   = activeRetailPackage.tier;
  document.getElementById('retail-checkout-amount').innerText = activeRetailPackage.amount;
  document.getElementById('retail-checkout-cost').innerText   = '$' + activeRetailPackage.cost.toFixed(2);
  document.getElementById('retail-checkout-bdt').innerText    = activeRetailPackage.bdt;

  document.getElementById('retailRechargeModal').style.display = 'none';
  document.getElementById('retailCheckoutModal').style.display = 'flex';
}

function completeRetailPurchase() {
  const formData = new FormData();
  formData.append('package_id',          activeRetailPackage.id);
  formData.append('csrfmiddlewaretoken', window.PageConfig.csrfToken);

  fetch('/user/topup/', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(d => {
      if (d.status === 'success') {
        if (d.invoice_id && confirm(d.message + '\n\nWould you like to view the invoice now?')) {
          window.open(`/billing/invoice/${d.invoice_id}/`, '_blank');
        }
        location.reload();
      } else {
        alert(d.message);
      }
    });
}

// --- Backdrop Click to Close ---
window.onclick = function (event) {
  if (event.target.classList.contains('dataset-modal-backdrop')) {
    event.target.classList.remove('open');
  }
  if (event.target.id === 'retailRechargeModal' || event.target.id === 'retailCheckoutModal') {
    event.target.style.display = 'none';
  }
};
