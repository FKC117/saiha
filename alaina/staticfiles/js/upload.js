// ============================================================
// upload.js — Dataset Upload & Empty Column Warning Logic
// ============================================================

let pendingUploadData = null;

function closeEmptyColumnModal() {
  document.getElementById('emptyColumnWarningModal').classList.remove('open');
  pendingUploadData = null;
}

async function confirmDropEmpty() {
  if (!pendingUploadData) return;

  const { form, endpoint } = pendingUploadData;
  const formData = new FormData(form);
  formData.append('drop_empty', 'true');

  document.getElementById('emptyColumnWarningModal').classList.remove('open');
  const btn = document.getElementById('upload-submit-btn');
  if (btn) btn.innerText = 'Cleaning & Uploading...';

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
      headers: { 'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value }
    });
    const data = await response.json();

    if (data.dataset_id || data.id) {
      window.location.href = '?dataset_id=' + (data.dataset_id || data.id);
    } else {
      alert('Error: ' + data.message);
      if (btn) btn.innerText = 'Upload Dataset';
      if (btn) btn.disabled = false;
    }
  } catch (err) {
    console.error(err);
    alert('Retry failed.');
    if (btn) btn.innerText = 'Upload Dataset';
    if (btn) btn.disabled = false;
  }
}

document.getElementById('file-input')?.addEventListener('change', function () {
  if (this.files?.length > 0) {
    const uploadZoneP = this.closest('.upload-zone')?.querySelector('p');
    if (uploadZoneP) uploadZoneP.innerText = 'Processing: ' + this.files[0].name;

    const form = this.closest('form') || document.getElementById('uploadDatasetForm');
    if (form) {
      if (typeof form.requestSubmit === 'function') {
        form.requestSubmit();
      } else {
        form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    }
  }
});

document.getElementById('uploadDatasetForm')?.addEventListener('submit', async function (e) {
  e.preventDefault();
  const btn = document.getElementById('upload-submit-btn');
  if (btn) {
    btn.innerText = 'Processing...';
    btn.disabled = true;
  }

  try {
    const response = await fetch(this.action, {
      method: 'POST',
      body: new FormData(this),
      headers: { 'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value }
    });
    const data = await response.json();

    if (data.status === 'warning' && data.empty_columns) {
      pendingUploadData = { form: this, endpoint: this.action };
      const listEl = document.getElementById('empty-column-list');
      listEl.innerHTML = data.empty_columns.map(col => `
        <span style="background: rgba(139, 92, 246, 0.1); color: #c4b5fd; padding: 4px 10px; border-radius: 6px; font-size: 12px; border: 1px solid rgba(139, 92, 246, 0.2);">
          ${col}
        </span>
      `).join('');
      document.getElementById('emptyColumnWarningModal').classList.add('open');
      return;
    }

    if (data.dataset_id || data.id) {
      window.location.href = '?dataset_id=' + (data.dataset_id || data.id);
    } else if (data.status === 'error') {
      alert('Error: ' + data.message);
    }
  } catch (err) {
    console.error(err);
    alert('Upload failed. Please check your network or file format.');
  } finally {
    if (!document.getElementById('emptyColumnWarningModal').classList.contains('open')) {
      if (btn) {
        btn.innerText = 'Upload Dataset';
        btn.disabled = false;
      }
    }
  }
});
