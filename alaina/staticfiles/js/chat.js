// ============================================================
// chat.js — Core Chat UI: Sidebar, Textarea, Submit, Namespace
// Renderer logic lives in renderers.js (loaded before this file).
// ============================================================

lucide.createIcons();

const userInitials = window.PageConfig.userInitials;
const siteLogoUrl  = window.PageConfig.siteLogoUrl;
const siteName     = window.PageConfig.siteName;

// --- 1. Global State ---
let isSubmitting = false;

// --- 2. Sidebar Toggle ---
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.toggle('open');

  const icon = document.querySelector('#mobile-toggle i') || document.querySelector('.toggle-btn i');
  if (icon && sidebar.classList.contains('open')) {
    icon.setAttribute('data-lucide', 'x');
  } else if (icon) {
    icon.setAttribute('data-lucide', 'panel-left-close');
  }
  lucide.createIcons();
}

// --- 3. Chat Textarea Auto-resize ---
const textarea = document.getElementById('chat-textarea');
textarea?.addEventListener('input', function () {
  this.style.height = 'auto';
  const newHeight   = this.scrollHeight;
  this.style.height = newHeight + 'px';
  this.style.overflowY = newHeight > 200 ? 'auto' : 'hidden';
  updateSendBtn();
});

function updateSendBtn() {
  const btn    = document.getElementById('submit-btn');
  btn.disabled = isSubmitting || !textarea.value.trim();
}

function escapedHTML(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function scrollToBottom() {
  const chatMessages = document.getElementById('chat-messages');
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- 4. Rich Message Renderer (delegated to renderers.js) ---
// renderRichMessage and renderArtifact live in window.ChatApp (set by renderers.js).

// --- 5. Chat Form Submit ---
async function handleChatSubmit(e) {
  if (e) e.preventDefault();
  if (isSubmitting) return;

  const form     = document.getElementById('chat-form');
  const formData = new FormData(form);
  const message  = formData.get('message');
  if (!message || !message.trim()) return;

  isSubmitting = true;
  updateSendBtn();

  textarea.value       = '';
  textarea.style.height = 'auto';
  requestAnimationFrame(() => {
    textarea.value = '';
    updateSendBtn();
  });

  const chatMessages = document.getElementById('chat-messages');

  chatMessages.insertAdjacentHTML('beforeend', `
    <div class="message user">
      <div class="avatar user">
        <span style="font-size: 11px; font-weight: 600; color: white;">${userInitials}</span>
      </div>
      <div class="message-content">${escapedHTML(message)}</div>
    </div>
  `);

  chatMessages.insertAdjacentHTML('beforeend', `
    <div class="message ai typing-indicator-msg">
      <div class="avatar ai">
        ${siteLogoUrl
          ? `<img src="${siteLogoUrl}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">`
          : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
               <rect width="24" height="24" rx="6" fill="#8B5CF6"/>
               <path d="M7 12L12 7L17 12L12 17L7 12Z" fill="white"/>
             </svg>`}
      </div>
      <div class="message-content">
        <div class="typing-indicator" style="gap: 12px; display: flex; align-items: center;">
          <div style="display: flex; gap: 4px;">
            <div class="dot" style="background-color: var(--primary-color);"></div>
            <div class="dot" style="background-color: var(--primary-color); opacity: 0.7;"></div>
            <div class="dot" style="background-color: var(--primary-color); opacity: 0.4;"></div>
          </div>
          <span class="typing-status" style="font-size: 11px; color: #a1a1aa; font-weight: 500; font-family: var(--font-secondary);">AI is thinking...</span>
        </div>
      </div>
    </div>
  `);
  scrollToBottom();

  try {
    const response = await fetch('/api/chat-analysis/', {
      method: 'POST',
      body: JSON.stringify({
        message,
        session_id: formData.get('session_id'),
        dataset_id: formData.get('dataset_id')
      }),
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken':  formData.get('csrfmiddlewaretoken')
      }
    });
    await response.json();
    // Socket/polling handles the actual response rendering
  } catch (err) {
    console.error(err);
  } finally {
    isSubmitting = false;
    updateSendBtn();
  }
}

document.getElementById('chat-form')?.addEventListener('submit', handleChatSubmit);
textarea?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleChatSubmit();
  }
});

// ============================================================
// ChatApp Namespace — expose shared state to sibling modules
// ============================================================
window.ChatApp = window.ChatApp || {};

/** Called by websocket.js to flip isSubmitting and sync the button */
window.ChatApp.setSubmitting = function (val) {
  isSubmitting = val;
  updateSendBtn();
};

window.ChatApp.scrollToBottom = scrollToBottom;

