// ============================================================
// websocket.js — Real-time WebSocket Connection & Message Handler
// Contract: reads from window.ChatApp (set by renderers.js + chat.js).
// ============================================================

(function () {
  if (!window.PageConfig.isAuthenticated) return;

  const protocol  = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const sessionId = window.PageConfig.sessionId;
  if (!sessionId) return;

  const socketUrl = `${protocol}//${window.location.host}/ws/notifications/${sessionId}/`;

  let socket = null;
  const maxReconnectAttempts = 5;
  const baseReconnectDelayMs = 2000;
  const maxReconnectDelayMs = 30000;
  let reconnectAttempts = 0;

  function connect() {
    socket = new WebSocket(socketUrl);
    socket.onopen  = () => {
      reconnectAttempts = 0;
      console.log('Chat UI WebSocket connected');
    };
    socket.onerror = (err) => { console.error('Socket error:', err); socket.close(); };
    socket.onclose = (event) => {
      if (event.code === 4001 || event.code === 4003 || event.code === 4401 || event.code === 4403) return;
      if (reconnectAttempts >= maxReconnectAttempts) return;

      reconnectAttempts += 1;
      const delay = Math.min(baseReconnectDelayMs * (2 ** (reconnectAttempts - 1)), maxReconnectDelayMs);
      setTimeout(connect, delay);
    };

    socket.onmessage = function (e) {
      const data         = JSON.parse(e.data);
      const chatMessages = document.getElementById('chat-messages');
      const indicatorMsg = document.querySelector('.typing-indicator-msg');

      // Status update only (no event_type yet)
      if (data.type === 'notification' && !data.event_type) {
        if (indicatorMsg && data.message) {
          const statusEl = indicatorMsg.querySelector('.typing-status');
          if (statusEl) statusEl.innerText = data.message;
        }
        return;
      }

      // Final response received — remove typing indicator
      if (indicatorMsg) {
        indicatorMsg.remove();
        window.ChatApp.setSubmitting(false); // also calls updateSendBtn internally
      }

      if (data.type === 'chat_trace' || data.event_type === 'agent_message') {
        const msgFormat = (data.message_type === 'ai' || data.message_type === 'system') ? 'ai' : null;
        if (msgFormat === 'ai') {
          const logoUrl = window.PageConfig.siteLogoUrl;
          let msgId = (data.metadata && data.metadata.id) ? 'msg-' + data.metadata.id : 'msg-streaming-response';
          let msgEl = document.getElementById(msgId);

          if (!msgEl) {
            chatMessages.insertAdjacentHTML('beforeend', `
              <div class="message ai" id="${msgId}" data-rich-msg="true">
                <div class="avatar ai">
                  ${logoUrl
                    ? `<img src="${logoUrl}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;">`
                    : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                         <rect width="24" height="24" rx="6" fill="#8B5CF6"/>
                         <path d="M7 12L12 7L17 12L12 17L7 12Z" fill="white"/>
                       </svg>`}
                </div>
                <div class="message-content">
                  <div class="rich-blocks-top"></div>
                  <div class="text-content"></div>
                  <div class="rich-blocks-bottom"></div>
                </div>
              </div>
            `);
            msgEl = document.getElementById(msgId);
          }
          window.ChatApp.renderRichMessage(msgEl, data.content || '', data.metadata);
          window.ChatApp.scrollToBottom();
        }
      }
    };
  }

  // Re-render all historical rich messages on page load
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
      document.querySelectorAll('[data-rich-msg="true"]').forEach(msgEl => {
        const msgId          = msgEl.id.replace('msg-', '');
        const metadataScript = document.getElementById('meta-' + msgId);
        let metadata         = null;
        if (metadataScript) {
          try { metadata = JSON.parse(metadataScript.textContent); } catch (e) { /* ignore */ }
        }
        const contextEl = msgEl.querySelector('.text-content');
        if (contextEl && contextEl.innerText.trim()) {
          window.ChatApp.renderRichMessage(msgEl, contextEl.innerText.trim(), metadata);
        }
      });
      window.ChatApp.scrollToBottom();
    }, 200);
  });

  connect();
})();
