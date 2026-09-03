(function () {
  "use strict";

  if (window.__AXERA_DOCS_ASSISTANT__) {
    return;
  }
  window.__AXERA_DOCS_ASSISTANT__ = true;

  const script = document.currentScript;
  const apiUrl = script?.dataset.apiUrl || "/api/docs-assistant";
  const language = script?.dataset.language || document.documentElement.lang || "zh_CN";
  const isEnglish = language.toLowerCase().startsWith("en");

  function randomId() {
    if (window.crypto?.randomUUID) {
      return window.crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
      const random = Math.random() * 16;
      const value = char === "x" ? random : (random & 0x3) | 0x8;
      return Math.floor(value).toString(16);
    });
  }

  function storageValue(storage, key) {
    try {
      return storage.getItem(key);
    } catch (error) {
      console.warn("Docs assistant browser storage is unavailable.", error);
      return null;
    }
  }

  function setStorageValue(storage, key, value) {
    try {
      storage.setItem(key, value);
    } catch (error) {
      console.warn("Docs assistant browser storage is unavailable.", error);
    }
  }

  function removeStorageValue(storage, key) {
    try {
      storage.removeItem(key);
    } catch (error) {
      console.warn("Docs assistant browser storage is unavailable.", error);
    }
  }

  const storageNamespace = (() => {
    try {
      const parsed = new URL(apiUrl, window.location.href);
      return `${parsed.origin}${parsed.pathname}`;
    } catch (error) {
      return apiUrl;
    }
  })();
  const clientStorageKey = `axera-docs-assistant-client:${storageNamespace}`;
  const conversationStorageKey = `axera-docs-assistant-conversation:${storageNamespace}`;
  const clientId = storageValue(window.localStorage, clientStorageKey) || randomId();
  let conversationId = storageValue(window.sessionStorage, conversationStorageKey);
  // The previous implementation kept this key in localStorage. Drop that stale
  // browser-only pointer when switching to tab-scoped conversations.
  removeStorageValue(window.localStorage, conversationStorageKey);
  setStorageValue(window.localStorage, clientStorageKey, clientId);
  if (conversationId) {
    setStorageValue(window.sessionStorage, conversationStorageKey, conversationId);
  }

  const copy = isEnglish
    ? {
        assistant: "Docs assistant",
        welcome: "Hello. What can I help you find?",
        placeholder: "Ask about AXERA documentation...",
        newSession: "New chat",
        send: "Send",
        close: "Close",
        sources: "Sources",
        retrieval: "Retrieval preview",
        llm: "Grounded answer",
        waiting: "Searching documentation...",
        failed: "The assistant is unavailable. Please try again later.",
      }
    : {
        assistant: "智能助手",
        welcome: "你好，有什么可以帮你？",
        placeholder: "询问 AXERA 文档...",
        newSession: "新会话",
        send: "发送",
        close: "关闭",
        sources: "参考来源",
        retrieval: "检索预览",
        llm: "文档回答",
        waiting: "正在检索文档...",
        failed: "助手暂时不可用，请稍后重试。",
      };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function icon(name) {
    const node = element("i", `fa-solid fa-${name}`);
    node.setAttribute("aria-hidden", "true");
    return node;
  }

  function initialize() {
    if (document.getElementById("docs-ai-launcher")) {
      return;
    }

    const launcher = element("button", "docs-ai-launcher");
    launcher.id = "docs-ai-launcher";
    launcher.type = "button";
    launcher.title = copy.assistant;
    launcher.setAttribute("aria-label", copy.assistant);
    launcher.setAttribute("aria-expanded", "false");
    launcher.append(icon("comments"), element("span", "docs-ai-launcher__text", copy.assistant));

    const overlay = element("div", "docs-ai-overlay");
    overlay.id = "docs-ai-overlay";

    const drawer = element("section", "docs-ai-drawer");
    drawer.id = "docs-ai-drawer";
    drawer.setAttribute("role", "dialog");
    drawer.setAttribute("aria-modal", "true");
    drawer.setAttribute("aria-labelledby", "docs-ai-title");
    drawer.setAttribute("aria-hidden", "true");

    const header = element("header", "docs-ai-header");
    const heading = element("div", "docs-ai-heading");
    heading.append(icon("comments"), element("h2", "docs-ai-title", copy.assistant));
    heading.querySelector("h2").id = "docs-ai-title";
    const mode = element("span", "docs-ai-mode", copy.retrieval);
    const newSession = element("button", "docs-ai-icon-button docs-ai-new-session");
    newSession.type = "button";
    newSession.title = copy.newSession;
    newSession.setAttribute("aria-label", copy.newSession);
    newSession.append(icon("plus"), element("span", "docs-ai-new-session__label", copy.newSession));
    const close = element("button", "docs-ai-icon-button");
    close.type = "button";
    close.title = copy.close;
    close.setAttribute("aria-label", copy.close);
    close.append(icon("xmark"));
    header.append(heading, mode, newSession, close);

    const log = element("div", "docs-ai-log");
    log.setAttribute("role", "log");
    log.setAttribute("aria-live", "polite");

    const composer = element("form", "docs-ai-composer");
    const input = element("textarea", "docs-ai-input");
    input.rows = 2;
    input.maxLength = 4000;
    input.placeholder = copy.placeholder;
    input.setAttribute("aria-label", copy.placeholder);
    const send = element("button", "docs-ai-send");
    send.type = "submit";
    send.title = copy.send;
    send.setAttribute("aria-label", copy.send);
    send.append(icon("paper-plane"));
    composer.append(input, send);

    drawer.append(header, log, composer);
    overlay.append(drawer);
    document.body.append(launcher, overlay);

    let lastFocused = null;
    let pending = false;
    let sessionGeneration = 0;

    function scrollToLatest() {
      log.scrollTop = log.scrollHeight;
    }

    function appendSources(container, sources) {
      if (!Array.isArray(sources) || sources.length === 0) {
        return;
      }
      const section = element("section", "docs-ai-sources");
      section.append(element("div", "docs-ai-sources__title", copy.sources));
      const list = element("ol", "docs-ai-sources__list");
      sources.forEach((source) => {
        const item = element("li", "docs-ai-source");
        if (source.url) {
          const link = element("a", "docs-ai-source__link", source.title || source.document || source.url);
          link.href = source.url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          item.append(link);
        } else {
          item.append(element("span", "docs-ai-source__label", source.title || source.document || copy.sources));
        }
        if (typeof source.score === "number") {
          item.append(element("span", "docs-ai-source__score", source.score.toFixed(3)));
        }
        list.append(item);
      });
      section.append(list);
      container.append(section);
    }

    function appendMessage(role, text, sources, extraClass) {
      const row = element("div", `docs-ai-message docs-ai-message--${role}${extraClass ? ` ${extraClass}` : ""}`);
      const bubble = element("div", "docs-ai-message__content", text);
      row.append(bubble);
      appendSources(row, sources);
      log.append(row);
      scrollToLatest();
      return row;
    }

    appendMessage("assistant", copy.welcome);

    function conversationUrl() {
      if (!conversationId) {
        return null;
      }
      const parsed = new URL(apiUrl, window.location.href);
      parsed.pathname = `${parsed.pathname.replace(/\/$/, "")}/conversations/${encodeURIComponent(conversationId)}`;
      parsed.search = "";
      return parsed.toString();
    }

    function persistConversationId(value) {
      if (value && value !== conversationId) {
        conversationId = value;
        setStorageValue(window.sessionStorage, conversationStorageKey, conversationId);
      }
    }

    async function restoreConversation() {
      const restoreGeneration = sessionGeneration;
      const url = conversationUrl();
      if (!url) {
        return;
      }
      try {
        const response = await fetch(url, {
          headers: { "X-Docs-Assistant-Client": clientId },
        });
        if (response.status === 404) {
          return;
        }
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
        }
        if (
          restoreGeneration === sessionGeneration &&
          Array.isArray(payload.messages) &&
          payload.messages.length > 0
        ) {
          log.replaceChildren();
          payload.messages.forEach((message) => {
            if (message.role === "user" || message.role === "assistant") {
              appendMessage(message.role, String(message.content || ""), message.sources);
            }
          });
        }
      } catch (error) {
        console.warn("Docs assistant conversation restore failed.", error);
      }
    }

    restoreConversation();

    function startNewSession() {
      if (pending) {
        return;
      }
      sessionGeneration += 1;
      conversationId = null;
      removeStorageValue(window.sessionStorage, conversationStorageKey);
      log.replaceChildren();
      mode.textContent = copy.retrieval;
      appendMessage("assistant", copy.welcome);
      input.focus();
    }

    function openDrawer() {
      lastFocused = document.activeElement;
      overlay.classList.add("is-open");
      drawer.classList.add("is-open");
      drawer.setAttribute("aria-hidden", "false");
      launcher.setAttribute("aria-expanded", "true");
      document.body.classList.add("docs-ai-open");
      window.setTimeout(() => input.focus(), 180);
    }

    function closeDrawer() {
      overlay.classList.remove("is-open");
      drawer.classList.remove("is-open");
      drawer.setAttribute("aria-hidden", "true");
      launcher.setAttribute("aria-expanded", "false");
      document.body.classList.remove("docs-ai-open");
      if (lastFocused instanceof HTMLElement) {
        lastFocused.focus();
      }
    }

    launcher.addEventListener("click", openDrawer);
    newSession.addEventListener("click", startNewSession);
    close.addEventListener("click", closeDrawer);
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) {
        closeDrawer();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && overlay.classList.contains("is-open")) {
        closeDrawer();
      }
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        composer.requestSubmit();
      }
    });

    composer.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message || pending) {
        return;
      }

      appendMessage("user", message);
      input.value = "";
      input.disabled = true;
      send.disabled = true;
      newSession.disabled = true;
      pending = true;
      const loading = appendMessage("assistant", copy.waiting, [], "is-loading");

      try {
        const response = await fetch(apiUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            conversation_id: conversationId,
            client_id: clientId,
            page: {
              title: document.querySelector("main h1, article h1")?.textContent?.trim() || document.title,
              url: window.location.href,
              language,
            },
          }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        persistConversationId(payload.conversation_id);
        loading.remove();
        const answer = String(payload.answer || copy.failed);
        appendMessage("assistant", answer, payload.sources);
        mode.textContent = payload.mode === "llm" ? copy.llm : copy.retrieval;
      } catch (error) {
        loading.remove();
        const failure = appendMessage("assistant", copy.failed, [], "is-error");
        failure.title = error instanceof Error ? error.message : String(error);
      } finally {
        pending = false;
        input.disabled = false;
        send.disabled = false;
        newSession.disabled = false;
        input.focus();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
