document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("compare-form");
  const q1 = document.getElementById("q1");
  const q2 = document.getElementById("q2");
  const fixTyposBox = document.getElementById("fix-typos");
  const submitBtn = document.getElementById("submit-btn");
  const errorEl = document.getElementById("form-error");
  const langWarnQ1 = document.getElementById("lang-warning-q1");
  const langWarnQ2 = document.getElementById("lang-warning-q2");

  const placeholder = document.getElementById("result-placeholder");
  const card = document.getElementById("result-card");
  const verdictLabel = document.getElementById("verdict-label");
  const verdictSub = document.getElementById("verdict-sub");
  const gaugeFill = document.getElementById("gauge-fill");
  const gaugeValue = document.getElementById("gauge-value");
  const breakdown = document.getElementById("breakdown");
  const spellingNote = document.getElementById("spelling-note");
  const overlapQ1 = document.getElementById("overlap-q1");
  const overlapQ2 = document.getElementById("overlap-q2");

  const LANG_NAMES = {
    en: "English", fr: "French", es: "Spanish", de: "German", it: "Italian",
    pt: "Portuguese", nl: "Dutch", ru: "Russian", zh: "Chinese", ja: "Japanese",
    ko: "Korean", ar: "Arabic", hi: "Hindi", tr: "Turkish", pl: "Polish",
  };

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      q1.value = chip.dataset.q1;
      q2.value = chip.dataset.q2;
      form.requestSubmit();
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.hidden = true;

    const question1 = q1.value.trim();
    const question2 = q2.value.trim();
    if (!question1 || !question2) {
      errorEl.textContent = "Please enter both questions.";
      errorEl.hidden = false;
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Comparing…";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question1,
          question2,
          fix_typos: fixTyposBox.checked,
        }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Something went wrong.");
      }

      renderResult(data);
      saveToHistory(question1, question2, data);
    } catch (err) {
      errorEl.textContent = err.message || "Something went wrong. Please try again.";
      errorEl.hidden = false;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Compare questions";
    }
  });

  function renderLangWarning(el, langInfo) {
    if (langInfo && !langInfo.is_english) {
      const name = LANG_NAMES[langInfo.lang] || langInfo.lang;
      el.textContent = `Detected as ${name}, not English — the model was trained on English only, so this result may be unreliable.`;
      el.hidden = false;
    } else {
      el.hidden = true;
    }
  }

  function renderResult(data) {
    placeholder.hidden = true;
    card.hidden = false;

    renderLangWarning(langWarnQ1, data.language && data.language.question1);
    renderLangWarning(langWarnQ2, data.language && data.language.question2);

    const pct = Math.round(data.probability * 100);
    const isDup = data.label === "duplicate";

    verdictLabel.textContent = isDup
      ? "Likely the same question"
      : "Likely different questions";
    verdictLabel.className = "verdict-label " + (isDup ? "is-dup" : "is-not");

    verdictSub.textContent = isDup
      ? "A reader would probably expect the same answer to both."
      : "These seem to ask for different information, even if some words overlap.";

    gaugeFill.style.width = pct + "%";
    gaugeFill.className = "gauge-fill " + (isDup ? "" : "is-not");
    gaugeValue.textContent = pct + "%";

    // Spelling-correction transparency note
    if (data.spelling) {
      const parts = [];
      ["question1", "question2"].forEach((key, i) => {
        const info = data.spelling[key];
        if (info && info.corrections && info.corrections.length) {
          const fixes = info.corrections
            .map((c) => `"${c.original}" → "${c.corrected}"`)
            .join(", ");
          parts.push(`Q${i + 1}: ${fixes}`);
        }
      });
      if (parts.length) {
        spellingNote.innerHTML = "<strong>Typos fixed before comparing</strong> — " + parts.join(" · ");
        spellingNote.hidden = false;
      } else {
        spellingNote.hidden = true;
      }
    } else {
      spellingNote.hidden = true;
    }

    // Shared-word highlighting
    if (data.highlight) {
      overlapQ1.innerHTML = renderHighlighted(data.highlight.question1);
      overlapQ2.innerHTML = renderHighlighted(data.highlight.question2);
    } else {
      overlapQ1.innerHTML = "";
      overlapQ2.innerHTML = "";
    }

    breakdown.innerHTML = "";
    data.breakdown.forEach((group) => {
      const groupEl = document.createElement("div");
      groupEl.className = "breakdown-group";

      const h4 = document.createElement("h4");
      h4.textContent = group.label;
      groupEl.appendChild(h4);

      group.items.forEach((item) => {
        const row = document.createElement("div");
        row.className = "breakdown-row";
        row.innerHTML = `<span class="name">${item.name}</span><span class="value">${item.value}</span>`;
        groupEl.appendChild(row);
      });

      breakdown.appendChild(groupEl);
    });
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderHighlighted(tokens) {
    if (!tokens) return "";
    return tokens
      .map((t) => (t.shared ? `<mark>${escapeHtml(t.text)}</mark>` : escapeHtml(t.text)))
      .join("");
  }

  // ---------------- History (client-side only, localStorage) ----------------
  const HISTORY_KEY = "echomatch_history";
  const HISTORY_MAX = 20;
  const historyList = document.getElementById("history-list");
  const historyEmpty = document.getElementById("history-empty");
  const clearHistoryBtn = document.getElementById("clear-history-btn");

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function saveToHistory(question1, question2, result) {
    try {
      const history = loadHistory();
      history.unshift({
        question1,
        question2,
        label: result.label,
        probability: result.probability,
        timestamp: Date.now(),
      });
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_MAX)));
      renderHistory();
    } catch (e) {
      // localStorage unavailable or full — history is a nice-to-have, fail silently
    }
  }

  function truncate(s, n) {
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  function renderHistory() {
    const history = loadHistory();
    historyList.querySelectorAll(".history-item").forEach((el) => el.remove());

    if (history.length === 0) {
      historyEmpty.hidden = false;
      return;
    }
    historyEmpty.hidden = true;

    history.forEach((entry) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "history-item";

      const pct = Math.round(entry.probability * 100);
      const isDup = entry.label === "duplicate";

      btn.innerHTML = `
        <p class="history-item-q">${escapeHtml(truncate(entry.question1, 50))}</p>
        <p class="history-item-q">${escapeHtml(truncate(entry.question2, 50))}</p>
        <div class="history-item-meta">
          <span class="${isDup ? "is-dup" : "is-not"}">${isDup ? "Duplicate" : "Not duplicate"}</span>
          <span>${pct}%</span>
        </div>
      `;
      btn.addEventListener("click", () => {
        q1.value = entry.question1;
        q2.value = entry.question2;
        window.scrollTo({ top: 0, behavior: "smooth" });
        form.requestSubmit();
      });
      li.appendChild(btn);
      historyList.appendChild(li);
    });
  }

  clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  renderHistory();

  // ---------------- Batch CSV compare ----------------
  const batchForm = document.getElementById("batch-form");
  const batchFile = document.getElementById("batch-file");
  const batchSubmitBtn = document.getElementById("batch-submit-btn");
  const batchStatus = document.getElementById("batch-status");

  batchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    batchStatus.hidden = true;

    const file = batchFile.files[0];
    if (!file) return;

    batchSubmitBtn.disabled = true;
    batchSubmitBtn.textContent = "Comparing…";

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/predict_batch", { method: "POST", body: formData });

      if (!res.ok) {
        let message = "Something went wrong with that file.";
        try {
          const data = await res.json();
          message = data.error || message;
        } catch (e) {
          /* response wasn't JSON */
        }
        throw new Error(message);
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "echomatch_results.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      batchStatus.textContent = "Done — your results CSV has started downloading.";
      batchStatus.className = "tool-status is-success";
      batchStatus.hidden = false;
    } catch (err) {
      batchStatus.textContent = err.message || "Something went wrong with that file.";
      batchStatus.className = "tool-status is-error";
      batchStatus.hidden = false;
    } finally {
      batchSubmitBtn.disabled = false;
      batchSubmitBtn.textContent = "Upload & compare";
    }
  });
});
