/* mneme showcase — page bootstrappers.
 *
 * Each template includes one bootstrapper. Functions are scoped to a single
 * page so loading the JS on every page is harmless.
 */

(function () {
  // ---------- helpers ----------

  function pct(x) {
    return (x * 100).toFixed(1) + "%";
  }

  function bytesPretty(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MB";
    return (n / 1024 / 1024 / 1024).toFixed(2) + " GB";
  }

  function layerBadge(layer) {
    return `<span class="layer-${layer}">${layer}</span>`;
  }

  function escape(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function timeAgo(ts) {
    const now = Date.now() / 1000;
    const dt = Math.max(0, now - ts);
    if (dt < 1) return "just now";
    if (dt < 60) return Math.floor(dt) + "s ago";
    if (dt < 3600) return Math.floor(dt / 60) + "m ago";
    return Math.floor(dt / 3600) + "h ago";
  }

  function ageFmt(s) {
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m";
    if (s < 86400) return Math.floor(s / 3600) + "h";
    return Math.floor(s / 86400) + "d";
  }

  // Bind a value to every [data-stat=KEY] element.
  function setStat(key, value) {
    document.querySelectorAll(`[data-stat="${key}"]`).forEach((el) => {
      el.textContent = value;
    });
  }

  // ---------- DASHBOARD ----------

  window.startDashboard = function () {
    let timer = null;

    async function tick() {
      try {
        const resp = await fetch("/api/stats");
        const s = await resp.json();
        setStat("queries", s.queries);
        setStat("hits_exact", s.hits_exact);
        setStat("hits_semantic", s.hits_semantic);
        setStat("misses", s.misses);
        setStat("hit_rate_pct", pct(s.hit_rate));
        setStat("cache_entries", s.cache_entries);
        setStat("memory_kb", bytesPretty(s.memory_bytes_estimate));
        setStat("index_memory_actual", s.index_memory_bytes != null ? bytesPretty(s.index_memory_bytes) : "—");
        setStat("index_tombstones", s.index_tombstone_count != null ? s.index_tombstone_count : "—");
        setStat("vector_dtype", s.vector_dtype);
        setStat("similarity_threshold", s.similarity_threshold);
        syncThresholdSlider(s.similarity_threshold);
        setStat("model", s.model);
        setStat("embedder_fingerprint", s.embedder_fingerprint);
        setStat("avg_llm_seconds", s.avg_llm_seconds.toFixed(2));
        setStat("llm_seconds_saved", s.llm_seconds_saved.toFixed(1) + " s");
        setStat("spark_ok", s.spark_ok ? "reachable" : "unreachable");

        // namespace breakdown
        const tbody = document.querySelector("#ns-table tbody");
        if (tbody) {
          tbody.innerHTML = "";
          const names = Object.keys(s.namespaces).sort();
          if (names.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="muted">no traffic yet</td></tr>`;
          }
          for (const ns of names) {
            const c = s.namespaces[ns];
            tbody.insertAdjacentHTML(
              "beforeend",
              `<tr><td>${escape(ns)}</td><td>${c.queries}</td><td>${c.hits_exact}</td><td>${c.hits_semantic}</td><td>${c.misses}</td></tr>`,
            );
          }
        }

        // recent queries
        const rb = document.getElementById("recent-body");
        if (rb) {
          rb.innerHTML = "";
          if (s.recent.length === 0) {
            rb.innerHTML = `<tr><td colspan="6" class="muted">no traffic yet</td></tr>`;
          }
          for (const r of s.recent.slice(0, 20)) {
            rb.insertAdjacentHTML(
              "beforeend",
              `<tr>
                <td class="muted">${timeAgo(r.ts)}</td>
                <td>${escape(r.namespace)}</td>
                <td>${layerBadge(r.layer)}</td>
                <td>${r.latency_ms.toFixed(1)} ms</td>
                <td class="query-cell" title="${escape(r.query)}">${escape(r.query)}</td>
                <td><code>${escape(r.intent)}</code></td>
              </tr>`,
            );
          }
        }
      } catch (e) {
        // network blip; quiet.
      }
    }

    document.getElementById("clear-btn")?.addEventListener("click", async () => {
      const scope = document.getElementById("clear-scope")?.value || "";
      const label = scope ? `wipe namespace "${scope}"` : "wipe ALL namespaces and reset counters";
      if (!confirm(`Confirm: ${label}?`)) return;
      await fetch("/api/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ namespace: scope }),
      });
      tick();
    });

    document.getElementById("compact-btn")?.addEventListener("click", async () => {
      const result = document.getElementById("compact-result");
      if (result) result.textContent = "compacting…";
      try {
        const resp = await fetch("/api/compact", { method: "POST" });
        const r = await resp.json();
        if (result) {
          result.textContent = r.reclaimed === 0
            ? `nothing to reclaim — index is already clean (${bytesPretty(r.index_memory_bytes)})`
            : `reclaimed ${r.reclaimed} tombstones · index now ${bytesPretty(r.index_memory_bytes)}`;
        }
        tick();
      } catch (e) {
        if (result) result.textContent = "compact failed; check logs";
      }
    });

    tick();
    timer = setInterval(tick, 1000);
    bindThresholdSlider();
  };

  // Threshold slider: debounced POST on drag, immediate on release. Reads
  // the live value from the dashboard's /api/stats poll so it stays in sync
  // when other clients / API callers change it.
  function bindThresholdSlider() {
    const slider = document.getElementById("threshold-slider");
    if (!slider) return;
    const valueEl = document.getElementById("threshold-value");
    let debounce = null;

    async function commit(v) {
      try {
        await fetch("/api/threshold", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value: v }),
        });
      } catch (e) {
        // Non-fatal; the next stats tick will reconcile.
      }
    }

    slider.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      valueEl.textContent = v.toFixed(2);
      clearTimeout(debounce);
      debounce = setTimeout(() => commit(v), 150);
    });
    slider.addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      clearTimeout(debounce);
      commit(v);
    });
  }

  // Called from the dashboard tick with the latest /api/stats payload. The
  // slider element re-syncs only when not actively being dragged.
  function syncThresholdSlider(live) {
    const slider = document.getElementById("threshold-slider");
    if (!slider) return;
    if (slider.matches(":active")) return;
    if (Math.abs(parseFloat(slider.value) - live) <= 0.005) return;
    slider.value = live;
    document.getElementById("threshold-value").textContent = live.toFixed(2);
  }

  // ---------- TRY IT ----------

  function renderResult(elId, r) {
    const el = document.getElementById(elId);
    el.innerHTML = "";
    const rows = [
      ["intent", `<code>${escape(r.intent)}</code>`],
      ["layer", layerBadge(r.layer)],
      r.similarity != null ? ["similarity", r.similarity.toFixed(4)] : null,
      r.confidence != null ? ["confidence", r.confidence.toFixed(3)] : null,
      r.age_seconds != null ? ["entry age", ageFmt(r.age_seconds)] : null,
      ["namespace", r.namespace],
      ["latency", r.latency_ms.toFixed(1) + " ms"],
      r.llm_seconds != null ? ["LLM call", r.llm_seconds.toFixed(2) + " s"] : null,
    ].filter(Boolean);
    for (const [k, v] of rows) {
      el.insertAdjacentHTML("beforeend", `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`);
    }
  }

  window.startTryIt = function () {
    document.querySelectorAll("[data-preset]").forEach((b) => {
      b.addEventListener("click", () => {
        document.getElementById("query").value = b.dataset.preset;
      });
    });

    async function classify(bypass) {
      const q = document.getElementById("query").value.trim();
      if (!q) return;
      const ns = document.getElementById("namespace").value;
      const targetEl = bypass ? "result-bypassed" : "result-cached";
      document.getElementById(targetEl).innerHTML = `<div class="empty">Calling…</div>`;
      const resp = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, namespace: ns, bypass }),
      });
      const r = await resp.json();
      renderResult(targetEl, r);
    }

    document.getElementById("classify-form").addEventListener("submit", (e) => {
      e.preventDefault();
      classify(false);
    });
    document.getElementById("bypass-btn").addEventListener("click", () => {
      classify(true);
    });
  };

  // ---------- STRESS TEST ----------

  window.startStress = function () {
    let chart = null;
    const startBtn = document.getElementById("stress-start");
    const cancelBtn = document.getElementById("stress-cancel");
    let abortCtl = null;

    function setRunning(running) {
      startBtn.disabled = running;
      cancelBtn.disabled = !running;
    }

    function ensureChart() {
      if (chart) {
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.update();
        return;
      }
      const ctx = document.getElementById("hit-chart").getContext("2d");
      chart = new Chart(ctx, {
        type: "line",
        data: {
          labels: [],
          datasets: [{
            label: "Cumulative hit rate",
            data: [],
            borderColor: "#4ea7ff",
            backgroundColor: "rgba(78, 167, 255, 0.15)",
            fill: true,
            pointRadius: 0,
            tension: 0.2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            y: { min: 0, max: 1, ticks: { callback: (v) => (v * 100) + "%" }, grid: { color: "#2a3340" } },
            x: { grid: { color: "#2a3340" } },
          },
          plugins: { legend: { display: false } },
        },
      });
    }

    startBtn.addEventListener("click", () => {
      const ns = document.getElementById("stress-namespace").value;
      const resetFirst = document.getElementById("stress-reset-first")?.checked || false;
      const bypass = document.getElementById("stress-bypass")?.checked || false;
      ensureChart();
      document.getElementById("stress-tail").innerHTML = "";
      document.getElementById("stress-bar").style.width = "0%";
      document.getElementById("stress-counter").textContent = "0 / ?";
      document.getElementById("stress-hitrate").textContent = "hit rate: 0%";
      document.getElementById("stress-current").textContent =
        resetFirst ? "clearing cache…" : (bypass ? "bypassing cache…" : "starting…");

      setRunning(true);
      abortCtl = new AbortController();

      // EventSource doesn't support POST + JSON, so we use fetch + reader.
      fetch("/api/stress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ namespace: ns, reset_first: resetFirst, bypass: bypass }),
        signal: abortCtl.signal,
      })
        .then((resp) => {
          const reader = resp.body.getReader();
          const decoder = new TextDecoder();
          let buf = "";
          function pump() {
            return reader.read().then(({ done, value }) => {
              if (done) {
                document.getElementById("stress-current").textContent = "done";
                setRunning(false);
                return;
              }
              buf += decoder.decode(value, { stream: true });
              let sep;
              while ((sep = buf.indexOf("\n\n")) !== -1) {
                const chunk = buf.slice(0, sep);
                buf = buf.slice(sep + 2);
                if (chunk.startsWith("data: ")) {
                  const json = chunk.slice(6);
                  let payload;
                  try { payload = JSON.parse(json); } catch (e) { continue; }
                  if (payload.seq) handleEvent(payload);
                } else if (chunk.startsWith("event: end")) {
                  // handled by `done` above
                }
              }
              return pump();
            });
          }
          return pump();
        })
        .catch(() => { setRunning(false); });
    });

    cancelBtn.addEventListener("click", () => {
      if (abortCtl) abortCtl.abort();
      setRunning(false);
      document.getElementById("stress-current").textContent = "cancelled";
    });

    function handleEvent(p) {
      const pct = (p.seq / p.total) * 100;
      document.getElementById("stress-bar").style.width = pct.toFixed(1) + "%";
      document.getElementById("stress-counter").textContent = `${p.seq} / ${p.total}`;
      document.getElementById("stress-hitrate").textContent = `hit rate: ${(p.cumulative_hit_rate * 100).toFixed(1)}%`;
      document.getElementById("stress-current").textContent = p.query.slice(0, 60);

      chart.data.labels.push(p.seq);
      chart.data.datasets[0].data.push(p.cumulative_hit_rate);
      chart.update("none");

      const tbody = document.getElementById("stress-tail");
      tbody.insertAdjacentHTML(
        "afterbegin",
        `<tr>
          <td>${p.seq}</td>
          <td>${layerBadge(p.layer)}</td>
          <td>${p.latency_ms.toFixed(1)} ms</td>
          <td class="query-cell" title="${escape(p.query)}">${escape(p.query)}</td>
          <td><code>${escape(p.intent)}</code></td>
        </tr>`
      );
      // Keep only top 12.
      while (tbody.children.length > 12) tbody.removeChild(tbody.lastChild);
    }
  };

  // ---------- INSPECTOR ----------

  window.startInspector = function () {
    let entries = [];

    async function load() {
      const resp = await fetch("/api/inspector");
      const data = await resp.json();
      entries = data.entries;
      const namespaces = [...new Set(entries.map((e) => e.namespace))].sort();
      const sel = document.getElementById("filter-ns");
      const cur = sel.value;
      sel.innerHTML = `<option value="">(all)</option>` + namespaces.map((n) => `<option value="${escape(n)}">${escape(n)}</option>`).join("");
      sel.value = cur;
      render();
    }

    function render() {
      const ns = document.getElementById("filter-ns").value;
      const q = document.getElementById("filter-q").value.toLowerCase();
      const filtered = entries.filter((e) => {
        if (ns && e.namespace !== ns) return false;
        if (q && !(e.query.toLowerCase().includes(q) || e.response.toLowerCase().includes(q))) return false;
        return true;
      });
      document.getElementById("entry-count").textContent = `${filtered.length} entries`;
      const tbody = document.getElementById("entries-body");
      tbody.innerHTML = "";
      if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="muted">no entries match the filter</td></tr>`;
        return;
      }
      for (const e of filtered) {
        tbody.insertAdjacentHTML(
          "beforeend",
          `<tr>
            <td>${e.id}</td>
            <td>${escape(e.namespace)}</td>
            <td class="q">${escape(e.query)}</td>
            <td><code>${escape(e.response)}</code></td>
            <td>${ageFmt(e.age_seconds)}</td>
            <td>${e.access_count}</td>
          </tr>`
        );
      }
    }

    document.getElementById("filter-ns").addEventListener("change", render);
    document.getElementById("filter-q").addEventListener("input", render);
    document.getElementById("refresh-btn").addEventListener("click", load);

    load();
  };

  // ---------- TENANTS ----------

  window.startTenants = function () {
    document.querySelectorAll("[data-tenant]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const ns = btn.dataset.tenant;
        const paneId = ns === "tenant_a" ? "tenant-a-pane" : "tenant-b-pane";
        const pane = document.getElementById(paneId);
        pane.innerHTML = `<div class="empty">Running…</div>`;
        btn.disabled = true;
        try {
          const resp = await fetch("/api/tenants/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ namespace: ns }),
          });
          const data = await resp.json();
          renderTenant(pane, data);
        } finally {
          btn.disabled = false;
        }
      });
    });
  };

  function renderTenant(pane, data) {
    const hits = data.results.filter((r) => r.layer !== "miss").length;
    const html = [];
    html.push(`<table class="recent-table">
      <thead><tr><th>Layer</th><th>Latency</th><th>Query</th><th>→ Intent</th></tr></thead>
      <tbody>`);
    for (const r of data.results) {
      html.push(`<tr>
        <td>${layerBadge(r.layer)}</td>
        <td>${r.latency_ms.toFixed(1)} ms</td>
        <td class="query-cell" title="${escape(r.query)}">${escape(r.query)}</td>
        <td><code>${escape(r.intent)}</code></td>
      </tr>`);
    }
    html.push(`</tbody></table>`);
    html.push(`<div class="sub" style="margin-top:8px;"><strong>${hits}</strong> / ${data.results.length} cache hits</div>`);
    pane.innerHTML = html.join("");
  }

  // -------------------------------------------------------------------------
  // Dedup page
  // -------------------------------------------------------------------------

  const _DEDUP_SAMPLE = [
    "Apple announces new MacBook Pro with M5 chip",
    "Apple announces new MacBook Pro with M5 chip",
    "Apple announces new MacBook Pro with M5 chip processor",
    "Google launches new Pixel phone",
    "Google launches a new Pixel phone today",
    "Microsoft releases Windows 12",
    "Apple announces new iPad with M5 chip",
  ].join("\n");

  window.startDedup = function () {
    const input = document.getElementById("dedup-input");
    const tbody = document.getElementById("dedup-results");
    const summary = document.getElementById("dedup-summary");

    document.getElementById("dedup-load-sample")?.addEventListener("click", () => {
      input.value = _DEDUP_SAMPLE;
    });
    document.getElementById("dedup-clear")?.addEventListener("click", () => {
      tbody.innerHTML = "";
      summary.textContent = "";
    });
    document.getElementById("dedup-run")?.addEventListener("click", async () => {
      const items = (input.value || "")
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
      if (items.length === 0) return;
      summary.textContent = "running…";
      const resp = await fetch("/api/dedup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      });
      const data = await resp.json();
      const rows = data.results || [];
      tbody.innerHTML = "";
      let kept = 0, dropped = 0;
      rows.forEach((r, i) => {
        const decision = r.is_duplicate
          ? `<span class="layer-badge layer-exact">DROP</span>`
          : `<span class="layer-badge layer-miss">KEEP</span>`;
        const sim = r.similarity != null ? r.similarity.toFixed(3) : "—";
        tbody.insertAdjacentHTML(
          "beforeend",
          `<tr>
            <td>${i + 1}</td>
            <td>${decision}</td>
            <td>${layerBadge(r.layer)}</td>
            <td>${sim}</td>
            <td>${r.latency_ms.toFixed(2)} ms</td>
            <td class="query-cell">${escape(r.content)}</td>
          </tr>`,
        );
        if (r.is_duplicate) dropped++; else kept++;
      });
      summary.innerHTML = `<strong>${kept}</strong> kept · <strong>${dropped}</strong> dropped as near-duplicates`;
    });
  };

  // -------------------------------------------------------------------------
  // Translate page
  // -------------------------------------------------------------------------

  const _TRANSLATE_SAMPLE = "How do I reset my password?";
  let _translateHistory = [];

  window.startTranslate = function () {
    const input = document.getElementById("translate-input");
    const target = document.getElementById("translate-target");
    const empty = document.getElementById("translate-result-empty");
    const card = document.getElementById("translate-result");
    const out = document.getElementById("translate-output");
    const layerEl = document.getElementById("translate-layer");
    const latEl = document.getElementById("translate-latency");
    const llmEl = document.getElementById("translate-llm");
    const llmRow = document.getElementById("translate-llm-row");
    const histBody = document.getElementById("translate-history");

    document.getElementById("translate-load-sample")?.addEventListener("click", () => {
      input.value = _TRANSLATE_SAMPLE;
    });
    document.getElementById("translate-run")?.addEventListener("click", async () => {
      const text = (input.value || "").trim();
      const tgt = target.value;
      if (!text) return;
      empty.style.display = "none";
      card.style.display = "block";
      out.textContent = "translating…";
      latEl.textContent = ""; llmEl.textContent = "";
      const resp = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, target_lang: tgt }),
      });
      const r = await resp.json();
      out.textContent = r.translation;
      layerEl.innerHTML = layerBadge(r.layer);
      latEl.textContent = r.latency_ms.toFixed(2) + " ms";
      if (r.llm_seconds != null) {
        llmEl.textContent = r.llm_seconds.toFixed(2) + " s";
        llmRow.style.display = "inline";
      } else {
        llmRow.style.display = "none";
      }
      _translateHistory.unshift({ ...r });
      _translateHistory = _translateHistory.slice(0, 20);
      histBody.innerHTML = _translateHistory
        .map((h, i) => `<tr>
          <td>${i + 1}</td>
          <td>${layerBadge(h.layer)}</td>
          <td>${h.latency_ms.toFixed(2)} ms</td>
          <td>en→${escape(h.target_lang)}: ${escape(h.source.slice(0, 60))}</td>
          <td>${escape(h.translation.slice(0, 80))}</td>
        </tr>`)
        .join("");
    });
  };

  // -------------------------------------------------------------------------
  // Agent memory page
  // -------------------------------------------------------------------------

  const _AGENT_SAMPLE = "Summarize the latest pull request and flag any breaking API changes";
  let _agentHistory = [];

  window.startAgent = function () {
    const input = document.getElementById("agent-input");
    const agentSel = document.getElementById("agent-id");
    const empty = document.getElementById("agent-result-empty");
    const card = document.getElementById("agent-result");
    const planEl = document.getElementById("agent-plan");
    const layerEl = document.getElementById("agent-layer");
    const latEl = document.getElementById("agent-latency");
    const llmEl = document.getElementById("agent-llm");
    const llmRow = document.getElementById("agent-llm-row");
    const histBody = document.getElementById("agent-history");

    document.getElementById("agent-load-sample")?.addEventListener("click", () => {
      input.value = _AGENT_SAMPLE;
    });
    document.getElementById("agent-run")?.addEventListener("click", async () => {
      const task = (input.value || "").trim();
      const agentId = agentSel.value;
      if (!task) return;
      empty.style.display = "none";
      card.style.display = "block";
      planEl.textContent = "generating plan…";
      latEl.textContent = ""; llmEl.textContent = "";
      const resp = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task, agent_id: agentId }),
      });
      const r = await resp.json();
      planEl.textContent = r.plan;
      layerEl.innerHTML = layerBadge(r.layer);
      latEl.textContent = r.latency_ms.toFixed(2) + " ms";
      if (r.llm_seconds != null) {
        llmEl.textContent = r.llm_seconds.toFixed(2) + " s";
        llmRow.style.display = "inline";
      } else {
        llmRow.style.display = "none";
      }
      _agentHistory.unshift({ ...r });
      _agentHistory = _agentHistory.slice(0, 20);
      histBody.innerHTML = _agentHistory
        .map((h, i) => `<tr>
          <td>${i + 1}</td>
          <td><code>${escape(h.agent_id)}</code></td>
          <td>${layerBadge(h.layer)}</td>
          <td>${h.latency_ms.toFixed(2)} ms</td>
          <td class="query-cell">${escape(h.task)}</td>
        </tr>`)
        .join("");
    });
  };

  // -------------------------------------------------------------------------
  // RAG page
  // -------------------------------------------------------------------------

  const _RAG_SAMPLE = "How do I reset my password?";
  let _ragHistory = [];

  window.startRAG = function () {
    const input = document.getElementById("rag-input");
    const empty = document.getElementById("rag-result-empty");
    const card = document.getElementById("rag-result");
    const ans = document.getElementById("rag-answer");
    const ctxEl = document.getElementById("rag-contexts");
    const layerEl = document.getElementById("rag-layer");
    const latEl = document.getElementById("rag-latency");
    const llmEl = document.getElementById("rag-llm");
    const llmRow = document.getElementById("rag-llm-row");
    const histBody = document.getElementById("rag-history");

    document.getElementById("rag-load-sample")?.addEventListener("click", () => {
      input.value = _RAG_SAMPLE;
    });
    document.getElementById("rag-run")?.addEventListener("click", async () => {
      const question = (input.value || "").trim();
      if (!question) return;
      empty.style.display = "none";
      card.style.display = "block";
      ans.textContent = "retrieving + synthesizing…";
      ctxEl.innerHTML = "";
      latEl.textContent = ""; llmEl.textContent = "";
      const resp = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const r = await resp.json();
      ans.textContent = r.answer;
      layerEl.innerHTML = layerBadge(r.layer);
      latEl.textContent = r.latency_ms.toFixed(2) + " ms";
      if (r.llm_seconds != null) {
        llmEl.textContent = r.llm_seconds.toFixed(2) + " s";
        llmRow.style.display = "inline";
      } else {
        llmRow.style.display = "none";
      }
      ctxEl.innerHTML = (r.contexts || [])
        .map((c, i) => `<div style="margin: 8px 0; padding: 10px; background: var(--bg-card-2); border-radius: 6px; border-left: 3px solid var(--accent);">
          <div class="sub" style="margin-bottom: 4px;"><code>[${i + 1}] ${escape(r.chunk_ids[i] || "?")}</code></div>
          <div style="font-size: 13px; line-height: 1.5;">${escape(c)}</div>
        </div>`)
        .join("");
      _ragHistory.unshift({ ...r });
      _ragHistory = _ragHistory.slice(0, 20);
      histBody.innerHTML = _ragHistory
        .map((h, i) => `<tr>
          <td>${i + 1}</td>
          <td>${layerBadge(h.layer)}</td>
          <td>${h.latency_ms.toFixed(2)} ms</td>
          <td class="query-cell">${escape(h.question)}</td>
          <td><code>${escape((h.chunk_ids || [])[0] || "—")}</code></td>
        </tr>`)
        .join("");
    });
  };
})();
