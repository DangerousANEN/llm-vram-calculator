/*!
 * Page controller: wires the pinned catalogue to the form, the ledger table and
 * the share/copy controls. All validation lives in state.js.
 */
(function () {
  "use strict";

  var S = window.VRAMState;
  var CATALOG = window.VRAM_CATALOG;
  var $ = function (id) { return document.getElementById(id); };

  var el = {
    model: $("model"),
    quant: $("quant"),
    ctx: $("ctx"),
    ctxRange: $("ctxRange"),
    ctxMax: $("ctx-max"),
    batch: $("batch"),
    kvDtype: $("kvdtype"),
    overhead: $("overhead"),
    notices: $("notices"),
    weights: $("out-weights"),
    weightsBytes: $("out-weights-bytes"),
    weightsSrc: $("out-weights-src"),
    kv: $("out-kv"),
    kvBytes: $("out-kv-bytes"),
    kvPerToken: $("out-kv-pertoken"),
    subtotal: $("out-subtotal"),
    subtotalBytes: $("out-subtotal-bytes"),
    overheadOut: $("out-overhead"),
    overheadBytes: $("out-overhead-bytes"),
    total: $("out-total"),
    totalBytes: $("out-total-bytes"),
    formula: $("formula"),
    prov: $("prov"),
    limits: $("limits"),
    status: $("status"),
    shareTg: $("share-tg"),
    copyReport: $("copy-report"),
    copyLink: $("copy-link"),
    live: $("results-live")
  };

  var state = null;

  function currentModel() {
    return S.modelById(CATALOG, state.modelId) || CATALOG.models[0];
  }

  function readControls() {
    return {
      modelId: el.model.value,
      quant: el.quant.value,
      context: parseInt(el.ctx.value, 10),
      batch: parseInt(el.batch.value, 10),
      overhead: el.overhead.value === "" ? 0 : parseFloat(el.overhead.value),
      kvDtype: el.kvDtype.value
    };
  }

  function fillModels() {
    el.model.innerHTML = "";
    CATALOG.models.forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.display_name + " (" + m.config.num_hidden_layers + " layers)";
      el.model.appendChild(opt);
    });
  }

  function fillQuants(model) {
    var quants = S.quantKeys(model);
    el.quant.innerHTML = "";
    quants.forEach(function (q) {
      var opt = document.createElement("option");
      opt.value = q;
      opt.textContent = q + " \u00b7 " + S.fmtGib(model.weights[q].bytes);
      el.quant.appendChild(opt);
    });
  }

  function showNotices(notices) {
    if (!notices || !notices.length) {
      el.notices.hidden = true;
      el.notices.textContent = "";
      return;
    }
    el.notices.hidden = false;
    el.notices.innerHTML = "";
    notices.forEach(function (n) {
      var div = document.createElement("div");
      div.textContent = n;
      el.notices.appendChild(div);
    });
  }

  function render() {
    var model = currentModel();
    var ledger = S.computeLedger(model, state);
    var w = model.weights[state.quant];

    el.weights.textContent = S.fmtGib(ledger.weightsBytes);
    el.weightsBytes.textContent = S.fmtBytes(ledger.weightsBytes);
    el.weightsSrc.textContent = "exact official file";
    el.weightsSrc.className = "badge exact";
    el.weightsSrc.title =
      (w.source === "official_gguf" ? "official GGUF file listing" : "official safetensors shard sizes") +
      " @ revision " + String(w.gguf_revision || model.source.revision).slice(0, 8) +
      " - " + (w.detail || "");

    el.kv.textContent = S.fmtGib(ledger.kvBytes);
    el.kvBytes.textContent = S.fmtBytes(ledger.kvBytes);
    el.kvPerToken.textContent = ledger.kvPerToken + " B/token/seq";
    el.subtotal.textContent = S.fmtGib(ledger.subtotalBytes);
    el.subtotalBytes.textContent = S.fmtBytes(ledger.subtotalBytes);
    // never print "0.000 GiB" for something that is unknown rather than zero
    el.overheadOut.textContent = ledger.overheadBytes ? S.fmtGib(ledger.overheadBytes) : "\u2014";
    el.overheadBytes.textContent = ledger.overheadBytes
      ? "manual bound, not estimated"
      : "unknown \u00b7 not estimated";
    el.total.textContent = S.fmtGib(ledger.totalBytes);
    el.totalBytes.textContent = S.fmtBytes(ledger.totalBytes);
    el.formula.textContent =
      "KV bytes = 2 (key+value) \u00d7 " + ledger.layers + " layers \u00d7 " + ledger.kvHeads +
      " KV heads \u00d7 head_dim " + ledger.headDim + " \u00d7 " + ledger.elemBytes +
      " bytes/element \u00d7 " + state.context + " tokens \u00d7 " + state.batch + " sequence(s) = " +
      S.fmtBytes(ledger.kvBytes);

    el.prov.innerHTML = "";
    var rows = [
      ["model repo", model.source.repo + " @ " + String(model.source.revision).slice(0, 12)],
      ["config sha256", String(model.source.config_sha256).slice(0, 32) + "\u2026"],
      ["license", model.license],
      ["max context", String(S.maxContextOf(model)) + " tokens (config.max_position_embeddings)"],
      ["weights file", (w.files && w.files[0] ? w.files[0].path : "") + (w.files && w.files.length > 1 ? " + " + (w.files.length - 1) + " more" : "")]
    ];
    rows.forEach(function (r) {
      var line = document.createElement("div");
      line.innerHTML = "<span>" + r[0] + ": </span><code></code>";
      line.querySelector("code").textContent = r[1];
      el.prov.appendChild(line);
    });
    var link = document.createElement("a");
    link.href = model.source.revision_url;
    link.rel = "noopener noreferrer nofollow";
    link.target = "_blank";
    link.textContent = "Open the pinned revision on Hugging Face \u2197";
    el.prov.appendChild(link);

    var limits = [
      "Runtime overhead (CUDA context, allocator fragmentation, activations, workspaces) is not estimated \u2014 add a manual bound only if you measured one.",
      "The total is an arithmetic sum of bytes: it is not measured VRAM and not a guarantee that the model fits.",
      "Only uniform full attention is modelled: MoE, MLA, sliding windows and hybrid layers are refused, not approximated.",
      "Only fp16/bf16 KV cache is modelled; quantized KV cache has no exact-size claim here.",
      "Tensor parallelism, CUDA graphs, paged-attention block padding and prefix caching are not modelled.",
      "Partial CPU offload is not modelled: the GPU resident set then depends on the runtime.",
      "Context is capped at this config's max_position_embeddings; rope/YaRN extrapolation is not modelled."
    ];
    el.limits.innerHTML = "";
    limits.forEach(function (t) {
      var li = document.createElement("li");
      li.textContent = t;
      el.limits.appendChild(li);
    });

    updateUrl();
    updateShare(model, ledger);
    el.live.textContent = "Ledger updated: " + S.fmtGib(ledger.totalBytes) + " total for " + model.display_name + ".";
  }

  function updateUrl() {
    var qs = S.buildQuery(state);
    try {
      history.replaceState(null, "", location.pathname + "?" + qs + location.hash);
    } catch (err) {
      /* file:// or sandboxed iframes may refuse; the app still works */
    }
  }

  function updateShare(model, ledger) {
    var url = location.href;
    var text = S.reportText(model, state, ledger);
    var tg = "https://t.me/share/url?url=" + encodeURIComponent(url) + "&text=" + encodeURIComponent(text);
    el.shareTg.href = tg;
    el.shareTg.dataset.report = text;
  }

  function syncFromState() {
    el.model.value = state.modelId;
    fillQuants(currentModel());
    el.quant.value = state.quant;
    el.ctx.value = String(state.context);
    el.ctxRange.value = String(Math.min(state.context, parseInt(el.ctxRange.max, 10)));
    el.batch.value = String(state.batch);
    el.kvDtype.value = state.kvDtype;
    el.overhead.value = state.overhead ? String(state.overhead) : "";
    el.ctxMax.textContent = "max " + S.maxContextOf(currentModel());
    el.ctxRange.max = String(S.maxContextOf(currentModel()));
  }

  function onChange() {
    var form = readControls();
    var parsed = S.parseState(new URLSearchParams({
      m: form.modelId,
      q: form.quant,
      c: String(Number.isFinite(form.context) ? form.context : 8192),
      b: String(Number.isFinite(form.batch) ? form.batch : 1),
      o: el.overhead.value,
      k: form.kvDtype
    }).toString(), CATALOG);
    state = parsed.state;
    syncFromState();
    showNotices(parsed.notices);
    render();
  }

  function copyText(text, okMessage) {
    function done() {
      el.status.textContent = okMessage;
      setTimeout(function () { el.status.textContent = ""; }, 2500);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else {
      fallback();
    }
    function fallback() {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.top = "-1000px";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        done();
      } catch (err) {
        el.status.textContent = "Copy failed \u2014 select the text manually.";
      }
      document.body.removeChild(ta);
    }
  }

  function init() {
    if (!CATALOG || !CATALOG.models || !CATALOG.models.length) {
      el.notices.hidden = false;
      el.notices.textContent = "Model catalogue failed to load. Reload the page.";
      return;
    }
    fillModels();
    var parsed = S.parseState(location.search, CATALOG);
    state = parsed.state;
    syncFromState();
    el.model.addEventListener("change", function () { state = { modelId: el.model.value, quant: null }; onChange(); });
    el.quant.addEventListener("change", onChange);
    el.batch.addEventListener("change", onChange);
    el.kvDtype.addEventListener("change", onChange);
    el.overhead.addEventListener("change", onChange);
    el.ctx.addEventListener("change", onChange);
    el.ctxRange.addEventListener("input", function () {
      el.ctx.value = el.ctxRange.value;
      onChange();
    });
    el.copyReport.addEventListener("click", function () {
      var model = currentModel();
      copyText(S.reportText(model, state, S.computeLedger(model, state)), "Report copied to clipboard.");
    });
    el.copyLink.addEventListener("click", function () {
      copyText(location.href, "Share link copied to clipboard.");
    });
    showNotices(parsed.notices);
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
