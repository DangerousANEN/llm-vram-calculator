/*!
 * URL-state handling for the VRAM calculator.
 *
 * Pure functions only (no DOM), so the same code is unit-tested under node and
 * used by the page. Every incoming value is validated against the pinned
 * catalogue; nothing from the URL is trusted or evaluated.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.VRAMState = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var MAX_CONTEXT = 1048576;
  var MAX_BATCH = 4096;
  var MAX_OVERHEAD_GIB = 4096;
  var KV_DTYPES = { fp16: 2, bf16: 2 };
  var GIB = 1024 * 1024 * 1024;

  function toInt(value, fallback) {
    if (typeof value === "number") {
      return Number.isFinite(value) ? Math.trunc(value) : fallback;
    }
    if (typeof value !== "string") return fallback;
    var trimmed = value.trim();
    if (!/^[0-9]{1,9}$/.test(trimmed)) return fallback;
    var n = parseInt(trimmed, 10);
    return Number.isFinite(n) ? n : fallback;
  }

  function modelById(catalog, id) {
    if (!catalog || !catalog.models) return null;
    var wanted = String(id || "").toLowerCase();
    for (var i = 0; i < catalog.models.length; i += 1) {
      var m = catalog.models[i];
      if (String(m.id).toLowerCase() === wanted) return m;
      if (m.short_name && String(m.short_name).toLowerCase() === wanted) return m;
    }
    return null;
  }

  function quantKeys(model) {
    if (!model || !model.weights) return [];
    var order = ["bf16", "fp16", "q8_0", "q6_k", "q5_k_m", "q5_0", "q4_k_m", "q4_0", "q3_k_m", "q2_k"];
    var out = [];
    for (var i = 0; i < order.length; i += 1) {
      if (Object.prototype.hasOwnProperty.call(model.weights, order[i])) out.push(order[i]);
    }
    return out;
  }

  function maxContextOf(model) {
    var cfg = (model && model.config) || {};
    var v = toInt(cfg.max_position_embeddings, 0);
    if (!v) return MAX_CONTEXT;
    return Math.min(v, MAX_CONTEXT);
  }

  /**
   * Validate URL (or form) parameters.
   * @returns {{state: object, notices: string[]}} never throws.
   */
  function parseState(search, catalog) {
    var notices = [];
    var params;
    try {
      params = new URLSearchParams(search || "");
    } catch (err) {
      params = new URLSearchParams("");
      notices.push("The share link could not be parsed; showing defaults.");
    }

    var catalogModels = (catalog && catalog.models) || [];
    if (!catalogModels.length) {
      throw new Error("catalogue is empty");
    }
    var model = modelById(catalog, params.get("m")) || catalogModels[0];
    if (params.get("m") && !modelById(catalog, params.get("m"))) {
      notices.push("Unknown model in the link; showing " + model.display_name + ".");
    }

    var quants = quantKeys(model);
    var quant = String(params.get("q") || "").toLowerCase();
    if (quants.indexOf(quant) === -1) {
      if (quant) notices.push("Unsupported quantisation '" + quant + "' for this model; using " + quants[0] + ".");
      quant = quants[0];
    }

    var maxCtx = maxContextOf(model);
    var context = toInt(params.get("c"), 8192);
    if (context < 1) {
      notices.push("Context must be at least 1 token; using 8192.");
      context = 8192;
    }
    if (context > maxCtx) {
      notices.push(
        "Context " + context + " exceeds this config's max_position_embeddings (" + maxCtx + "); clamped to " + maxCtx + "."
      );
      context = maxCtx;
    }

    var batch = toInt(params.get("b"), 1);
    if (batch < 1) {
      notices.push("Batch must be at least 1; using 1.");
      batch = 1;
    }
    if (batch > MAX_BATCH) {
      notices.push("Batch " + batch + " exceeds the supported bound " + MAX_BATCH + "; clamped.");
      batch = MAX_BATCH;
    }

    var overheadRaw = params.get("o");
    var overhead = 0;
    if (overheadRaw !== null && overheadRaw !== "") {
      if (!/^[0-9]{1,4}([.][0-9]{1,3})?$/.test(overheadRaw.trim())) {
        notices.push("Manual overhead must be a non-negative number of GiB; ignored.");
      } else {
        overhead = Math.min(parseFloat(overheadRaw), MAX_OVERHEAD_GIB);
      }
    }

    var kvDtype = String(params.get("k") || "fp16").toLowerCase();
    if (!Object.prototype.hasOwnProperty.call(KV_DTYPES, kvDtype)) {
      notices.push("KV dtype '" + kvDtype + "' is not modelled (only fp16/bf16); using fp16.");
      kvDtype = "fp16";
    }

    return {
      state: {
        modelId: model.id,
        quant: quant,
        context: context,
        batch: batch,
        overhead: overhead,
        kvDtype: kvDtype
      },
      notices: notices
    };
  }

  function buildQuery(state) {
    var params = new URLSearchParams();
    params.set("m", state.modelId);
    params.set("q", state.quant);
    params.set("c", String(state.context));
    params.set("b", String(state.batch));
    if (state.overhead && state.overhead > 0) params.set("o", String(state.overhead));
    if (state.kvDtype && state.kvDtype !== "fp16") params.set("k", state.kvDtype);
    return params.toString();
  }

  /** Byte-level ledger. Mirrors vramcalc.core.compute_ledger. */
  function computeLedger(model, state) {
    var cfg = model.config || {};
    var heads = toInt(cfg.num_attention_heads, 0);
    var kvHeads = toInt(cfg.num_key_value_heads, heads);
    var hidden = toInt(cfg.hidden_size, 0);
    var headDim = toInt(cfg.head_dim, heads && hidden % heads === 0 ? hidden / heads : 0);
    var layers = toInt(cfg.num_hidden_layers, 0);
    var elemBytes = KV_DTYPES[state.kvDtype] || 2;
    var weights = (model.weights[state.quant] || {}).bytes || 0;
    var kvPerToken = 2 * layers * kvHeads * headDim * elemBytes;
    var kvBytes = kvPerToken * state.context * state.batch;
    var overheadBytes = Math.round((state.overhead || 0) * GIB);
    return {
      weightsBytes: weights,
      kvPerToken: kvPerToken,
      kvBytes: kvBytes,
      overheadBytes: overheadBytes,
      subtotalBytes: weights + kvBytes,
      totalBytes: weights + kvBytes + overheadBytes,
      layers: layers,
      kvHeads: kvHeads,
      headDim: headDim,
      elemBytes: elemBytes
    };
  }

  function gib(bytes) {
    return bytes / GIB;
  }

  function fmtGib(bytes) {
    var v = gib(bytes);
    if (v >= 100) return v.toFixed(1) + " GiB";
    if (v >= 10) return v.toFixed(2) + " GiB";
    return v.toFixed(3) + " GiB";
  }

  function fmtBytes(bytes) {
    return String(bytes).replace(/\B(?=(\d{3})+(?!\d))/g, "\u202f") + " B";
  }

  function reportText(model, state, ledger) {
    var src = model.source || {};
    var lines = [
      "LLM memory ledger (config-grounded, NOT measured VRAM)",
      "model:    " + model.display_name + " [" + state.quant + "] (" + model.id + ")",
      "source:   " + src.repo + " @ " + String(src.revision || "").slice(0, 12),
      "config:   sha256 " + String(src.config_sha256 || "").slice(0, 16),
      "geometry: " + ledger.layers + " layers, " + ledger.kvHeads + " KV heads, head_dim " + ledger.headDim,
      "context:  " + state.context + " tokens x " + state.batch + " sequence(s), KV " + state.kvDtype,
      "weights:  " + fmtGib(ledger.weightsBytes) + " (" + ledger.weightsBytes + " B)",
      "KV cache: " + fmtGib(ledger.kvBytes) + " (" + ledger.kvBytes + " B, " + ledger.kvPerToken + " B/token/seq)",
      "subtotal: " + fmtGib(ledger.subtotalBytes),
      "overhead: " + fmtGib(ledger.overheadBytes) + " (manual bound, not estimated)",
      "TOTAL:    " + fmtGib(ledger.totalBytes) + " - ledger sum, not measured VRAM, not a fit guarantee",
      "limits:   runtime overhead unknown; fp16/bf16 KV only; no MoE/MLA/sliding-window/offload modelling"
    ];
    return lines.join("\n");
  }

  return {
    MAX_CONTEXT: MAX_CONTEXT,
    MAX_BATCH: MAX_BATCH,
    KV_DTYPES: KV_DTYPES,
    parseState: parseState,
    buildQuery: buildQuery,
    computeLedger: computeLedger,
    quantKeys: quantKeys,
    maxContextOf: maxContextOf,
    modelById: modelById,
    fmtGib: fmtGib,
    fmtBytes: fmtBytes,
    gib: gib,
    reportText: reportText
  };
});
