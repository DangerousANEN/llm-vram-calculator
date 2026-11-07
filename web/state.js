/*!
 * URL-state handling for the VRAM calculator.
 *
 * Pure functions only (no DOM), so the same code is unit-tested under node and
 * used by the page. Every incoming value is validated against the pinned
 * catalogue; nothing from the URL is trusted or evaluated.
 *
 * Bounds intentionally mirror vramcalc.core (MAX_CONTEXT_TOKENS, MAX_BATCH,
 * MAX_POSITIONS, MAX_KV_BYTES) so the browser and the CLI accept/reject the
 * same inputs instead of quietly producing different ledgers.
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
  var MAX_POSITIONS = 33554432;
  var MAX_KV_BYTES = 1099511627776; // 1 TiB
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

  function isBlank(raw) {
    return raw === null || raw === undefined || String(raw).trim() === "";
  }

  /** Strict whole-number parse for user input: returns null instead of guessing. */
  function parseWhole(raw) {
    var s = String(raw).trim();
    if (!/^[0-9]+$/.test(s)) return null;
    var n = parseInt(s, 10);
    return Number.isFinite(n) ? n : null;
  }

  function stateError(code, message) {
    var err = new Error(message);
    err.code = code;
    return err;
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

  function geometryOf(model) {
    var cfg = (model && model.config) || {};
    var heads = toInt(cfg.num_attention_heads, 0);
    var kvHeads = toInt(cfg.num_key_value_heads, heads);
    var hidden = toInt(cfg.hidden_size, 0);
    var headDim = toInt(cfg.head_dim, heads && hidden % heads === 0 ? hidden / heads : 0);
    var layers = toInt(cfg.num_hidden_layers, 0);
    return { heads: heads, kvHeads: kvHeads, hidden: hidden, headDim: headDim, layers: layers };
  }

  function kvBytesPerTokenFor(model, kvDtype) {
    var elem = KV_DTYPES[kvDtype];
    if (!elem) return 0;
    var g = geometryOf(model);
    return 2 * g.layers * g.kvHeads * g.headDim * elem;
  }

  /**
   * Validate URL (or form) parameters.
   * @returns {{state: object, notices: string[]}} never throws on bad values;
   *          sanitises them and records a human-readable notice.
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

    var kvRaw = String(params.get("k") || "fp16").toLowerCase();
    var kvDtype = kvRaw;
    if (!Object.prototype.hasOwnProperty.call(KV_DTYPES, kvDtype)) {
      notices.push("KV dtype '" + kvRaw + "' is not modelled (only fp16/bf16); using fp16.");
      kvDtype = "fp16";
    }

    // ---- context ---------------------------------------------------------
    var maxCtx = maxContextOf(model);
    var context = 8192;
    var ctxRaw = params.get("c");
    if (!isBlank(ctxRaw)) {
      var parsedCtx = parseWhole(ctxRaw);
      if (parsedCtx === null) {
        notices.push("Context '" + String(ctxRaw).trim() + "' is not a whole number of tokens; using 8192.");
      } else if (parsedCtx < 1) {
        notices.push("Context must be at least 1 token; using 8192.");
      } else {
        context = parsedCtx;
      }
    }
    if (context > maxCtx) {
      notices.push(
        "Context " + context + " exceeds this config's max_position_embeddings (" + maxCtx + "); clamped to " + maxCtx + "."
      );
      context = maxCtx;
    }

    // ---- batch -----------------------------------------------------------
    var batch = 1;
    var batchRaw = params.get("b");
    if (!isBlank(batchRaw)) {
      var parsedBatch = parseWhole(batchRaw);
      if (parsedBatch === null) {
        notices.push("Batch '" + String(batchRaw).trim() + "' is not a whole number of sequences; using 1.");
      } else if (parsedBatch < 1) {
        notices.push("Batch must be at least 1; using 1.");
      } else {
        batch = parsedBatch;
      }
    }
    if (batch > MAX_BATCH) {
      notices.push("Batch " + batch + " exceeds the supported bound " + MAX_BATCH + "; clamped.");
      batch = MAX_BATCH;
    }

    // ---- supported context*batch bound (parity with vramcalc.core) -------
    if (context * batch > MAX_POSITIONS) {
      var reducedBatch = Math.max(1, Math.floor(MAX_POSITIONS / context));
      batch = reducedBatch < batch ? reducedBatch : 1;
      notices.push(
        "context x batch exceeds the supported bound (" + MAX_POSITIONS + " positions); showing context " +
          context + " x batch " + batch + "."
      );
    }

    // ---- supported KV-cache bound ---------------------------------------
    var kvPerToken = kvBytesPerTokenFor(model, kvDtype);
    if (kvPerToken > 0) {
      var maxSequences = Math.floor(MAX_KV_BYTES / kvPerToken);
      if (maxSequences < 1) {
        context = 1;
        batch = 1;
        notices.push("KV geometry exceeds the 1 TiB supported bound even for one token; showing context 1 x batch 1.");
      } else if (context * batch > maxSequences) {
        batch = Math.max(1, Math.floor(maxSequences / context));
        notices.push("Derived KV cache would exceed the 1 TiB supported bound; reduced batch to " + batch + ".");
      }
    }

    // ---- manual overhead bound ------------------------------------------
    // Absent => unknown (null). Present => an explicit user bound, including 0.
    var overheadRaw = params.get("o");
    var overhead = null;
    if (!isBlank(overheadRaw)) {
      var overheadStr = String(overheadRaw).trim();
      if (!/^[0-9]{1,4}([.][0-9]{1,3})?$/.test(overheadStr)) {
        notices.push(
          "Manual overhead must be a non-negative number of GiB (max " + MAX_OVERHEAD_GIB + "); treated as unknown."
        );
      } else {
        var overheadValue = parseFloat(overheadStr);
        if (overheadValue > MAX_OVERHEAD_GIB) {
          notices.push("Manual overhead " + overheadValue + " GiB exceeds the supported bound " + MAX_OVERHEAD_GIB + " GiB; clamped.");
          overhead = MAX_OVERHEAD_GIB;
        } else {
          overhead = overheadValue;
        }
      }
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
    // null/undefined means "unknown"; an explicit bound (even 0) is preserved.
    if (state.overhead !== null && state.overhead !== undefined && state.overhead !== "") {
      params.set("o", String(state.overhead));
    }
    if (state.kvDtype && state.kvDtype !== "fp16") params.set("k", state.kvDtype);
    return params.toString();
  }

  /** Byte-level ledger. Mirrors vramcalc.core.compute_ledger, including refusals. */
  function computeLedger(model, state) {
    if (!model || !model.config || !model.weights) {
      throw stateError("invalid_model", "model entry is incomplete");
    }
    var quant = String((state && state.quant) || "");
    if (!Object.prototype.hasOwnProperty.call(model.weights, quant) || !model.weights[quant]) {
      throw stateError("unsupported_quant", "quant '" + quant + "' has no official file size for " + model.id);
    }
    var context = state.context;
    if (typeof context !== "number" || !Number.isInteger(context) || context < 1 || context > MAX_CONTEXT) {
      throw stateError("invalid_context", "context must be a whole number of tokens in 1.." + MAX_CONTEXT + ", got " + context);
    }
    var batch = state.batch;
    if (typeof batch !== "number" || !Number.isInteger(batch) || batch < 1 || batch > MAX_BATCH) {
      throw stateError("invalid_batch", "batch must be a whole number in 1.." + MAX_BATCH + ", got " + batch);
    }
    if (context * batch > MAX_POSITIONS) {
      throw stateError(
        "positions_too_large",
        "context x batch = " + context * batch + " exceeds the supported bound (" + MAX_POSITIONS + ")"
      );
    }
    var kvDtype = String(state.kvDtype || "fp16");
    var elemBytes = KV_DTYPES[kvDtype];
    if (!elemBytes) {
      throw stateError("unsupported_kv_dtype", "KV dtype '" + kvDtype + "' is not modelled (only fp16/bf16)");
    }

    var g = geometryOf(model);
    var weights = (model.weights[quant] || {}).bytes;
    if (!Number.isFinite(weights) || weights < 1) {
      throw stateError("unsupported_quant", "quant '" + quant + "' has no usable byte size for " + model.id);
    }
    var kvPerToken = 2 * g.layers * g.kvHeads * g.headDim * elemBytes;
    var kvBytes = kvPerToken * context * batch;
    if (kvBytes > MAX_KV_BYTES) {
      throw stateError("kv_too_large", "derived KV cache " + kvBytes + " bytes exceeds the supported bound (" + MAX_KV_BYTES + ")");
    }

    var overheadUnknown = state.overhead === null || state.overhead === undefined || state.overhead === "";
    var overheadBytes = 0;
    if (!overheadUnknown) {
      var overhead = state.overhead;
      if (typeof overhead !== "number" || !Number.isFinite(overhead) || overhead < 0 || overhead > MAX_OVERHEAD_GIB) {
        throw stateError("invalid_overhead", "manual overhead must be a finite number of GiB in 0.." + MAX_OVERHEAD_GIB + ", got " + overhead);
      }
      overheadBytes = Math.round(overhead * GIB);
    }

    return {
      weightsBytes: weights,
      kvPerToken: kvPerToken,
      kvBytes: kvBytes,
      overheadBytes: overheadBytes,
      overheadUnknown: overheadUnknown,
      positions: context * batch,
      subtotalBytes: weights + kvBytes,
      totalBytes: weights + kvBytes + overheadBytes,
      layers: g.layers,
      kvHeads: g.kvHeads,
      headDim: g.headDim,
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

  function overheadLine(state, ledger) {
    var unknown =
      ledger.overheadUnknown === true ||
      state.overhead === null ||
      state.overhead === undefined ||
      state.overhead === "";
    if (unknown) {
      // deliberately not "0.000 GiB (manual bound)": unknown is not a measured zero
      return "overhead: unknown - not estimated by this tool";
    }
    var suffix = ledger.overheadBytes === 0 ? ", explicitly set to zero" : "";
    return (
      "overhead: " + fmtGib(ledger.overheadBytes) + " (" + ledger.overheadBytes + " B, manual bound" + suffix + ", not estimated)"
    );
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
      "units:    GiB (1 GiB = 1024^3 bytes); exact byte counts in parentheses",
      "weights:  " + fmtGib(ledger.weightsBytes) + " (" + ledger.weightsBytes + " B)",
      "KV cache: " + fmtGib(ledger.kvBytes) + " (" + ledger.kvBytes + " B, " + ledger.kvPerToken + " B/token/seq)",
      "subtotal: " + fmtGib(ledger.subtotalBytes) + " (" + ledger.subtotalBytes + " B, weights + KV)",
      overheadLine(state, ledger),
      "TOTAL:    " + fmtGib(ledger.totalBytes) + " (" + ledger.totalBytes + " B) - ledger sum, not measured VRAM, not a fit guarantee",
      "limits:   runtime overhead unknown; fp16/bf16 KV only; no MoE/MLA/sliding-window/offload modelling"
    ];
    return lines.join("\n");
  }

  return {
    MAX_CONTEXT: MAX_CONTEXT,
    MAX_BATCH: MAX_BATCH,
    MAX_POSITIONS: MAX_POSITIONS,
    MAX_KV_BYTES: MAX_KV_BYTES,
    MAX_OVERHEAD_GIB: MAX_OVERHEAD_GIB,
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
