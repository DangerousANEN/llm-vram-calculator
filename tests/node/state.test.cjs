/**
 * Node unit tests for the URL-state library (no DOM needed).
 * Run: node tests/node/state.test.cjs
 */
"use strict";

const path = require("path");
const assert = require("assert");
const S = require(path.join(__dirname, "..", "..", "web", "state.js"));
const catalog = require(path.join(__dirname, "..", "..", "vramcalc", "data", "models.json"));

let passed = 0;
function test(name, fn) {
  try {
    fn();
    passed += 1;
  } catch (err) {
    console.error("FAIL:", name);
    console.error(err && err.message ? err.message : err);
    process.exit(1);
  }
}

const cat = { models: catalog.models };
const qwen25 = catalog.models[0];
const qwen3 = catalog.models[1];

function ledgerFor(model, over) {
  const state = Object.assign(
    { modelId: model.id, quant: "q4_k_m", context: 8192, batch: 1, overhead: null, kvDtype: "fp16" },
    over || {}
  );
  return { state, led: S.computeLedger(model, state) };
}

test("defaults are applied when the query is empty", () => {
  const { state, notices } = S.parseState("", cat);
  assert.strictEqual(state.modelId, "qwen2.5-7b-instruct");
  assert.strictEqual(state.quant, "bf16");
  assert.strictEqual(state.context, 8192);
  assert.strictEqual(state.batch, 1);
  assert.strictEqual(state.overhead, null);
  assert.deepStrictEqual(notices, []);
});

test("a valid share link round-trips", () => {
  const q = S.buildQuery({ modelId: "qwen3-8b", quant: "q4_k_m", context: 16384, batch: 2, overhead: 1.5, kvDtype: "fp16" });
  const { state, notices } = S.parseState("?" + q, cat);
  assert.strictEqual(state.modelId, "qwen3-8b");
  assert.strictEqual(state.quant, "q4_k_m");
  assert.strictEqual(state.context, 16384);
  assert.strictEqual(state.batch, 2);
  assert.strictEqual(state.overhead, 1.5);
  assert.deepStrictEqual(notices, []);
});

test("fractional context/batch are rejected instead of silently truncated", () => {
  const { state, notices } = S.parseState("?c=1.5&b=2.5", cat);
  assert.strictEqual(state.context, 8192);
  assert.strictEqual(state.batch, 1);
  assert.ok(notices.some((n) => /Context '1\.5' is not a whole number/.test(n)), notices);
  assert.ok(notices.some((n) => /Batch '2\.5' is not a whole number/.test(n)), notices);
});

test("unknown model falls back with a notice", () => {
  const { state, notices } = S.parseState("?m=llama-3-70b", cat);
  assert.strictEqual(state.modelId, "qwen2.5-7b-instruct");
  assert.ok(notices.some((n) => /Unknown model/.test(n)));
});

test("quant not published for the model falls back", () => {
  const { state, notices } = S.parseState("?m=qwen3-8b&q=q2_k", cat);
  assert.strictEqual(state.modelId, "qwen3-8b");
  assert.notStrictEqual(state.quant, "q2_k");
  assert.ok(notices.some((n) => /Unsupported quantisation/.test(n)));
});

test("absurd context is clamped to the model max", () => {
  const { state, notices } = S.parseState("?m=qwen2.5-7b-instruct&c=999999", cat);
  assert.strictEqual(state.context, 32768);
  assert.ok(notices.some((n) => /clamped/.test(n)));
});

test("invalid context numerics are sanitised with a notice, never silently precise", () => {
  for (const bad of ["NaN", "-1", "1.5", "abc"]) {
    const { state, notices } = S.parseState("?m=qwen3-8b&c=" + encodeURIComponent(bad), cat);
    assert.ok(Number.isInteger(state.context) && state.context >= 1, "c=" + bad + " -> " + state.context);
    assert.ok(notices.some((n) => /context/i.test(n)), "missing notice for c=" + bad);
  }
  // an absent value is a default, not an error
  const blank = S.parseState("?m=qwen3-8b&c=", cat);
  assert.strictEqual(blank.state.context, 8192);
  assert.deepStrictEqual(blank.notices, []);
});

test("negative and garbage numbers never produce NaN state", () => {
  const { state, notices } = S.parseState("?c=-5&b=abc&o=-1", cat);
  assert.strictEqual(state.context, 8192);
  assert.strictEqual(state.batch, 1);
  assert.strictEqual(state.overhead, null);
  assert.ok(notices.some((n) => /context/i.test(n)));
  assert.ok(notices.some((n) => /batch/i.test(n)));
  assert.ok(notices.some((n) => /overhead/i.test(n)));
});

test("batch is bounded with a notice", () => {
  // a huge batch is reduced by the supported bounds; a single-token context
  // isolates the MAX_BATCH ceiling itself
  const huge = S.parseState("?b=99999", cat);
  assert.ok(huge.state.batch <= S.MAX_BATCH, huge.state.batch);
  assert.ok(huge.notices.some((n) => /batch|supported bound/i.test(n)), huge.notices);

  const { state, notices } = S.parseState("?c=1&b=99999", cat);
  assert.strictEqual(state.batch, 4096);
  assert.strictEqual(state.context, 1);
  assert.ok(notices.some((n) => /batch/i.test(n)));
});

test("context x batch respects the python MAX_POSITIONS bound", () => {
  assert.strictEqual(S.MAX_POSITIONS, 33554432);
  const { state, notices } = S.parseState("?m=qwen3-8b&c=40960&b=4096", cat);
  assert.ok(state.context * state.batch <= S.MAX_POSITIONS, state.context * state.batch);
  assert.ok(notices.some((n) => /supported bound|positions/i.test(n)), notices);
});

test("derived KV cache is kept under the 1 TiB python bound", () => {
  assert.strictEqual(S.MAX_KV_BYTES, Math.pow(2, 40));
  const { state } = S.parseState("?m=qwen3-8b&c=40960&b=4096", cat);
  const led = S.computeLedger(qwen3, Object.assign({ kvDtype: "fp16", overhead: null }, state));
  assert.ok(led.kvBytes <= S.MAX_KV_BYTES, led.kvBytes);
});

test("unknown kv dtype falls back to fp16", () => {
  const { state, notices } = S.parseState("?k=fp8", cat);
  assert.strictEqual(state.kvDtype, "fp16");
  assert.ok(notices.some((n) => /not modelled/.test(n)));
});

test("omitted overhead stays unknown; explicit zero stays explicit", () => {
  assert.strictEqual(S.parseState("", cat).state.overhead, null);
  assert.strictEqual(ledgerFor(qwen25).led.overheadUnknown, true);

  const zero = S.parseState("?o=0", cat).state;
  assert.strictEqual(zero.overhead, 0);
  assert.strictEqual(S.computeLedger(qwen25, zero).overheadUnknown, false);
});

test("manual overhead beyond the supported bound clamps with a notice", () => {
  const { state, notices } = S.parseState("?o=9000", cat);
  assert.strictEqual(state.overhead, S.MAX_OVERHEAD_GIB);
  assert.ok(notices.some((n) => /overhead/i.test(n)), notices);

  const junk = S.parseState("?o=100000", cat);
  assert.strictEqual(junk.state.overhead, null);
  assert.ok(junk.notices.some((n) => /overhead/i.test(n)));
});

test("computeLedger refuses invalid numerics instead of returning NaN", () => {
  for (const bad of [0, -1, 1.5, NaN, "8192", null]) {
    assert.throws(
      () => S.computeLedger(qwen25, { quant: "q4_k_m", context: bad, batch: 1, kvDtype: "fp16", overhead: null }),
      /context/i
    );
  }
  assert.throws(
    () => S.computeLedger(qwen25, { quant: "q4_k_m", context: 8192, batch: 4097, kvDtype: "fp16", overhead: null }),
    /batch/i
  );
  assert.throws(
    () => S.computeLedger(qwen25, { quant: "q4_k_m", context: 32768, batch: 4096, kvDtype: "fp16", overhead: null }),
    /position|bound/i
  );
  assert.throws(
    () => S.computeLedger(qwen25, { quant: "q4_k_m", context: 8192, batch: 1, kvDtype: "fp8", overhead: null }),
    /kv/i
  );
});

test("ledger bytes match the python implementation", () => {
  const { led } = ledgerFor(qwen25, { quant: "q4_k_m", context: 8192, batch: 1 });
  assert.strictEqual(led.weightsBytes, 4683073632);
  assert.strictEqual(led.kvPerToken, 57344);
  assert.strictEqual(led.kvBytes, 469762048);
  assert.strictEqual(led.totalBytes, 5152835680);

  const { led: led3 } = ledgerFor(qwen3, { quant: "bf16", context: 32768, batch: 2 });
  assert.strictEqual(led3.weightsBytes, 16381516776);
  assert.strictEqual(led3.kvBytes, 147456 * 32768 * 2);
});

test("report text carries the honesty disclaimer", () => {
  const { state, led } = ledgerFor(qwen25, { context: 8192, batch: 1 });
  const text = S.reportText(qwen25, state, led);
  assert.ok(/NOT measured VRAM/.test(text));
  assert.ok(/not a fit guarantee/.test(text));
  assert.ok(text.indexOf(qwen25.source.revision.slice(0, 12)) > -1);
});

test("report text preserves explicit unknown overhead and source units", () => {
  const { state, led } = ledgerFor(qwen25, { context: 8192, batch: 1 });
  const text = S.reportText(qwen25, state, led);
  const line = text.split("\n").filter((l) => /^overhead:/.test(l))[0];
  assert.ok(/unknown/i.test(line), line);
  assert.ok(!/manual bound/i.test(line), line);
  assert.ok(/GiB/.test(text) && /B\)/.test(text), "units and exact bytes must be present");

  const bound = { state: Object.assign({}, state, { overhead: 1.5 }), led: S.computeLedger(qwen25, Object.assign({}, state, { overhead: 1.5 })) };
  const line2 = S.reportText(qwen25, bound.state, bound.led).split("\n").filter((l) => /^overhead:/.test(l))[0];
  assert.ok(/manual bound/i.test(line2), line2);
  assert.ok(/1\.5/.test(line2) || /1\.610/.test(line2), line2);
});

test("formatters keep binary units", () => {
  assert.strictEqual(S.fmtGib(1024 * 1024 * 1024), "1.000 GiB");
  assert.ok(S.fmtBytes(1234567).indexOf("B") > 0);
});

test("empty catalog throws instead of rendering nonsense", () => {
  assert.throws(() => S.parseState("", { models: [] }));
});

console.log("state.test.cjs: OK (" + passed + " tests)");
