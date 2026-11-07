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

test("defaults are applied when the query is empty", () => {
  const { state, notices } = S.parseState("", cat);
  assert.strictEqual(state.modelId, "qwen2.5-7b-instruct");
  assert.strictEqual(state.quant, "bf16");
  assert.strictEqual(state.context, 8192);
  assert.strictEqual(state.batch, 1);
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

test("negative and garbage numbers never produce NaN state", () => {
  const { state } = S.parseState("?c=-5&b=abc&o=-1", cat);
  assert.strictEqual(state.context, 8192);
  assert.strictEqual(state.batch, 1);
  assert.strictEqual(state.overhead, 0);
});

test("batch is bounded", () => {
  const { state } = S.parseState("?b=99999", cat);
  assert.strictEqual(state.batch, 4096);
});

test("unknown kv dtype falls back to fp16", () => {
  const { state, notices } = S.parseState("?k=fp8", cat);
  assert.strictEqual(state.kvDtype, "fp16");
  assert.ok(notices.some((n) => /not modelled/.test(n)));
});

test("ledger bytes match the python implementation", () => {
  const state = { modelId: qwen25.id, quant: "q4_k_m", context: 8192, batch: 1, overhead: 0, kvDtype: "fp16" };
  const led = S.computeLedger(qwen25, state);
  assert.strictEqual(led.weightsBytes, 4683073632);
  assert.strictEqual(led.kvPerToken, 57344);
  assert.strictEqual(led.kvBytes, 469762048);
  assert.strictEqual(led.totalBytes, 5152835680);

  const q3 = { modelId: qwen3.id, quant: "bf16", context: 32768, batch: 2, overhead: 0, kvDtype: "fp16" };
  const led3 = S.computeLedger(qwen3, q3);
  assert.strictEqual(led3.weightsBytes, 16381516776);
  assert.strictEqual(led3.kvBytes, 147456 * 32768 * 2);
});

test("report text carries the honesty disclaimer", () => {
  const state = { modelId: qwen25.id, quant: "q4_k_m", context: 8192, batch: 1, overhead: 0, kvDtype: "fp16" };
  const text = S.reportText(qwen25, state, S.computeLedger(qwen25, state));
  assert.ok(/NOT measured VRAM/.test(text));
  assert.ok(/not a fit guarantee/.test(text));
  assert.ok(text.indexOf(qwen25.source.revision.slice(0, 12)) > -1);
});

test("formatters keep binary units", () => {
  assert.strictEqual(S.fmtGib(1024 * 1024 * 1024), "1.000 GiB");
  assert.ok(S.fmtBytes(1234567).indexOf("B") > 0);
});

test("empty catalog throws instead of rendering nonsense", () => {
  assert.throws(() => S.parseState("", { models: [] }));
});

console.log("state.test.cjs: OK (" + passed + " tests)");
