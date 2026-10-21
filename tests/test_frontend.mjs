import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const appScript = readFileSync(new URL("../app/static/app.js", import.meta.url), "utf8");

async function submitWith(responses) {
  let submit;
  let fetchCalls = 0;
  const button = { disabled: false };
  const status = { textContent: "" };
  const results = { hidden: false };
  const cards = { innerHTML: "" };
  const form = {
    elements: {
      model: { selectedOptions: [{ textContent: "NuExtract3 · 4B-class, slower" }] },
      force_ocr: { checked: false },
    },
    querySelector: () => button,
    addEventListener: (name, handler) => { if (name === "submit") submit = handler; },
  };
  const elements = {
    "#extract-form": form,
    "#status": status,
    "#results": results,
    "#cards": cards,
    "#meta": { textContent: "" },
    "#theme-toggle": { setAttribute() {}, addEventListener() {} },
  };
  const document = {
    documentElement: { dataset: { theme: "light" } },
    querySelector: (selector) => elements[selector],
  };
  runInNewContext(appScript, {
    document,
    fetch: async () => responses[fetchCalls++],
    FormData: class {},
    setTimeout: (callback) => callback(),
  });
  await submit({ preventDefault() {} });
  return {
    status: status.textContent,
    buttonDisabled: button.disabled,
    resultsHidden: results.hidden,
    cardsHTML: cards.innerHTML,
    fetchCalls,
  };
}

test("empty gateway response reports HTTP status instead of JSON.parse error", async () => {
  const result = await submitWith([{
    ok: false,
    status: 502,
    statusText: "Bad Gateway",
    text: async () => "",
  }]);
  assert.match(result.status, /HTTP 502.*empty response/);
  assert.equal(result.buttonDisabled, false);
});

test("non-JSON success response reports invalid response", async () => {
  const result = await submitWith([{
    ok: true,
    status: 200,
    statusText: "OK",
    text: async () => "<html>tunnel error</html>",
  }]);
  assert.match(result.status, /empty or invalid response/);
});

test("UI polls until the background job completes", async () => {
  const jsonResponse = (body, status = 200) => ({
    ok: true, status, statusText: "OK", text: async () => JSON.stringify(body),
  });
  const result = await submitWith([
    jsonResponse({ job_id: "job-1", status: "running" }, 202),
    jsonResponse({ status: "running" }),
    jsonResponse({ status: "completed", result: {
      model: "numind/NuExtract3", pages_processed: 1,
      metrics: { fields_extracted: 1, fields_requested: 1, total_seconds: 20, inference_seconds: 19 },
      results: [{ field: "Invoice number", value: "INV-4421", confidence: null, page: 1, evidence: "INV-4421" }],
    } }),
  ]);
  assert.equal(result.fetchCalls, 3);
  assert.equal(result.resultsHidden, false);
  assert.match(result.cardsHTML, /INV-4421/);
  assert.equal(result.status, "");
});
