const form = document.querySelector("#extract-form");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const cards = document.querySelector("#cards");
const themeToggle = document.querySelector("#theme-toggle");

const setTheme = (theme) => {
  document.documentElement.dataset.theme = theme;
  themeToggle.setAttribute("aria-pressed", String(theme === "dark"));
  themeToggle.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
};

setTheme(document.documentElement.dataset.theme || "light");

themeToggle.addEventListener("click", () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(theme);
  try {
    localStorage.setItem("theme", theme);
  } catch {
    // The selected theme still applies for this session when storage is unavailable.
  }
});

const escapeHtml = (text) => String(text ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
})[char]);

const readJsonResponse = async (response) => {
  const rawBody = await response.text();
  let body;
  try {
    body = rawBody ? JSON.parse(rawBody) : null;
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail
      : Array.isArray(body?.detail) ? body.detail.map((item) => item.msg).join("; ")
      : response.statusText || "Request failed";
    throw new Error(`Extraction failed (HTTP ${response.status}): ${detail}${rawBody ? "" : " — empty response; check the server logs or tunnel."}`);
  }
  if (!body) throw new Error("Extraction returned an empty or invalid response. Check the server logs or tunnel.");
  return body;
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selectedModel = form.elements.model.selectedOptions[0].textContent.split(" · ")[0];
  status.textContent = form.elements.force_ocr.checked
    ? `Rendering and OCRing every page, then extracting with ${selectedModel}…`
    : `Extracting document fields with ${selectedModel}…`;
  results.hidden = true;
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const submitted = await readJsonResponse(await fetch("/api/extract/jobs", {
      method: "POST", body: new FormData(form)
    }));
    if (!submitted.job_id) throw new Error("Extraction job was not created.");
    const started = Date.now();
    let body;
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const job = await readJsonResponse(await fetch(`/api/extract/jobs/${encodeURIComponent(submitted.job_id)}`));
      if (job.status === "completed") {
        body = job.result;
        break;
      }
      if (job.status === "failed") throw new Error(job.detail || "Extraction failed. Check the server logs.");
      if (job.status !== "running") throw new Error("Extraction returned an unknown job status.");
      status.textContent = `${selectedModel} is extracting… ${Math.floor((Date.now() - started) / 1000)}s elapsed`;
    }
    if (!body || !body.metrics || !Array.isArray(body.results)) {
      throw new Error("Extraction returned an empty or invalid response. Check the server logs or tunnel.");
    }
    const metrics = body.metrics;
    document.querySelector("#meta").textContent = `${metrics.fields_extracted}/${metrics.fields_requested} fields · ${body.pages_processed} page${body.pages_processed === 1 ? "" : "s"} · ${metrics.total_seconds.toFixed(2)}s total · ${metrics.inference_seconds.toFixed(2)}s inference · ${body.model}`;
    cards.innerHTML = body.results.map((item) => {
      const confidence = !item.value ? "No matching source quote" : item.confidence == null ? "Quoted from source" : `${Math.round(item.confidence * 100)}% confidence`;
      return `<article class="card">
        <h3>${escapeHtml(item.field)}</h3>
        <p class="value">${item.value ? escapeHtml(item.value) : "Not found"}</p>
        <p class="confidence">${confidence}${item.page ? ` · page ${item.page}` : ""}</p>
        ${item.evidence ? `<blockquote>${escapeHtml(item.evidence)}</blockquote>` : ""}
      </article>`;
    }).join("");
    results.hidden = false;
    status.textContent = "";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

const v2Form = document.querySelector("#v2-form");
if (v2Form) {
  const inputMode = document.querySelector("#v2-input-mode");
  const templateInput = document.querySelector("#v2-template");
  const fieldsInput = document.querySelector("#v2-fields");
  inputMode.addEventListener("change", () => {
    const useTemplate = inputMode.value === "template";
    templateInput.hidden = !useTemplate;
    templateInput.disabled = !useTemplate;
    templateInput.required = useTemplate;
    fieldsInput.hidden = useTemplate;
    fieldsInput.disabled = useTemplate;
    fieldsInput.required = !useTemplate;
    document.querySelector("#v2-template-label").hidden = !useTemplate;
    document.querySelector("#v2-fields-label").hidden = useTemplate;
  });

  v2Form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = v2Form.querySelector("button");
    const v2Status = document.querySelector("#v2-status");
    const v2Results = document.querySelector("#v2-results");
    const model = v2Form.elements.model.value;
    button.disabled = true;
    v2Results.hidden = true;
    v2Status.textContent = model === "numind/NuExtract3"
      ? "NuExtract3 is starting locally; this CPU run may take more than two minutes…"
      : "Inspecting pages and extracting structured data locally…";
    try {
      const submitted = await readJsonResponse(await fetch("/api/v2/extract/jobs", {
        method: "POST", body: new FormData(v2Form)
      }));
      const started = Date.now();
      let body;
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        const job = await readJsonResponse(await fetch(`/api/v2/extract/jobs/${encodeURIComponent(submitted.job_id)}`));
        if (job.status === "completed" || job.status === "incomplete") {
          body = job.result;
          break;
        }
        if (job.status === "failed") throw new Error(job.detail || "Extraction failed.");
        if (job.status !== "running") throw new Error("Extraction returned an unknown job status.");
        v2Status.textContent = `${model} is extracting… ${Math.floor((Date.now() - started) / 1000)}s elapsed`;
      }
      document.querySelector("#v2-data").textContent = JSON.stringify(body.data, null, 2);
      document.querySelector("#v2-meta").textContent = `${body.pages_processed} pages · ${body.metrics.total_seconds.toFixed(2)}s · ${body.metrics.model_calls} model calls · ${body.model}`;
      const entries = Object.entries(body.evidence);
      document.querySelector("#v2-evidence").innerHTML = entries.length
        ? entries.map(([path, source]) => `<p><strong>${escapeHtml(path)}</strong> · page ${escapeHtml(source.page)} · “${escapeHtml(source.quote)}”</p>`).join("")
        : "<p>No supported values found.</p>";
      v2Results.hidden = false;
      v2Status.textContent = body.warnings.length ? body.warnings.join(" ") : "";
    } catch (error) {
      v2Status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
}
