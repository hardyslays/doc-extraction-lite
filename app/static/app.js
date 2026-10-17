const form = document.querySelector("#extract-form");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const cards = document.querySelector("#cards");

const escapeHtml = (text) => String(text ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
})[char]);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  status.textContent = "Extracting page by page with the local model…";
  results.hidden = true;
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const response = await fetch("/api/extract", { method: "POST", body: new FormData(form) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Extraction failed.");
    document.querySelector("#meta").textContent = `${body.pages_processed} page${body.pages_processed === 1 ? "" : "s"} · ${body.model}`;
    cards.innerHTML = body.results.map((item) => {
      const confidence = `${Math.round(item.confidence * 100)}% confidence`;
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
