/* ===== Tapis front-end ===== */
const $ = (s, r = document) => r.querySelector(s);
const grid = $("#grid");
const statusEl = $("#status");
const resultsTitle = $("#results-title");
const resultsEyebrow = $("#results-eyebrow");
let warmed = false;

/* ---------- nav scroll state ---------- */
addEventListener("scroll", () => {
  $("#nav").classList.toggle("scrolled", scrollY > 30);
}, { passive: true });

/* ---------- scroll reveal ---------- */
const revealIO = new IntersectionObserver((entries) => {
  entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); revealIO.unobserve(e.target); } });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((el) => revealIO.observe(el));

/* ---------- hero 3D parallax ---------- */
const stage = $("#stage"), stack = $("#stack");
if (stage) {
  stage.addEventListener("pointermove", (e) => {
    const r = stage.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    stack.style.transform = `rotateY(${x * 18}deg) rotateX(${-y * 16}deg)`;
  });
  stage.addEventListener("pointerleave", () => { stack.style.transform = ""; });
}

/* ---------- filters ---------- */
const LABELS = { color: "Colour", pattern: "Pattern", style: "Style", shape: "Shape" };
let facetCols = [];
async function loadFacets() {
  try {
    const data = await (await fetch("/api/facets")).json();
    const box = $("#filters");
    facetCols = Object.keys(data);
    box.innerHTML = facetCols.map((c) => {
      const opts = ['<option value="Any">Any</option>']
        .concat(data[c].map((v) => `<option value="${v}">${v}</option>`)).join("");
      return `<div class="facet"><label>${LABELS[c] || c}</label><select data-col="${c}">${opts}</select></div>`;
    }).join("");
  } catch (e) { /* facets optional */ }
}
function gatherFilters() {
  const f = {};
  document.querySelectorAll("#filters select").forEach((s) => {
    if (s.value && s.value !== "Any") f[s.dataset.col] = s.value;
  });
  return f;
}

/* ---------- rendering ---------- */
function tagRow(it) {
  return ["color", "pattern", "style", "shape"]
    .filter((k) => it[k]).map((k) => `<span class="tag">${it[k]}</span>`).join("");
}
function cardHTML(it, isResult) {
  const match = isResult && it.match != null
    ? `<div class="match"><b>${it.match}%</b> match</div>` : "";
  const why = isResult && it.why ? `<p class="why">${it.why}</p>` : "";
  return `<article class="card">
    <div class="card-tilt">
      <figure class="card-fig">${match}<img src="${it.image_url}" alt="${it.title}" loading="lazy" /><div class="card-sheen"></div></figure>
      <div class="card-body">
        <h3 class="card-title">${it.title}</h3>
        <div class="tags">${tagRow(it)}</div>
        ${why}
      </div>
    </div></article>`;
}
function attachCard(card, i) {
  card.style.transitionDelay = `${Math.min(i, 12) * 45}ms`;
  revealIO.observe(card);
  const tilt = $(".card-tilt", card);
  const fig = $(".card-fig", card);
  card.addEventListener("pointermove", (e) => {
    const r = card.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
    tilt.style.transform = `rotateY(${(px - 0.5) * 10}deg) rotateX(${(0.5 - py) * 10}deg) translateY(-6px)`;
    fig.style.setProperty("--mx", `${px * 100}%`);
    fig.style.setProperty("--my", `${py * 100}%`);
  });
  card.addEventListener("pointerleave", () => { tilt.style.transform = ""; });
}
function render(items, isResult) {
  grid.innerHTML = items.map((it) => cardHTML(it, isResult)).join("");
  grid.querySelectorAll(".card").forEach(attachCard);
}
function skeletons(n = 8) {
  grid.innerHTML = Array.from({ length: n }, () =>
    `<article class="card in skel"><div class="card-tilt"><div class="card-fig"></div>
     <div class="skel-line"></div><div class="skel-line short"></div></div></article>`).join("");
}
function setStatus(msg, busy) {
  if (!msg) { statusEl.hidden = true; return; }
  statusEl.hidden = false;
  statusEl.innerHTML = (busy ? '<span class="spin"></span>' : "") + msg;
}

/* ---------- search flows ---------- */
async function runText() {
  const q = $("#text-query").value.trim();
  if (!q) { $("#text-query").focus(); return; }
  resultsEyebrow.textContent = "Text search";
  resultsTitle.textContent = `Results for “${q}”`;
  beginSearch();
  try {
    const r = await fetch("/api/search/text", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, top_k: 12, filters: gatherFilters() }),
    });
    finishSearch(await r.json());
  } catch (e) { setStatus("Search failed — is the server running?", false); }
}
let lastFile = null;          // remember the uploaded photo so the slider can re-rank it
function alphaValue() { const s = $("#alpha"); return s ? Number(s.value) / 100 : 0.6; }
async function runImage(file) {
  lastFile = file;
  resultsEyebrow.textContent = "Image search";
  resultsTitle.textContent = "Closest pieces to your photo";
  beginSearch();
  const fd = new FormData();
  fd.append("file", file);
  fd.append("top_k", "12");
  fd.append("filters", JSON.stringify(gatherFilters()));
  fd.append("alpha", String(alphaValue()));
  try {
    const r = await fetch("/api/search/image", { method: "POST", body: fd });
    finishSearch(await r.json());
  } catch (e) { setStatus("Search failed — is the server running?", false); }
}
function beginSearch() {
  document.getElementById("results").scrollIntoView({ behavior: "smooth" });
  skeletons();
  setStatus(warmed ? "Searching the collection…" : "Warming up the visual model — the first search takes a moment…", true);
}
function finishSearch(data) {
  warmed = true;
  const items = (data && data.results) || [];
  if (!items.length) { grid.innerHTML = ""; setStatus(data.message || "No matches — try relaxing the filters.", false); return; }
  setStatus(`${items.length} pieces, ranked by visual similarity.`, false);
  render(items, true);
}

/* ---------- wire up ---------- */
$("#text-btn").addEventListener("click", runText);
$("#text-query").addEventListener("keydown", (e) => { if (e.key === "Enter") runText(); });

/* design <-> colour slider: live readout + re-rank the current photo on release */
const alphaEl = $("#alpha");
if (alphaEl) {
  const sync = () => {
    const d = Number(alphaEl.value);
    $("#mb-design").textContent = `${d}%`;
    $("#mb-color").textContent = `${100 - d}%`;
  };
  alphaEl.addEventListener("input", sync);
  alphaEl.addEventListener("change", () => { if (lastFile) runImage(lastFile); });
  sync();
}

const dz = $("#dropzone"), fileInput = $("#file-input"), preview = $("#query-preview");
function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  preview.src = url; preview.hidden = false;
  runImage(file);
}
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });

/* ---------- initial load ---------- */
(async function init() {
  await loadFacets();
  try {
    const data = await (await fetch("/api/featured?n=12")).json();
    render((data && data.items) || [], false);
  } catch (e) { setStatus("Could not load the collection — is the server running?", false); }
})();
