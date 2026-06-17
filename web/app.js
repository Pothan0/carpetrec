/* ===== Tapis front-end — Awwwards layer + visual search ===== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const FINE = matchMedia("(pointer: fine)").matches;
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

const grid = $("#grid");
const statusEl = $("#status");
const resultsTitle = $("#results-title");
const resultsEyebrow = $("#results-eyebrow");
let warmed = false;

/* ---------- intro: reveal title in sync with the curtain lift (~1.9s) ---------- */
const markLoaded = () => document.body.classList.add("loaded");
setTimeout(markLoaded, REDUCED ? 0 : 1700);
addEventListener("load", () => setTimeout(markLoaded, REDUCED ? 0 : 1700));

/* ---------- Lenis smooth scroll (CDN, graceful fallback to native) ---------- */
let lenis = null;
if (window.Lenis && !REDUCED) {
  lenis = new window.Lenis({ lerp: 0.09, wheelMultiplier: 1, smoothWheel: true });
  // anchor links scroll smoothly through Lenis
  $$('a[href^="#"]').forEach((a) => a.addEventListener("click", (e) => {
    const t = document.querySelector(a.getAttribute("href"));
    if (t) { e.preventDefault(); lenis.scrollTo(t, { offset: -10 }); }
  }));
}

/* ---------- nav scroll state ---------- */
addEventListener("scroll", () => {
  $("#nav").classList.toggle("scrolled", scrollY > 30);
}, { passive: true });

/* ---------- scroll reveal ---------- */
const revealIO = new IntersectionObserver((entries) => {
  entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); revealIO.unobserve(e.target); } });
}, { threshold: 0.12 });
$$(".reveal, .reveal-up").forEach((el) => revealIO.observe(el));

/* ---------- custom cursor (fine pointers, motion ok) ---------- */
const cursorState = { x: innerWidth / 2, y: innerHeight / 2, dx: innerWidth / 2, dy: innerHeight / 2, rx: innerWidth / 2, ry: innerHeight / 2 };
let curDot = null, curRing = null;
if (FINE && !REDUCED) {
  curDot = $("#cursor-dot"); curRing = $("#cursor-ring");
  const ring = curRing;
  document.body.classList.add("cursor-ready", "fine");
  addEventListener("pointermove", (e) => { cursorState.x = e.clientX; cursorState.y = e.clientY; }, { passive: true });
  addEventListener("pointerover", (e) => {
    if (e.target.closest("a,button,label,input,.card,[data-cursor]")) ring.classList.add("hot");
  });
  addEventListener("pointerout", (e) => {
    if (e.target.closest("a,button,label,input,.card,[data-cursor]")) ring.classList.remove("hot");
  });
  addEventListener("pointerdown", () => ring.classList.add("down"));
  addEventListener("pointerup", () => ring.classList.remove("down"));
}

/* ---------- hero 3D parallax (mouse + scroll recede) ---------- */
const stack = $("#stack");
let hero3d = null;                 // set if the WebGL hero initialises (then CSS stack stands down)
const heroMouse = { tx: 0, ty: 0, cx: 0, cy: 0 };
if (!REDUCED) {
  addEventListener("pointermove", (e) => {
    heroMouse.tx = (e.clientX / innerWidth - 0.5);
    heroMouse.ty = (e.clientY / innerHeight - 0.5);
  }, { passive: true });
}

/* Cache layout-triggering reads OUT of the rAF loop (reading scrollHeight every frame
   forces a synchronous reflow — the main source of scroll jitter). */
const scrollProg = $("#scroll-prog");
let maxScroll = Math.max(1, document.documentElement.scrollHeight - innerHeight);
const recomputeMax = () => { maxScroll = Math.max(1, document.documentElement.scrollHeight - innerHeight); };
addEventListener("resize", recomputeMax, { passive: true });
addEventListener("load", recomputeMax);

/* ---------- one rAF loop drives Lenis + cursor + hero + scroll bar ---------- */
function frame(t) {
  if (lenis) lenis.raf(t);
  // cursor
  if (curDot && curRing) {
    cursorState.dx = lerp(cursorState.dx, cursorState.x, 0.35);
    cursorState.dy = lerp(cursorState.dy, cursorState.y, 0.35);
    cursorState.rx = lerp(cursorState.rx, cursorState.x, 0.16);
    cursorState.ry = lerp(cursorState.ry, cursorState.y, 0.16);
    curDot.style.transform = `translate3d(${cursorState.dx}px,${cursorState.dy}px,0) translate(-50%,-50%)`;
    curRing.style.transform = `translate3d(${cursorState.rx}px,${cursorState.ry}px,0) translate(-50%,-50%)`;
  }
  // hero parallax (mouse damping is always needed; the WebGL hero reuses heroMouse.cx/cy)
  if (!REDUCED) {
    heroMouse.cx = lerp(heroMouse.cx, heroMouse.tx, 0.07);
    heroMouse.cy = lerp(heroMouse.cy, heroMouse.ty, 0.07);
  }
  const p = clamp(scrollY / innerHeight, 0, 1);       // 0 at top → 1 once scrolled a screen
  if (stack && !REDUCED && !hero3d) {                  // CSS-3D stack (fallback when no WebGL)
    const ry = heroMouse.cx * 22;
    const rx = -heroMouse.cy * 16 - p * 14;
    stack.style.transform = `rotateX(${rx}deg) rotateY(${ry}deg) translateZ(${-p * 240}px)`;
    stack.style.opacity = String(1 - p * 0.7);
  }
  if (hero3d) renderHero3D(p);                          // legacy WebGL hero (unused)
  if (atelier && scrollY < innerHeight) renderAtelier(t);   // Living Atelier (only while hero is on screen)
  if (constellation && mapOn) renderConstellation();    // spatial similarity map
  // scroll progress (maxScroll is cached → no per-frame reflow)
  if (scrollProg) scrollProg.style.transform = "scaleX(" + clamp(scrollY / maxScroll, 0, 1) + ")";
  tickAmbient();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* ---------- magnetic buttons ---------- */
if (FINE && !REDUCED) {
  $$(".magnetic").forEach((el) => {
    el.style.transition = "transform .35s var(--ease)";
    let r = null;
    el.addEventListener("pointerenter", () => { r = el.getBoundingClientRect(); });
    el.addEventListener("pointermove", (e) => {
      if (!r) r = el.getBoundingClientRect();
      const mx = e.clientX - (r.left + r.width / 2);
      const my = e.clientY - (r.top + r.height / 2);
      el.style.transform = `translate(${mx * 0.3}px, ${my * 0.45}px)`;
    });
    el.addEventListener("pointerleave", () => { el.style.transform = ""; r = null; });
  });
}

/* ---------- WebGL hero (Three.js, optional + graceful) ---------- */
function initHero3D(urls) {
  if (!window.THREE || REDUCED || !urls.length) return;
  const canvas = $("#hero-gl"), stage = $("#stage");
  if (!canvas || !stage) return;
  let renderer;
  // MSAA off + high-performance hint: far smoother on integrated GPUs; textured/fogged
  // billboards barely show aliasing.
  try { renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: "high-performance" }); }
  catch (e) { return; }                               // no WebGL → keep the CSS-3D stack
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  if (THREE.sRGBEncoding !== undefined) renderer.outputEncoding = THREE.sRGBEncoding;

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  cam.position.z = 6;
  const group = new THREE.Group();
  scene.add(group);

  const layout = [
    { x: -1.15, y: 0.25, z: 0.6, r: -0.10, s: 2.1 },
    { x: 1.05, y: -0.30, z: 0.0, r: 0.09, s: 1.9 },
    { x: -0.45, y: -1.05, z: -0.8, r: 0.07, s: 1.7 },
    { x: 0.75, y: 0.95, z: -1.7, r: -0.06, s: 1.5 },
  ];
  const loader = new THREE.TextureLoader();
  const meshes = [];
  urls.slice(0, 4).forEach((url, i) => {
    const L = layout[i];
    const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
    mesh.position.set(L.x, L.y, L.z);
    mesh.rotation.z = L.r;
    mesh.userData = { baseY: L.y, ph: i * 1.7 };
    group.add(mesh); meshes.push(mesh);
    loader.load(url, (tex) => {
      if (THREE.SRGBColorSpace !== undefined) tex.colorSpace = THREE.SRGBColorSpace;
      else if (THREE.sRGBEncoding !== undefined) tex.encoding = THREE.sRGBEncoding;
      const a = (tex.image && tex.image.width / tex.image.height) || 1;
      // fit the LARGER side to L.s so every rug occupies a similar footprint (no distortion)
      const [pw, ph2] = a >= 1 ? [L.s, L.s / a] : [L.s * a, L.s];
      mesh.scale.set(pw, ph2, 1);
      mat.map = tex; mat.opacity = i === 3 ? 0.92 : 1; mat.needsUpdate = true;
    });
  });

  function resize() {
    const r = stage.getBoundingClientRect();
    if (!r.width || !r.height) return;
    renderer.setSize(r.width, r.height, false);
    cam.aspect = r.width / r.height; cam.updateProjectionMatrix();
  }
  resize();
  addEventListener("resize", resize, { passive: true });
  setTimeout(resize, 400);                            // re-fit after fonts/layout settle

  hero3d = { renderer, scene, cam, group, meshes };
  stage.classList.add("gl-on");
}

function renderHero3D(p) {
  const { renderer, scene, cam, group, meshes } = hero3d;
  group.rotation.y += (heroMouse.cx * 0.5 - group.rotation.y) * 0.06;
  group.rotation.x += ((-heroMouse.cy * 0.4 - p * 0.35) - group.rotation.x) * 0.06;
  const time = performance.now() * 0.001;
  for (const m of meshes) m.position.y = m.userData.baseY + Math.sin(time + m.userData.ph) * 0.06;
  cam.position.z = 6 + p * 3;
  renderer.domElement.style.opacity = String(1 - p * 0.7);
  renderer.render(scene, cam);
}

/* ---------- Living Atelier: immersive 3D hero environment (perspective rug cloud + fog) ---------- */
let atelier = null;

async function initAtelier() {
  if (!window.THREE || REDUCED) return;
  const canvas = document.getElementById("atelier-gl");
  if (!canvas) return;
  let layout, atlas;
  try {
    [layout, atlas] = await Promise.all([
      fetch("/static/constellation/layout.json").then((r) => r.json()),
      fetch("/static/constellation/atlas.json").then((r) => r.json()),
    ]);
  } catch (e) { return; }
  if (!layout || !layout.items || !atlas) return;

  let renderer;
  // MSAA off + high-performance hint: far smoother on integrated GPUs; textured/fogged
  // billboards barely show aliasing.
  try { renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: "high-performance" }); }
  catch (e) { return; }
  renderer.setPixelRatio(Math.min(1.25, devicePixelRatio || 1));
  if (THREE.SRGBColorSpace !== undefined) renderer.outputColorSpace = THREE.SRGBColorSpace;
  else if (THREE.sRGBEncoding !== undefined) renderer.outputEncoding = THREE.sRGBEncoding;

  const items = layout.items.filter((it) => !it.eval_only);
  const N = items.length;
  const cellMap = {};
  atlas.items.forEach((a) => { cellMap[a.sku] = a; });

  const aPos = new Float32Array(N * 3), aCell = new Float32Array(N * 2),
        aSize = new Float32Array(N), aPhase = new Float32Array(N);
  items.forEach((it, i) => {
    aPos[i * 3] = it.d[0] * 6.2;          // design X spread
    aPos[i * 3 + 1] = it.d[1] * 4.0;      // design Y spread
    // colour-influenced depth + a unique per-instance ramp so no two rugs are coplanar
    // (kills z-fighting flicker) and the volume reads deeper
    aPos[i * 3 + 2] = it.c[0] * 3.6 + (i / N - 0.5) * 6.0;
    const c = cellMap[it.sku] || { col: 0, row: 0 };
    aCell[i * 2] = c.col; aCell[i * 2 + 1] = c.row;
    aSize[i] = 0.36 + Math.abs(it.c[1]) * 0.14;
    aPhase[i] = i * 0.137;
  });

  const base = new THREE.PlaneGeometry(1, 1);
  const geo = new THREE.InstancedBufferGeometry();
  geo.index = base.index; geo.attributes.position = base.attributes.position; geo.attributes.uv = base.attributes.uv;
  geo.instanceCount = N;
  geo.setAttribute("aPos", new THREE.InstancedBufferAttribute(aPos, 3));
  geo.setAttribute("aCell", new THREE.InstancedBufferAttribute(aCell, 2));
  geo.setAttribute("aSize", new THREE.InstancedBufferAttribute(aSize, 1));
  geo.setAttribute("aPhase", new THREE.InstancedBufferAttribute(aPhase, 1));

  const tex = new THREE.TextureLoader().load("/static/constellation/atlas.png");
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = true;
  if (THREE.SRGBColorSpace !== undefined) tex.colorSpace = THREE.SRGBColorSpace;
  else if (THREE.sRGBEncoding !== undefined) tex.encoding = THREE.sRGBEncoding;
  try { tex.anisotropy = renderer.capabilities.getMaxAnisotropy(); } catch (e) {}

  const uniforms = {
    uAtlas: { value: tex }, uGrid: { value: new THREE.Vector2(atlas.cols, atlas.rows) },
    uTime: { value: 0 }, uFog: { value: new THREE.Color(0x161210) },
    uNear: { value: 5.0 }, uFar: { value: 17.0 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: false, depthTest: true, depthWrite: true,
    vertexShader:
      "attribute vec3 aPos; attribute vec2 aCell; attribute float aSize; attribute float aPhase;" +
      "uniform float uTime;" +
      "varying vec2 vUv; varying vec2 vCell; varying float vDepth;" +
      "void main(){ vUv=uv; vCell=aCell;" +
      "  vec3 p = aPos; p.y += sin(uTime*0.5 + aPhase)*0.13;" +
      "  vec4 mv = modelViewMatrix * vec4(p,1.0);" +
      "  mv.xy += position.xy * aSize;" +                  // billboard toward camera
      "  vDepth = -mv.z;" +
      "  gl_Position = projectionMatrix * mv; }",
    fragmentShader:
      "precision highp float;" +
      "uniform sampler2D uAtlas; uniform vec2 uGrid; uniform vec3 uFog; uniform float uNear; uniform float uFar;" +
      "varying vec2 vUv; varying vec2 vCell; varying float vDepth;" +
      "void main(){" +
      "  vec2 auv = vec2((vCell.x+vUv.x)/uGrid.x, 1.0-(vCell.y+1.0-vUv.y)/uGrid.y);" +
      "  vec4 t = texture2D(uAtlas, auv); if(t.a<0.5) discard;" +          // opaque billboards
      "  float f = clamp((vDepth-uNear)/(uFar-uNear), 0.0, 1.0);" +
      "  gl_FragColor = vec4(mix(t.rgb, uFog, f), 1.0); }",               // dissolve into fog with distance
  });

  const group = new THREE.Group();
  const mesh = new THREE.Mesh(geo, mat); mesh.frustumCulled = false; group.add(mesh);
  const scene = new THREE.Scene(); scene.add(group);
  const cam = new THREE.PerspectiveCamera(55, 1, 0.1, 100); cam.position.set(0, 0, 8.5);

  function resize() {
    const r = canvas.getBoundingClientRect(); if (!r.width || !r.height) return;
    renderer.setSize(r.width, r.height, false);
    cam.aspect = r.width / r.height; cam.updateProjectionMatrix();
  }
  resize(); addEventListener("resize", resize, { passive: true }); setTimeout(resize, 300);

  atelier = { renderer, scene, group, cam, uniforms, mx: 0, my: 0, tmx: 0, tmy: 0, drag: false, lastX: 0 };
  addEventListener("pointermove", (e) => { atelier.tmx = e.clientX / innerWidth - 0.5; atelier.tmy = e.clientY / innerHeight - 0.5; }, { passive: true });
  canvas.addEventListener("pointerdown", (e) => { atelier.drag = true; atelier.lastX = e.clientX; });
  addEventListener("pointerup", () => { if (atelier) atelier.drag = false; });
  addEventListener("pointermove", (e) => { if (atelier && atelier.drag) { atelier.group.rotation.y += (e.clientX - atelier.lastX) * 0.005; atelier.lastX = e.clientX; } });
}

function renderAtelier(t) {
  const a = atelier;
  a.uniforms.uTime.value = t * 0.001;
  if (!a.drag) a.group.rotation.y += 0.0007;          // slow auto-rotate
  a.mx = lerp(a.mx, a.tmx, 0.035); a.my = lerp(a.my, a.tmy, 0.035);   // heavier damping
  a.cam.position.x = a.mx * 1.5; a.cam.position.y = -a.my * 1.05;     // gentler parallax
  a.cam.lookAt(0, 0, 0);
  a.renderer.render(a.scene, a.cam);
}

/* ---------- The Atlas: spatial similarity map (one InstancedMesh + atlas, WebGL) ---------- */
let constellation = null;
let mapOn = false;

async function initConstellation() {
  if (!window.THREE || REDUCED) return;
  const canvas = document.getElementById("atlas-gl");
  if (!canvas) return;
  let layout, atlas;
  try {
    [layout, atlas] = await Promise.all([
      fetch("/static/constellation/layout.json").then((r) => r.json()),
      fetch("/static/constellation/atlas.json").then((r) => r.json()),
    ]);
  } catch (e) { return; }
  if (!layout || !layout.items || !atlas) return;

  let renderer;
  // MSAA off + high-performance hint: far smoother on integrated GPUs; textured/fogged
  // billboards barely show aliasing.
  try { renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: "high-performance" }); }
  catch (e) { return; }
  renderer.setPixelRatio(Math.min(1.25, devicePixelRatio || 1));
  if (THREE.SRGBColorSpace !== undefined) renderer.outputColorSpace = THREE.SRGBColorSpace;
  else if (THREE.sRGBEncoding !== undefined) renderer.outputEncoding = THREE.sRGBEncoding;

  const items = layout.items;
  const N = items.length;
  const cell = {};
  atlas.items.forEach((a) => { cell[a.sku] = a; });
  const skuIndex = new Map();

  const aDesign = new Float32Array(N * 2), aColour = new Float32Array(N * 2), aCell = new Float32Array(N * 2);
  const aState = new Float32Array(N), aMatch = new Float32Array(N), aEval = new Float32Array(N);
  items.forEach((it, i) => {
    aDesign[i * 2] = it.d[0]; aDesign[i * 2 + 1] = it.d[1];
    aColour[i * 2] = it.c[0]; aColour[i * 2 + 1] = it.c[1];
    const c = cell[it.sku] || { col: 0, row: 0 };
    aCell[i * 2] = c.col; aCell[i * 2 + 1] = c.row;
    aEval[i] = it.eval_only ? 1 : 0;
    skuIndex.set(it.sku, i);
  });

  const base = new THREE.PlaneGeometry(1, 1);
  const geo = new THREE.InstancedBufferGeometry();
  geo.index = base.index;
  geo.attributes.position = base.attributes.position;
  geo.attributes.uv = base.attributes.uv;
  geo.instanceCount = N;
  geo.setAttribute("aDesign", new THREE.InstancedBufferAttribute(aDesign, 2));
  geo.setAttribute("aColour", new THREE.InstancedBufferAttribute(aColour, 2));
  geo.setAttribute("aCell", new THREE.InstancedBufferAttribute(aCell, 2));
  const stateAttr = new THREE.InstancedBufferAttribute(aState, 1);
  const matchAttr = new THREE.InstancedBufferAttribute(aMatch, 1);
  geo.setAttribute("aState", stateAttr);
  geo.setAttribute("aMatch", matchAttr);
  geo.setAttribute("aEval", new THREE.InstancedBufferAttribute(aEval, 1));

  const tex = new THREE.TextureLoader().load("/static/constellation/atlas.png");
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = true;
  if (THREE.SRGBColorSpace !== undefined) tex.colorSpace = THREE.SRGBColorSpace;
  else if (THREE.sRGBEncoding !== undefined) tex.encoding = THREE.sRGBEncoding;
  try { tex.anisotropy = renderer.capabilities.getMaxAnisotropy(); } catch (e) {}

  const uniforms = {
    uAtlas: { value: tex },
    uGrid: { value: new THREE.Vector2(atlas.cols, atlas.rows) },
    uMix: { value: alphaValue() },
    uBase: { value: 0.055 },
    uSearch: { value: 0 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthTest: false, depthWrite: false,
    vertexShader:
      "attribute vec2 aDesign; attribute vec2 aColour; attribute vec2 aCell;" +
      "attribute float aState; attribute float aMatch; attribute float aEval;" +
      "uniform float uMix; uniform float uBase;" +
      "varying vec2 vUv; varying vec2 vCell; varying float vState; varying float vEval;" +
      "void main(){ vUv=uv; vCell=aCell; vState=aState; vEval=aEval;" +
      "  vec2 cc = mix(aColour, aDesign, uMix);" +
      "  float size = uBase * (1.0 + aState*0.9 + aMatch*0.5);" +
      "  vec3 pos = vec3(cc + position.xy * size, aState*0.01);" +
      "  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos,1.0); }",
    fragmentShader:
      "precision highp float;" +
      "uniform sampler2D uAtlas; uniform vec2 uGrid; uniform float uSearch;" +
      "varying vec2 vUv; varying vec2 vCell; varying float vState; varying float vEval;" +
      "void main(){" +
      "  vec2 auv = vec2((vCell.x+vUv.x)/uGrid.x, 1.0-(vCell.y+1.0-vUv.y)/uGrid.y);" +
      "  vec4 t = texture2D(uAtlas, auv); if(t.a<0.04) discard;" +
      "  float dim = (uSearch>0.5 && vState<0.5) ? 0.18 : 1.0;" +
      "  float ev = vEval>0.5 ? 0.4 : 1.0;" +
      "  vec3 rgb = t.rgb + vState*0.04;" +
      "  gl_FragColor = vec4(rgb, t.a*dim*ev); }",
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;

  const scene = new THREE.Scene();
  scene.add(mesh);
  const cam = new THREE.OrthographicCamera(-1.2, 1.2, 1.2, -1.2, -10, 10);
  cam.position.z = 2;

  function resize() {
    const r = canvas.getBoundingClientRect();
    if (!r.width || !r.height) return;
    renderer.setSize(r.width, r.height, false);
    const asp = r.width / r.height;
    cam.left = -1.2 * asp; cam.right = 1.2 * asp; cam.top = 1.2; cam.bottom = -1.2;
    cam.updateProjectionMatrix();
  }

  constellation = { renderer, scene, cam, uniforms, items, skuIndex, stateAttr, matchAttr,
                    aState, aMatch, resize, canvas, mixCur: alphaValue(), camTarget: { x: 0, y: 0, z: 1 } };
  resize();
  addEventListener("resize", resize, { passive: true });
  wireConstellationInput(canvas);
}

async function toggleMap() {
  if (!constellation) { await initConstellation(); if (!constellation) return; }   // lazy build on first open
  mapOn = !mapOn;
  document.getElementById("results").classList.toggle("map-on", mapOn);
  const t = document.getElementById("view-toggle");
  if (t) { t.classList.toggle("on", mapOn); t.innerHTML = mapOn ? "&#9638;&nbsp; Grid view" : "&#9716;&nbsp; Map view"; }
  if (mapOn) { document.getElementById("atlas-stage").hidden = false; setTimeout(() => constellation.resize(), 30); }
}

function renderConstellation() {
  const c = constellation;
  c.mixCur = lerp(c.mixCur, alphaValue(), 0.08);
  c.uniforms.uMix.value = c.mixCur;
  if (c.camTarget) {
    c.cam.position.x = lerp(c.cam.position.x, c.camTarget.x, 0.08);
    c.cam.position.y = lerp(c.cam.position.y, c.camTarget.y, 0.08);
    const z = lerp(c.cam.zoom, c.camTarget.z, 0.08);
    if (Math.abs(z - c.cam.zoom) > 1e-4) { c.cam.zoom = z; c.cam.updateProjectionMatrix(); }
  }
  c.renderer.render(c.scene, c.cam);
}

function constellationCoord(i) {
  const c = constellation, it = c.items[i], m = c.mixCur;
  return [it.c[0] * (1 - m) + it.d[0] * m, it.c[1] * (1 - m) + it.d[1] * m];
}

function flyToResults(results) {
  const c = constellation;
  if (!c || !results.length) return;
  c.aState.fill(0); c.aMatch.fill(0);
  let firstIdx = -1;
  results.forEach((it, k) => {
    const i = c.skuIndex.get(it.sku);
    if (i == null) return;
    if (firstIdx < 0) firstIdx = i;
    c.aState[i] = k === 0 ? 2 : 1;
    c.aMatch[i] = (it.match || 0) / 100;
  });
  c.stateAttr.needsUpdate = true; c.matchAttr.needsUpdate = true;
  c.uniforms.uSearch.value = 1;
  if (firstIdx >= 0) { const [x, y] = constellationCoord(firstIdx); c.camTarget = { x, y, z: 4.5 }; }
}

function wireConstellationInput(canvas) {
  const c = () => constellation;
  const label = document.getElementById("atlas-label");
  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; lx = e.clientX; ly = e.clientY; c().camTarget = null;
    try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
  });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointermove", (e) => {
    const cam = c().cam, r = canvas.getBoundingClientRect();
    if (dragging) {
      cam.position.x -= (e.clientX - lx) / r.width * (cam.right - cam.left) / cam.zoom;
      cam.position.y += (e.clientY - ly) / r.height * (cam.top - cam.bottom) / cam.zoom;
      lx = e.clientX; ly = e.clientY;
      if (label) label.hidden = true;
      return;
    }
    const nx = ((e.clientX - r.left) / r.width) * 2 - 1, ny = -(((e.clientY - r.top) / r.height) * 2 - 1);
    const w = new THREE.Vector3(nx, ny, 0).unproject(cam);
    let best = -1, bd = 1e9;
    for (let i = 0; i < c().items.length; i++) {
      if (c().items[i].eval_only) continue;
      const [x, y] = constellationCoord(i);
      const dd = (x - w.x) * (x - w.x) + (y - w.y) * (y - w.y);
      if (dd < bd) { bd = dd; best = i; }
    }
    const th = 0.07 / cam.zoom;
    if (best >= 0 && bd < th * th && label) {
      const it = c().items[best];
      label.hidden = false;
      label.style.left = (e.clientX - r.left) + "px";
      label.style.top = (e.clientY - r.top) + "px";
      const ttl = (it.t || it.sku).replace(/^revival_/, "");
      label.innerHTML = ttl + (it.color ? " &middot; <b>" + it.color + "</b>" : "");
      applyTheme(ambHexes(it));
    } else if (label) { label.hidden = true; }
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const cam = c().cam;
    cam.zoom = clamp(cam.zoom * Math.pow(1.0016, -e.deltaY), 0.6, 14);
    cam.updateProjectionMatrix();
    c().camTarget = null;
  }, { passive: false });
  canvas.addEventListener("pointerenter", () => { if (lenis) lenis.stop(); });
  canvas.addEventListener("pointerleave", () => { if (lenis) lenis.start(); if (label) label.hidden = true; });
}

/* ---------- filters ---------- */
const LABELS = { color: "Colour", pattern: "Pattern", style: "Style", shape: "Shape" };
let facetCols = [];
async function loadFacets() {
  try {
    const data = await (await fetch("/api/facets")).json();
    // Filter dropdowns removed from the UI; we still fetch facets so the colour marquee
    // can pull real palette names. (gatherFilters() now always returns {}.)
    buildMarquee(data);
  } catch (e) { /* facets optional */ }
}
function gatherFilters() {
  const f = {};
  $$("#filters select").forEach((s) => { if (s.value && s.value !== "Any") f[s.dataset.col] = s.value; });
  return f;
}

/* ---------- ambient theming + marquee-as-control (colour is a search axis) ---------- */
const HUE = { black: "#2a2622", white: "#efe9dd", grey: "#9a9388", grayscale: "#9a9388", beige: "#cdb38c",
  cream: "#e6dcc4", sand: "#cdb38c", tan: "#bda079", neutral: "#bcae98", red: "#a8442f", brown: "#7c5436",
  orange: "#c47a3a", yellow: "#caa23c", green: "#6f7a4e", blue: "#4a6276", purple: "#6f5a78", pink: "#c08a86" };
const DESIGN_WORDS = ["Medallion", "Floral", "Geometric", "Tribal", "Moroccan", "Vintage", "Lattice", "Bordered", "Abstract", "Diamond"];

const NEUTRAL = new Set(["grey", "grayscale", "white", "black", "beige", "cream", "sand", "tan", "neutral"]);
function ambHexes(it) {
  if (!it) return null;
  const names = (it.palette ? String(it.palette).split(",") : [it.color || ""])
    .map((s) => s.trim().toLowerCase()).filter(Boolean)
    .sort((a, b) => (NEUTRAL.has(a) ? 1 : 0) - (NEUTRAL.has(b) ? 1 : 0));   // chromatic colours lead → coloured glow even on neutral rugs
  const hex = names.map((n) => HUE[n]).filter(Boolean);
  while (hex.length && hex.length < 3) hex.push(hex[hex.length - 1]);
  return hex.length >= 3 ? hex.slice(0, 3) : null;
}
const amb = {
  cur: [[168, 118, 62], [200, 150, 90], [120, 90, 60]],
  tgt: [[168, 118, 62], [200, 150, 90], [120, 90, 60]],
  sCur: 0.5, sTgt: 0.5,
};
function hexToRgb(h) {
  const m = /^#?([0-9a-f]{6})$/i.exec(h || "");
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function applyTheme(hexes) {
  amb.sTgt = 0.5 + (1 - alphaValue()) * 0.5;          // slider gates intensity (floor kept visible)
  const rgb = (hexes || []).map(hexToRgb).filter(Boolean);
  if (rgb.length >= 3) amb.tgt = rgb.slice(0, 3);
}
function tickAmbient() {                               // lerped in frame(); only writes while changing
  const settled = Math.abs(amb.sCur - amb.sTgt) < 0.004 &&
    amb.cur.every((c, i) => Math.abs(c[0] - amb.tgt[i][0]) < 0.6 &&
      Math.abs(c[1] - amb.tgt[i][1]) < 0.6 && Math.abs(c[2] - amb.tgt[i][2]) < 0.6);
  if (settled) return;
  amb.sCur = lerp(amb.sCur, amb.sTgt, 0.06);
  for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) amb.cur[i][k] = lerp(amb.cur[i][k], amb.tgt[i][k], 0.05);
  const a = amb.sCur;
  const css = (i, al) => `rgba(${Math.round(amb.cur[i][0])},${Math.round(amb.cur[i][1])},${Math.round(amb.cur[i][2])},${(al * a).toFixed(3)})`;
  const rs = document.documentElement.style;
  rs.setProperty("--amb-a", css(0, 1.0));
  rs.setProperty("--amb-b", css(1, 0.9));
  rs.setProperty("--amb-c", css(2, 0.82));
}
function fillMarquee(track, words, tinted) {
  if (!track || !words || !words.length) return;
  const one = words.map((w) => {
    const c = tinted && HUE[w.toLowerCase()] ? ` style="--c:${HUE[w.toLowerCase()]}"` : "";
    return `<span class="mq-word" data-cursor="link" data-q="${w}"${c}>${w}</span><b>·</b>`;
  }).join("");
  track.innerHTML = one + one;   // doubled for the seamless translateX(-50%) loop
  $$(".mq-word", track).forEach((el) => el.addEventListener("click", () => {
    const inp = $("#text-query"); if (inp) inp.value = el.dataset.q; runText();
  }));
}
function buildMarquee(facets) {
  fillMarquee($("#mq-design"), DESIGN_WORDS, false);
  fillMarquee($("#mq-color"), (facets && facets.color ? facets.color : []).slice(0, 12), true);
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
  const cfTag = it.color ? `<span class="cf-tag">${it.color}</span>` : "";
  return `<article class="card" data-cursor="view" data-amb="${(ambHexes(it) || []).join(",")}">
    <div class="card-tilt">
      <figure class="card-fig">${match}
        <img class="cf-base" src="${it.image_url}" alt="${it.title}" loading="lazy" />
        <img class="cf-color" src="${it.image_url}" alt="" aria-hidden="true" loading="lazy" />
        <span class="cf-seam"></span>${cfTag}
        <div class="card-sheen"></div>
      </figure>
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
  if (REDUCED) return;
  const tilt = $(".card-tilt", card);
  const fig = $(".card-fig", card);
  let cr = null;
  card.addEventListener("pointerenter", () => {
    cr = card.getBoundingClientRect();
    applyTheme(card.dataset.amb ? card.dataset.amb.split(",") : null);   // room takes on this rug
  });
  card.addEventListener("pointermove", (e) => {
    if (!cr) cr = card.getBoundingClientRect();
    const px = (e.clientX - cr.left) / cr.width, py = (e.clientY - cr.top) / cr.height;
    tilt.style.transform = `rotateY(${(px - 0.5) * 10}deg) rotateX(${(0.5 - py) * 10}deg) translateY(-6px)`;
    fig.style.setProperty("--mx", `${px * 100}%`);
    fig.style.setProperty("--my", `${py * 100}%`);
    // Split-Reveal: left of the cursor = grayscale (design axis), right = colour (colour axis)
    fig.style.setProperty("--wipe", `${clamp(px, 0, 1) * 100}%`);
  });
  card.addEventListener("pointerleave", () => { tilt.style.transform = ""; fig.style.setProperty("--wipe", "0%"); cr = null; });
}
function render(items, isResult) {
  grid.innerHTML = items.map((it) => cardHTML(it, isResult)).join("");
  // results animate in immediately (they appear below the fold after a search)
  $$(".card", grid).forEach((card, i) => { attachCard(card, i); if (isResult) requestAnimationFrame(() => card.classList.add("in")); });
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
  lastQuery = { type: "text", word: q };
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
let lastFile = null;
let lastQuery = {};
let skipMorphOnce = false;   // slider re-runs re-rank instantly without replaying the morph
function alphaValue() { const s = $("#alpha"); return s ? Number(s.value) / 100 : 0.8; }
async function runImage(file) {
  lastFile = file;
  lastQuery = { type: "image", src: preview.src };
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
  const results = document.getElementById("results");
  if (lenis) lenis.scrollTo(results, { offset: -10 }); else results.scrollIntoView({ behavior: "smooth" });
  skeletons();
  setStatus(warmed ? "Searching the collection…" : "Warming up the visual model — the first search takes a moment…", true);
}
/* ---------- The Living Search: query morphs into its #1 match ---------- */
function countUp(el, target, dur) {
  const start = performance.now();
  (function tick(now) {
    const t = clamp((now - start) / dur, 0, 1);
    el.textContent = Math.round(target * (1 - Math.pow(1 - t, 3)));   // ease-out cubic
    if (t < 1) requestAnimationFrame(tick);
  })(performance.now());
}
function playMorph(opts, onDone) {
  const overlay = $("#morph");
  let done = false;
  const callDone = () => { if (done) return; done = true; onDone(); };
  if (REDUCED || !overlay || !opts.matchSrc) return callDone();      // graceful: straight to grid
  const q = $("#morph-q"), wordEl = $("#morph-word"), m = $("#morph-m"),
        pctEl = $("#morph-pct"), axisEl = $("#morph-axis");
  const design = (opts.alpha == null ? 0.8 : opts.alpha) >= 0.5;
  let started = false;
  const cleanup = () => {
    q.className = "morph-layer"; q.removeAttribute("src"); q.style.display = "";
    m.classList.remove("reveal"); m.removeAttribute("src");
    wordEl.removeAttribute("style"); wordEl.textContent = "";
  };
  const pre = new Image();
  pre.onerror = () => { if (!started) callDone(); };               // can't load match → skip morph
  pre.onload = () => {
    if (done) return;                                              // safety net already fired
    started = true;
    m.src = opts.matchSrc; m.classList.remove("reveal");
    if (opts.imageSrc) { q.src = opts.imageSrc; q.style.display = ""; wordEl.style.display = "none"; }
    else { q.style.display = "none"; wordEl.style.display = ""; wordEl.textContent = opts.word || "your search"; }
    pctEl.textContent = "0";
    if (axisEl) axisEl.textContent = design ? "· by design" : "· by colour";
    overlay.classList.add("on");
    requestAnimationFrame(() => {                                  // phase A: dissolve to the engine's "ghost"
      if (opts.imageSrc) q.classList.add(design ? "ghost-design" : "ghost-colour");
      else { wordEl.style.filter = design ? "grayscale(1)" : "blur(10px)"; wordEl.style.opacity = ".5"; }
    });
    setTimeout(() => { m.classList.add("reveal"); countUp(pctEl, opts.pct || 0, 850); }, 480); // phase B: resolve to the match
    setTimeout(() => { overlay.classList.remove("on"); setTimeout(() => { cleanup(); callDone(); }, 480); }, 1550);
  };
  pre.src = opts.matchSrc;
  setTimeout(() => { if (!started) callDone(); }, 1500);            // network safety net
}

function updateRectifyPanel(data) {
  const panel = $("#rectify-panel");
  if (!panel) return;
  if (!data || lastQuery.type !== "image") { panel.hidden = true; return; }
  const meta = data.meta || {};
  $("#rx-before").src = data.query_preview || lastQuery.src || "";
  const afterFig = $("#rx-after-fig"), note = $("#rx-note");
  if (data.rectified_preview) {
    $("#rx-after").src = data.rectified_preview;
    $("#rx-after-cap").textContent = meta.wh_ratio ? `Top-down · ${meta.wh_ratio} : 1` : "Top-down";
    afterFig.hidden = false; note.hidden = true;
  } else {
    afterFig.hidden = true;
    note.textContent = meta.mask_found === false ? "Searched as uploaded" : "Already top-down — searched as-is";
    note.hidden = false;
  }
  panel.hidden = false;
}

function finishSearch(data) {
  warmed = true;
  updateRectifyPanel(data);
  const items = (data && data.results) || [];
  if (!items.length) { grid.innerHTML = ""; setStatus(data.message || "No matches — try relaxing the filters.", false); recomputeMax(); return; }
  const top = items[0];
  const reveal = () => {
    setStatus(`${items.length} pieces, ranked by visual similarity.`, false);
    render(items, true);
    applyTheme(ambHexes(top));     // ambient world takes on the #1 match
    recomputeMax();
  };
  if (mapOn && constellation) { reveal(); flyToResults(items); return; }   // Map view: the camera-descent is the reveal
  if (skipMorphOnce) { skipMorphOnce = false; reveal(); return; }   // slider re-rank: skip the morph
  try {
    playMorph({
      imageSrc: lastQuery.type === "image" ? lastQuery.src : null,
      word: lastQuery.word,
      matchSrc: top.image_url,
      pct: top.match,
      alpha: alphaValue(),
    }, reveal);
  } catch (e) { reveal(); }
}

/* ---------- wire up ---------- */
$("#text-btn").addEventListener("click", runText);
$("#text-query").addEventListener("keydown", (e) => { if (e.key === "Enter") runText(); });

const alphaEl = $("#alpha");
if (alphaEl) {
  const sync = () => {
    const d = Number(alphaEl.value);
    $("#mb-design").textContent = `${d}%`;
    $("#mb-color").textContent = `${100 - d}%`;
    // headings physically read the slider: design-heavy = sharper/bolder/tighter, colour = softer/wider
    const rs = document.documentElement.style;
    rs.setProperty("--wght", Math.round(400 + (d / 100) * 200));
    rs.setProperty("--opsz", Math.round(72 + (d / 100) * 72));
    rs.setProperty("--head-ls", `${(((100 - d) / 100) * 0.03).toFixed(3)}em`);
    amb.sTgt = 0.5 + ((100 - d) / 100) * 0.5;   // ambient intensity tracks the slider live
  };
  alphaEl.addEventListener("input", sync);
  // Slider only changes the design/colour blend — re-fuse the cached query instantly instead
  // of re-running SAM3 + DINO + colour (which a full image re-search would do).
  alphaEl.addEventListener("change", () => { if (lastQuery.type === "image") rerankAlpha(); });
  sync();
}

async function rerankAlpha() {
  setStatus("Re-ranking…", true);
  try {
    const r = await fetch("/api/rerank", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alpha: alphaValue(), top_k: 12, filters: gatherFilters() }),
    });
    const data = await r.json();
    const items = (data && data.results) || [];
    if (!items.length) {                                   // no cached query -> full re-search
      if (lastFile) { skipMorphOnce = true; runImage(lastFile); }
      return;
    }
    if (mapOn && constellation) { render(items, true); applyTheme(ambHexes(items[0])); flyToResults(items); }
    else { render(items, true); applyTheme(ambHexes(items[0])); }
    recomputeMax();
    setStatus(`${items.length} pieces, re-weighted ${$("#mb-design").textContent} design.`, false);
  } catch (e) {
    if (lastFile) { skipMorphOnce = true; runImage(lastFile); }
  }
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
    const items = (data && data.items) || [];
    render(items, false);
    recomputeMax();
    // seed the load with a COLOURFUL piece so the tint is obvious from the start
    const seed = items.find((it) => !NEUTRAL.has(String(it.color || "").toLowerCase())) || items[0];
    if (seed) applyTheme(ambHexes(seed));
    // Living Atelier — the immersive 3D hero environment (replaces the old floating stack)
    initAtelier();
    // Map view (Atlas) builds lazily on first toggle; just reveal + wire the button
    const tg = document.getElementById("view-toggle");
    if (tg && window.THREE && !REDUCED) { tg.hidden = false; tg.addEventListener("click", toggleMap); }
  } catch (e) { setStatus("Could not load the collection — is the server running?", false); }
})();
