'use strict';

// JD facets metadata (mirrors common.py)
const FACETS = [
  { key:'production_embeddings_retrieval', label:'Embeddings', must:true  },
  { key:'vector_db_hybrid_search',         label:'Vector DB',  must:true  },
  { key:'ranking_eval_frameworks',         label:'Ranking Eval', must:true },
  { key:'strong_python_production_code',   label:'Python',     must:true  },
  { key:'llm_finetuning',                  label:'LLM Finetune', must:false },
  { key:'learning_to_rank',                label:'L2R',        must:false },
  { key:'hrtech_marketplace_background',   label:'HR-tech',    must:false },
  { key:'distributed_systems_scale',       label:'Distributed', must:false },
  { key:'open_source_external_validation', label:'Open Source', must:false },
];

let allRows = [], filteredRows = [];
let sortCol = 'rank', sortDir = 1;
let pendingCsv = null, selectedFile = null;
let focusedIdx = -1;

const $ = id => document.getElementById(id);

const fileInput     = $('file-input');
const browseBtn     = $('browse-btn');
const clearBtn      = $('clear-btn');
const uploadZone    = $('upload-zone');
const uploadIdle    = $('upload-idle');
const uploadChosen  = $('upload-chosen');
const uploadFilename= $('upload-filename');
const progressWrap  = $('progress-wrap');
const progressFill  = $('progress-fill');
const progressText  = $('progress-text');
const topN          = $('top-n');
const presets       = document.querySelectorAll('.preset');
const btnRank       = $('btn-rank');
const btnDownload   = $('btn-download');
const search        = $('search');
const funnelEl      = $('funnel');
const fTotal        = $('f-total');
const fExcluded     = $('f-excluded');
const fRanked       = $('f-ranked');
const resultsTitle  = $('results-title');
const tableWrap     = $('table-wrap');
const tbody         = $('tbody');
const stateEmpty    = $('state-empty');
const stateLoading  = $('state-loading');
const stateError    = $('state-error');
const loadingText   = $('loading-text');
const errorText     = $('error-text');
const chips         = document.querySelectorAll('.chip');
const overlay       = $('overlay');
const drawer        = $('drawer');
const drawerClose   = $('drawer-close');
const headerHint    = $('header-hint');
const kbdHint       = $('kbd-hint');

// ---- File handling ----
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) pick(fileInput.files[0]); });

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.jsonl')) pick(f);
});

function pick(f) {
  selectedFile = f;
  uploadIdle.style.display   = 'none';
  uploadChosen.style.display = 'flex';
  uploadFilename.textContent  = f.name;
  btnRank.disabled = false;
}

clearBtn.addEventListener('click', e => {
  e.stopPropagation(); selectedFile = null; fileInput.value = '';
  uploadIdle.style.display   = 'flex';
  uploadChosen.style.display = 'none';
  btnRank.disabled = true;
});

// ---- Presets ----
presets.forEach(b => {
  b.addEventListener('click', () => {
    topN.value = b.dataset.n;
    presets.forEach(x => x.classList.remove('active'));
    b.classList.add('active');
  });
});
topN.addEventListener('input', () => {
  presets.forEach(b => b.classList.toggle('active', b.dataset.n === topN.value));
});

// ---- Run ----
btnRank.addEventListener('click', () => { if (selectedFile) doRank(selectedFile, parseInt(topN.value)||100); });

function doRank(file, n) {
  progressWrap.style.display = 'block';
  progressFill.style.width   = '0%';
  progressText.textContent   = 'Uploading...';
  setState('loading'); loadingText.textContent = 'Uploading file...';

  const form = new FormData();
  form.append('file', file);
  form.append('top_n', n);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/rank');

  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      const p = Math.round((e.loaded/e.total)*100);
      progressFill.style.width = p+'%';
      progressText.textContent = `Uploading ${p}%`;
    }
  });

  xhr.upload.addEventListener('load', () => {
    progressWrap.style.display = 'none';
    setState('loading');
    loadingText.textContent = `Ranking ${n.toLocaleString()} candidates...`;
  });

  xhr.addEventListener('load', () => {
    if (xhr.status === 200) {
      try { const d = JSON.parse(xhr.responseText); pendingCsv = d.csv; render(d); }
      catch { setErr('Unexpected server response.'); }
    } else {
      try { setErr(JSON.parse(xhr.responseText).detail||'Ranking failed.'); }
      catch { setErr(`Server error (${xhr.status}).`); }
    }
  });

  xhr.addEventListener('error', () => setErr('Network error. Is the server running?'));
  xhr.send(form);
}

// ---- Download ----
btnDownload.addEventListener('click', () => {
  if (!pendingCsv) return;
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([pendingCsv],{type:'text/csv'})),
    download: 'ranked_candidates.csv'
  });
  a.click();
});

// ---- Render ----
function render(data) {
  allRows = data.rows || [];
  animateCount(fTotal,    data.total_input);
  animateCount(fExcluded, data.honeypots_excluded ?? 0);
  animateCount(fRanked,   data.count);
  funnelEl.style.display  = 'flex';
  btnDownload.disabled    = false;
  resultsTitle.textContent = `Top ${data.count} candidates`;
  headerHint.textContent  = `${data.total_input.toLocaleString()} processed`;
  kbdHint.style.display   = 'block';
  applySort();
  setState('table');
}

function applySort() {
  const q = search.value.trim().toLowerCase();
  filteredRows = q
    ? allRows.filter(r =>
        [r.candidate_id, r.name, r.title, r.location, r.reasoning]
          .some(v => (v||'').toLowerCase().includes(q)))
    : [...allRows];

  filteredRows.sort((a,b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (sortCol==='rank'||sortCol==='score') { va=parseFloat(va)||0; vb=parseFloat(vb)||0; }
    else { va=String(va||'').toLowerCase(); vb=String(vb||'').toLowerCase(); }
    return va<vb ? -sortDir : va>vb ? sortDir : 0;
  });

  buildTable();
}

function buildTable() {
  tbody.innerHTML = '';
  const maxScore = filteredRows.length ? parseFloat(filteredRows[0].score) : 1;

  filteredRows.forEach((row, i) => {
    const rank = parseInt(row.rank,10);
    const score = parseFloat(row.score);
    const hue = Math.round(140 * (score / maxScore)); // green for high, dimmer for low
    const scoreColor = `hsl(${130+hue*0},60%,${30+hue*0.12}%)`;

    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    tr.style.setProperty('--score-color', score > 0.7 ? 'var(--g)' : score > 0.55 ? '#6B8B5A' : 'var(--t3)');
    tr.style.animationDelay = `${Math.min(i*18,300)}ms`;

    const badgeClass = rank===1?'r1':rank===2?'r2':rank===3?'r3':'';
    tr.innerHTML = `
      <td><span class="rank-num ${badgeClass}">${rank}</span></td>
      <td><span class="cand-id">${esc(row.candidate_id)}</span></td>
      <td><span class="cand-title">${esc(row.title||row.name||'')}</span></td>
      <td><span class="cand-loc">${esc(row.location||'')}</span></td>
      <td><span class="score-tag">${score.toFixed(4)}</span></td>
      <td><div class="reasoning-clip">${esc(clip(row.reasoning||'',105))}</div></td>
    `;
    tr.addEventListener('click', () => openDrawer(row, i));
    tbody.appendChild(tr);
  });
  focusedIdx = -1;
}

// ---- Search & sort ----
search.addEventListener('input', applySort);

chips.forEach(c => {
  c.addEventListener('click', () => {
    const col = c.dataset.col;
    sortDir = sortCol===col ? -sortDir : 1;
    sortCol = col;
    chips.forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    applySort();
  });
});

// ---- Keyboard navigation ----
document.addEventListener('keydown', e => {
  if (drawer.style.display !== 'none') {
    if (e.key==='Escape') closeDrawer();
    return;
  }
  if (!filteredRows.length) return;
  if (e.key==='j'||e.key==='ArrowDown') {
    e.preventDefault();
    setFocus(Math.min(focusedIdx+1, filteredRows.length-1));
  } else if (e.key==='k'||e.key==='ArrowUp') {
    e.preventDefault();
    setFocus(Math.max(focusedIdx-1,0));
  } else if (e.key==='Enter' && focusedIdx>=0) {
    openDrawer(filteredRows[focusedIdx], focusedIdx);
  }
});

function setFocus(idx) {
  focusedIdx = idx;
  tbody.querySelectorAll('tr').forEach((tr,i) => tr.classList.toggle('focused', i===idx));
  tbody.querySelectorAll('tr')[idx]?.scrollIntoView({block:'nearest'});
}

// ---- Drawer ----
function openDrawer(row, idx) {
  $('d-rank').textContent  = `Rank ${row.rank}`;
  $('d-id').textContent    = row.candidate_id;
  $('d-title').textContent = row.title||row.name||'';
  $('d-score').textContent = parseFloat(row.score).toFixed(4);
  $('d-reasoning').textContent = row.reasoning||'';

  const locRow = $('d-loc-row');
  locRow.style.display = row.location ? 'flex' : 'none';
  $('d-location').textContent = row.location||'';

  const expRow = $('d-exp-row');
  expRow.style.display = row.years_exp!=null ? 'flex' : 'none';
  $('d-exp').textContent = row.years_exp!=null ? `${row.years_exp} years` : '';

  // Score ring animation
  const score = parseFloat(row.score);
  const circ = 2*Math.PI*26; // 163.36
  const offset = circ*(1-score);
  setTimeout(()=>{ $('ring-fill').style.strokeDashoffset = offset; }, 60);

  // Facet pills
  buildFacets(row.facet_hits||{}, row.missed_must_haves||[]);

  overlay.style.display = 'block';
  drawer.style.display  = 'flex';
}

function buildFacets(hits, missedMust) {
  const grid = $('facets-grid');
  grid.innerHTML = '';
  FACETS.forEach(f => {
    const matched = (hits[f.key]||0) > 0;
    const isMissedMust = missedMust.includes(f.key);
    const pill = document.createElement('span');
    pill.className = 'facet-pill ' + (matched ? 'met' : isMissedMust ? 'miss-must' : 'miss-nice');
    pill.innerHTML = `<span class="facet-dot"></span>${esc(f.label)}${isMissedMust ? '<span class="facet-badge">REQ</span>' : ''}`;
    grid.appendChild(pill);
  });
}

function closeDrawer() {
  overlay.style.display = 'none';
  drawer.style.display  = 'none';
  $('ring-fill').style.strokeDashoffset = '163.36';
}

drawerClose.addEventListener('click', closeDrawer);
overlay.addEventListener('click', closeDrawer);

// ---- Helpers ----
function setState(s) {
  stateEmpty.style.display   = s==='empty'   ? 'flex' : 'none';
  stateLoading.style.display = s==='loading' ? 'flex' : 'none';
  stateError.style.display   = s==='error'   ? 'flex' : 'none';
  tableWrap.style.display    = s==='table'   ? 'block': 'none';
}
function setErr(msg) { errorText.textContent = msg; setState('error'); }

function animateCount(el, target) {
  const start = Date.now();
  const dur = 600;
  const from = 0;
  const fn = () => {
    const t = Math.min((Date.now()-start)/dur, 1);
    const ease = 1-Math.pow(1-t,3);
    el.textContent = Math.round(from + (target-from)*ease).toLocaleString();
    if (t<1) requestAnimationFrame(fn);
  };
  requestAnimationFrame(fn);
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function clip(s,n) { return s.length>n ? s.slice(0,n)+'...' : s; }
