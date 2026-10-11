// chessml web app: renders the server's game state and sends user actions to the API.
"use strict";

const GLYPH = { 1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚" };
const FILES = "abcdefgh";
const $ = (id) => document.getElementById(id);

let S = null;            // latest state from the server
let flipped = false;     // board orientation (white at the bottom when false)
let selected = null;     // selected square name, e.g. "e2"
let busy = false;        // a request is running
let hintMove = null;     // {from, to} of the engine's suggested move

const sqName = (i) => FILES[i % 8] + (Math.floor(i / 8) + 1);
const sqIndex = (name) => FILES.indexOf(name[0]) + 8 * (Number(name[1]) - 1);

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + kind;
}

function statusText() {
  if (S.outcome) {
    const r = S.outcome.reason.replaceAll("_", " ");
    return `Game over: ${S.outcome.result} (${r})`;
  }
  const side = S.turn === "white" ? "White" : "Black";
  const check = S.in_check ? " — check!" : "";
  if (S.mode === "analysis") return `Analysis board: ${side} to move${check}`;
  return (S.turn === S.human ? `Your move (${side})` : `Model to move (${side})`) + check;
}

// ------------------------------------------------------------------ board

function canMoveFrom(name) {
  if (busy || !S || S.outcome || S.model_to_move) return false;
  const p = S.pieces[sqIndex(name)];
  if (!p || (p > 0 ? "white" : "black") !== S.turn) return false;
  return S.legal.some((m) => m.from === name);
}

function renderBoard() {
  const board = $("board");
  board.innerHTML = "";
  const targets = selected ? new Set(S.legal.filter((m) => m.from === selected).map((m) => m.to)) : new Set();
  const kingInCheck = S.in_check
    ? sqName(S.pieces.findIndex((p) => p === (S.turn === "white" ? 6 : -6))) : null;
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const rank = flipped ? row : 7 - row;
      const file = flipped ? 7 - col : col;
      const i = rank * 8 + file;
      const name = sqName(i);
      const p = S.pieces[i];
      const sq = document.createElement("div");
      sq.className = "sq " + ((rank + file) % 2 ? "light" : "dark");
      sq.dataset.sq = name;
      if (S.last_move && (S.last_move.from === name || S.last_move.to === name)) sq.classList.add("last");
      if (selected === name) sq.classList.add("sel");
      if (hintMove && (hintMove.from === name || hintMove.to === name)) sq.classList.add("hint");
      if (targets.has(name)) sq.classList.add("target", ...(p ? ["occupied"] : []));
      if (kingInCheck === name) sq.classList.add("check");
      if (col === 0) sq.insertAdjacentHTML("beforeend", `<span class="coord rank">${rank + 1}</span>`);
      if (row === 7) sq.insertAdjacentHTML("beforeend", `<span class="coord file">${FILES[file]}</span>`);
      if (p) {
        const piece = document.createElement("span");
        piece.className = "piece " + (p > 0 ? "w" : "b");
        piece.textContent = GLYPH[Math.abs(p)];
        if (canMoveFrom(name)) {
          piece.addEventListener("pointerdown", (e) => startDrag(e, name, piece));
        } else {
          piece.classList.add("fixed");
        }
        sq.appendChild(piece);
      }
      sq.addEventListener("click", () => onSquare(name));
      board.appendChild(sq);
    }
  }
}

// Dragging uses pointer events (mouse, touch, pen). The board is NOT re-rendered while a
// piece is being dragged: removing the dragged element would cancel the drag.
let drag = null;          // {from, piece, ghost, startX, startY, moved}
let suppressClick = false;

function showTargets(from) {
  const targets = new Set(S.legal.filter((m) => m.from === from).map((m) => m.to));
  for (const sq of $("board").children) {
    const name = sq.dataset.sq;
    sq.classList.toggle("sel", name === from);
    sq.classList.toggle("target", targets.has(name));
    sq.classList.toggle("occupied", targets.has(name) && !!sq.querySelector(".piece"));
  }
}

function startDrag(e, from, piece) {
  if (busy || e.button > 0) return;
  e.preventDefault();  // no text selection or native image drag
  drag = { from, piece, ghost: null, startX: e.clientX, startY: e.clientY, moved: false };
  window.addEventListener("pointermove", onDragMove);
  window.addEventListener("pointerup", onDragEnd, { once: true });
  window.addEventListener("pointercancel", cancelDrag, { once: true });
}

function onDragMove(e) {
  if (!drag) return;
  if (!drag.moved) {
    if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) < 4) return;  // still a click
    drag.moved = true;
    const rect = drag.piece.getBoundingClientRect();
    const ghost = drag.piece.cloneNode(true);
    ghost.classList.add("ghost");
    ghost.style.width = `${rect.width}px`;
    ghost.style.fontSize = getComputedStyle(drag.piece).fontSize;
    document.body.appendChild(ghost);
    drag.ghost = ghost;
    drag.piece.classList.add("dragging");
    selected = drag.from;
    showTargets(drag.from);
  }
  drag.ghost.style.left = `${e.clientX}px`;
  drag.ghost.style.top = `${e.clientY}px`;
}

function onDragEnd(e) {
  window.removeEventListener("pointermove", onDragMove);
  const d = drag;
  drag = null;
  if (!d || !d.moved) return;  // a plain click: the square's click handler selects the piece
  d.ghost.remove();
  suppressClick = true;        // the browser still fires a click after this pointerup
  setTimeout(() => { suppressClick = false; }, 0);
  const target = document.elementFromPoint(e.clientX, e.clientY)?.closest(".sq");
  if (target && target.dataset.sq !== d.from) {
    tryMove(d.from, target.dataset.sq);
  } else {
    selected = null;
    renderBoard();
  }
}

function cancelDrag() {
  window.removeEventListener("pointermove", onDragMove);
  if (drag?.ghost) drag.ghost.remove();
  drag = null;
  selected = null;
  renderBoard();
}

function onSquare(name) {
  if (busy || suppressClick) return;
  if (selected && S.legal.some((m) => m.from === selected && m.to === name)) {
    tryMove(selected, name);
  } else if (canMoveFrom(name) && selected !== name) {
    selected = name;
    renderBoard();
  } else {
    selected = null;
    renderBoard();
  }
}

function tryMove(from, to) {
  const options = S.legal.filter((m) => m.from === from && m.to === to);
  selected = null;
  if (options.length === 0) return renderBoard();
  if (options.length === 1) return doMove(options[0].uci);
  // Promotion: let the user pick the piece.
  const promo = $("promo");
  const white = S.turn === "white";
  promo.innerHTML = "";
  for (const [letter, kind] of [["q", 5], ["r", 4], ["b", 3], ["n", 2]]) {
    const b = document.createElement("button");
    b.innerHTML = `<span class="piece ${white ? "w" : "b"}">${GLYPH[kind]}</span>`;
    b.onclick = () => { promo.hidden = true; doMove(from + to + letter); };
    promo.appendChild(b);
  }
  promo.hidden = false;
}

// ------------------------------------------------------------------ panel

function renderEval() {
  const e = S.eval;
  $("ev-w").style.width = `${e.white * 100}%`;
  $("ev-d").style.width = `${e.draw * 100}%`;
  $("ev-b").style.width = `${e.black * 100}%`;
  $("ev-wt").textContent = `White ${(e.white * 100).toFixed(0)}%`;
  $("ev-dt").textContent = `Draw ${(e.draw * 100).toFixed(0)}%`;
  $("ev-bt").textContent = `Black ${(e.black * 100).toFixed(0)}%`;
}

function renderMoves() {
  const blackFirst = S.start_fen.split(" ")[1] === "b";
  let n = Number(S.start_fen.split(" ")[5]) || 1;
  const parts = [];
  S.sans.forEach((san, i) => {
    const whiteMove = blackFirst ? i % 2 === 1 : i % 2 === 0;
    if (whiteMove) parts.push(`<span class="num">${n}.</span>`);
    else if (i === 0) parts.push(`<span class="num">${n}…</span>`);
    const cls = i === S.sans.length - 1 ? ' class="cur"' : "";
    parts.push(`<span${cls}>${san}</span>`);
    if (!whiteMove) n++;
  });
  $("moves").innerHTML = parts.join(" ") || '<span class="muted">No moves yet.</span>';
}

function renderSearch(search, title) {
  $("search-title").textContent = title;
  if (!search) {
    $("search").innerHTML = '<span class="muted">Raw policy: no search was run.</span>';
    return;
  }
  const rows = search.candidates.map((c) => `
    <tr class="${c.engine_choice ? "best" : ""}">
      <td class="mv">${c.move}</td>
      <td>${c.expected_score ?? "–"}</td>
      <td><span class="effort" style="width:${parseInt(c.search_effort) * 0.6}px"></span>${c.search_effort}</td>
      <td>${c.policy_prior}</td>
    </tr>
    <tr class="${c.engine_choice ? "best" : ""}"><td></td><td colspan="3">
      <span class="line">${c.main_line.join(" ")}</span>
      ${c.notes.map((n) => `<span class="note">${n}</span>`).join("")}
    </td></tr>`).join("");
  $("search").innerHTML = `
    <table><thead><tr><th>Move</th><th>Score</th><th>Search effort</th><th>Policy</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="muted small" style="margin-top:6px">${search.simulations} simulations · score = expected result for the side to move</div>`;
}

function renderExplanation(x) {
  $("explain-card").hidden = false;
  $("explain-move").textContent = x.move;
  // LLMs often use **bold** despite being asked for plain text: escape HTML, then render bold.
  const safe = x.text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  $("explain-text").innerHTML = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  const badge = $("explain-badge");
  badge.textContent = x.source === "llm" ? "LLM" : "template";
  badge.className = "badge " + x.source;
  $("explain-note").textContent = x.rejected.length
    ? `Guardrail: ${x.rejected.join("; ")}` + (x.source === "llm" ? " — fixed on retry." : " — fell back to the template.")
    : "";
  $("explain-facts").textContent = JSON.stringify(x.facts, null, 2);
}

function render() {
  renderBoard();
  renderEval();
  renderMoves();
  setStatus(statusText());
  $("sims").value = String(S.simulations);
  $("audience").value = S.audience;
  $("fen-box").hidden = S.mode !== "analysis";
  $("llm-name").textContent = S.llm ? `LLM: ${S.llm}` : "LLM: off (start the server with --llm ollama)";
  $("explain").disabled = !S.llm || S.simulations === 0;
  $("hint").disabled = S.simulations === 0 || !!S.outcome;
  $("explain").textContent = S.mode === "analysis" ? "Explain best move" : "Explain model's move";
}

// ------------------------------------------------------------------ actions

async function run(label, fn) {
  if (busy) return;
  busy = true;
  setStatus(label, "thinking");
  renderBoard();
  try {
    await fn();
  } catch (err) {
    busy = false;
    render();
    setStatus(err.message, "error");
    return;
  }
  busy = false;
  render();
}

async function engineMoveIfNeeded() {
  if (!S.model_to_move) return;
  setStatus("Model is thinking…", "thinking");
  const r = await api("/api/engine-move", {});
  S = r.state;
  renderSearch(r.search, `Why the model played ${r.san}: its search`);
}

function doMove(uci) {
  hintMove = null;
  run("Moving…", async () => {
    S = (await api("/api/move", { move: uci })).state;
    render();
    busy = true;
    await engineMoveIfNeeded();
  });
}

function newGame(fen) {
  const mode = $("mode").value;
  hintMove = null;
  flipped = mode === "play-black";
  $("explain-card").hidden = true;
  run("Starting…", async () => {
    S = await api("/api/new", {
      mode: mode === "analysis" ? "analysis" : "play",
      color: mode === "play-black" ? "black" : "white",
      fen: fen || null,
    });
    render();
    busy = true;
    await engineMoveIfNeeded();
  });
}

$("new").onclick = () => newGame();
$("load-fen").onclick = () => newGame($("fen").value.trim());
$("mode").onchange = () => { $("fen-box").hidden = $("mode").value !== "analysis"; };
$("flip").onclick = () => { flipped = !flipped; renderBoard(); };
$("undo").onclick = () => run("Undoing…", async () => {
  hintMove = null;
  S = await api("/api/undo", {});
});
$("hint").onclick = () => run("Searching…", async () => {
  const r = await api("/api/analyse", {});
  S = r.state;
  const best = r.search.best;
  hintMove = { from: best.slice(0, 2), to: best.slice(2, 4) };
  renderSearch(r.search, "Engine's top moves here");
});
$("explain").onclick = () => run(`Asking the LLM…`, async () => {
  const target = S.mode === "analysis" ? "position" : "last";
  const r = await api("/api/explain", { target });
  S = r.state;
  renderSearch(r.search, `Search behind ${r.explanation.move}`);
  renderExplanation(r.explanation);
});
$("sims").onchange = () => run("Saving…", async () => {
  S = await api("/api/settings", { simulations: Number($("sims").value) });
});
$("audience").onchange = () => run("Saving…", async () => {
  S = await api("/api/settings", { audience: $("audience").value });
});

(async () => {
  try {
    S = await api("/api/state");
    $("mode").value = S.mode === "analysis" ? "analysis" : (S.human === "black" ? "play-black" : "play-white");
    flipped = S.human === "black";
    render();
  } catch (err) {
    setStatus("Could not reach the chessml server: " + err.message, "error");
  }
})();
