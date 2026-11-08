/* ============================================================================
 * POMCP demo — UI controller
 * ==========================================================================*/
(function () {
  'use strict';

  const { RNG, Planner, ParticleBelief } = window.POMCP;
  const { TigerEnv, solveExact } = window.Tiger;

  const PARTICLE_COUNT = 1200;
  const MAX_EPISODE_STEPS = 60;
  const SPEEDS = { 1: 2000, 2: 1250, 3: 760, 4: 380, 5: 150 };

  const $ = (id) => document.getElementById(id);
  const nf = (n) => Math.round(n).toLocaleString('en-US');
  const pct = (b) => (100 * Math.max(0, Math.min(1, b))).toFixed(1) + '%';
  const signed = (x, d = 1) =>
    (x >= 0 ? '+' : '\u2212') + Math.abs(x).toFixed(d);
  const msLabel = (ms) => (ms < 1 ? '<1 ms' : `${ms} ms`);

  const ACTION_LABEL = {
    listen: 'Listen',
    'open-left': 'Open the left door',
    'open-right': 'Open the right door',
  };
  const ACTION_SHORT = {
    listen: 'listen',
    'open-left': 'open L',
    'open-right': 'open R',
  };
  const OBS_LABEL = { L: 'heard LEFT', R: 'heard RIGHT', none: 'no sound' };

  /* ------------------------------- State --------------------------------- */
  const state = {
    env: null,
    planner: null,
    rng: null,
    exact: null,
    exactAverage: 0,
    particles: null,

    trueState: null,
    belief: 0.5,
    particleBelief: 0.5,
    decisionBelief: 0.5,

    episode: 0,
    step: 0,
    return: 0,
    listens: 0,
    phase: 'idle', // ready | playing | ended
    log: [],

    pendingPlan: null,
    lastExact: null,
    lastOutcome: null,

    returns: [],
    safeOpens: 0,
    opens: 0,
    agreements: 0,
    decisions: 0,
    totalSims: 0,
    totalListens: 0,

    auto: false,
    timer: null,
    speed: 3,
  };

  /* ------------------------------ Helpers -------------------------------- */
  function bestActionAt(node) {
    let best = null;
    for (const id of ['listen', 'open-left', 'open-right']) {
      if (!node.Na.has(id)) continue;
      const q = node.Qa.get(id);
      if (!best || q > best.q) best = { id, q, n: node.Na.get(id) };
    }
    return best;
  }

  function simulateExactPolicyAverage(env, exact, episodes = 2500) {
    const rng = new RNG(0xc0ffee);
    let total = 0;
    for (let ep = 0; ep < episodes; ep++) {
      let s = env.sampleInitialState(rng);
      let b = 0.5;
      let ret = 0;
      for (let t = 0; t < MAX_EPISODE_STEPS; t++) {
        let a;
        if (b >= exact.threshold) a = 'open-right';
        else if (b <= 1 - exact.threshold) a = 'open-left';
        else a = 'listen';
        const tr = env.step(s, a, rng);
        ret += tr.reward;
        s = tr.state;
        if (tr.done) break;
        b = env.bayesUpdate(b, a, tr.observation);
      }
      total += ret;
    }
    return total / episodes;
  }

  function clearTimer() {
    if (state.timer !== null) {
      clearTimeout(state.timer);
      state.timer = null;
    }
  }

  /* --------------------------- Episode control --------------------------- */
  function newEpisode() {
    state.episode += 1;
    state.step = 0;
    state.return = 0;
    state.listens = 0;
    state.log = [];
    state.phase = 'ready';
    state.pendingPlan = null;
    state.lastExact = null;
    state.lastOutcome = null;
    state.decisionBelief = 0.5;

    state.trueState = state.env.sampleInitialState(state.rng);
    state.particles = new ParticleBelief(state.env, PARTICLE_COUNT, state.rng);
    state.belief = 0.5;
    state.particleBelief = state.particles.fractionTigerLeft();

    state.planner.reset();
    think();
  }

  function think() {
    if (!state.particles || state.phase === 'ended') return;
    state.decisionBelief = state.belief;
    const plan = state.planner.plan(state.particles, state.planner.simulations);
    state.pendingPlan = plan;
    state.lastExact = state.exact.valuesAt(state.belief);
    state.totalSims += plan.simulations;
  }

  function executeStep() {
    if (state.phase === 'ended') {
      newEpisode();
      return;
    }
    if (!state.pendingPlan) think();

    const plan = state.pendingPlan;
    const action = plan.action;
    const exactNow = state.exact.valuesAt(state.belief);
    state.decisions += 1;
    if (action === exactNow.best) state.agreements += 1;

    const decisionBelief = state.belief;
    const tr = state.env.step(state.trueState, action, state.rng);

    state.return += tr.reward;
    state.step += 1;
    if (action === 'listen') {
      state.listens += 1;
      state.totalListens += 1;
    }
    state.log.push({
      action,
      observation: tr.observation,
      reward: tr.reward,
      belief: decisionBelief,
      done: tr.done,
    });
    state.trueState = tr.state;

    if (tr.done) {
      const opened = action === 'open-left' ? 'L' : 'R';
      const tiger = tr.state.tiger;
      const correct = tr.reward > 0;
      state.opens += 1;
      if (correct) state.safeOpens += 1;
      state.returns.push(state.return);
      state.lastOutcome = {
        reward: tr.reward,
        opened,
        tiger,
        correct,
        decisionBelief,
        action,
      };
      state.decisionBelief = decisionBelief;
      state.phase = 'ended';
      return;
    }

    state.particles.update(action, tr.observation);
    state.particleBelief = state.particles.fractionTigerLeft();
    state.belief = state.env.bayesUpdate(state.belief, action, tr.observation);
    state.planner.advance(action, tr.observation);
    state.phase = 'playing';
    think();
  }

  function setAuto(on) {
    state.auto = on;
    clearTimer();
    renderPlayback();
    if (!state.auto) return;

    if (state.phase === 'ended') {
      newEpisode();
      renderAll();
      schedule(900);
      return;
    }
    schedule(state.phase === 'ready' ? 260 : 120);
  }

  function schedule(ms) {
    clearTimer();
    if (!state.auto) return;
    state.timer = setTimeout(tick, ms);
  }

  function tick() {
    if (!state.auto) return;
    if (state.phase === 'ended') {
      newEpisode();
      renderAll();
      schedule(950);
      return;
    }
    executeStep();
    renderAll();
    if (state.phase === 'ended') schedule(1900);
    else schedule(SPEEDS[state.speed] || 760);
  }

  function restartExperiment() {
    setAuto(false);
    state.episode = 0;
    state.returns = [];
    state.safeOpens = 0;
    state.opens = 0;
    state.agreements = 0;
    state.decisions = 0;
    state.totalSims = 0;
    state.totalListens = 0;
    newEpisode();
    renderAll();
  }

  /* ------------------------------- Render -------------------------------- */
  function renderAll() {
    renderEpisode();
    renderScene();
    renderBelief();
    renderParticles();
    renderDecision();
    renderQ();
    renderTree();
    renderExact();
    renderStats();
    renderReturnChart();
    renderTimeline();
    renderOutcome();
    renderPlayback();
  }

  function renderEpisode() {
    $('episode-num').textContent = state.episode || 1;
  }

  function renderScene() {
    const left = $('door-left');
    const right = $('door-right');
    left.classList.remove('open', 'safe', 'tiger', 'target');
    right.classList.remove('open', 'safe', 'tiger', 'target');

    if (state.pendingPlan && state.phase !== 'ended') {
      if (state.pendingPlan.action === 'open-left') left.classList.add('target');
      if (state.pendingPlan.action === 'open-right') right.classList.add('target');
    }

    if (state.lastOutcome) {
      const o = state.lastOutcome;
      const openedDoor = o.opened === 'L' ? left : right;
      openedDoor.classList.add('open', o.correct ? 'safe' : 'tiger');
      const icon = openedDoor.querySelector('.interior-icon');
      const note = openedDoor.querySelector('.interior-note');
      icon.textContent = o.correct ? '\u2728' : '\uD83D\uDC05';
      note.textContent = o.correct ? 'empty \u2014 safe' : 'tiger!';

      $('caption-left').textContent =
        o.opened === 'L' ? 'opened' : o.tiger === 'L' ? 'tiger was here' : 'safe';
      $('caption-right').textContent =
        o.opened === 'R' ? 'opened' : o.tiger === 'R' ? 'tiger was here' : 'safe';
    } else {
      $('caption-left').textContent = 'hidden';
      $('caption-right').textContent = 'hidden';
    }

    const hearing = $('hearing');
    const hearingText = $('hearing-text');
    const hearingIcon = $('hearing-icon');
    hearing.classList.remove('live', 'right');

    const last = state.log[state.log.length - 1];
    if (state.phase === 'ended') {
      hearingIcon.textContent = '\u2713';
      hearingText.textContent = 'door opened \u2014 truth revealed';
    } else if (last && !last.done && last.action === 'listen') {
      hearingIcon.textContent = '\uD83D\uDD0A';
      hearing.classList.add('live');
      if (last.observation === 'R') hearing.classList.add('right');
      hearingText.textContent =
        last.observation === 'L' ? 'heard LEFT' : 'heard RIGHT';
    } else if (state.step === 0) {
      hearingIcon.textContent = '\u25CC';
      hearingText.textContent = 'the tiger is silent \u2014 listen or open';
    } else {
      hearingIcon.textContent = '\u25CC';
      hearingText.textContent = 'waiting for the next decision';
    }
  }

  function renderBelief() {
    const b = state.belief;
    const thr = state.exact ? state.exact.threshold : 0.98;

    $('belief-value').textContent = pct(b);
    $('particle-value').textContent = pct(state.particleBelief);
    $('belief-fill').style.width = `${b * 100}%`;
    $('marker-exact').style.left = `${b * 100}%`;
    $('marker-particle').style.left = `${state.particleBelief * 100}%`;

    $('threshold-left').style.left = `${(1 - thr) * 100}%`;
    $('threshold-right').style.left = `${thr * 100}%`;
    $('zone-left').style.width = `${(1 - thr) * 100}%`;
    $('zone-right').style.width = `${(1 - thr) * 100}%`;

    $('belief-track').setAttribute(
      'aria-label',
      `Belief that the tiger is behind the left door: ${pct(b)}. ` +
        `Particle estimate: ${pct(state.particleBelief)}.`
    );
  }

  function renderParticles() {
    const host = $('particle-dots');
    if (!state.particles) {
      host.innerHTML = '';
      return;
    }
    const particles = state.particles.particles;
    const shown = 52;
    let left = 0;
    for (const p of particles) if (p.tiger === 'L') left += 1;
    let html = '';
    for (let i = 0; i < shown; i++) {
      const p = particles[Math.floor((i * particles.length) / shown)];
      html += `<i class="${p.tiger === 'L' ? 'left' : 'right'}"></i>`;
    }
    host.innerHTML = html;
    $('particle-caption').textContent =
      `${nf(particles.length)} particles \u00b7 ${nf(left)} left / ${nf(particles.length - left)} right`;
  }

  function renderDecision() {
    const title = $('decision-title');
    const sub = $('decision-sub');
    const dot = $('decision-dot');
    dot.classList.remove('listen', 'open');
    sub.classList.remove('good', 'bad');

    if (state.phase === 'ended' && state.lastOutcome) {
      const o = state.lastOutcome;
      dot.classList.add('open');
      title.textContent =
        `Opened the ${o.opened === 'L' ? 'left' : 'right'} door at ` +
        `${pct(o.decisionBelief)} belief`;
      const exact = state.lastExact;
      if (exact) {
        const exactLabel = ACTION_LABEL[exact.best];
        if (o.action === exact.best) {
          sub.textContent = `The exact planner agrees this was optimal at that belief.`;
          sub.classList.add('good');
        } else {
          sub.textContent =
            `Exact planner would have chosen \u201c${exactLabel}\u201d at this belief ` +
            `(threshold ${pct(state.exact.threshold)}).`;
          sub.classList.add('bad');
        }
      } else {
        sub.textContent = '';
      }
      return;
    }

    const plan = state.pendingPlan;
    if (!plan) {
      title.textContent = 'Planning\u2026';
      sub.textContent = '';
      return;
    }

    dot.classList.add(plan.action === 'listen' ? 'listen' : 'open');
    title.textContent = `POMCP chooses \u201c${ACTION_LABEL[plan.action]}\u201d`;
    const exact = state.lastExact;
    const match = exact && exact.best === plan.action;
    sub.textContent =
      `${nf(plan.simulations)} simulations \u00b7 ${msLabel(plan.ms)} \u00b7 ` +
      (match
        ? 'matches the exact planner'
        : exact
        ? `exact planner: ${ACTION_LABEL[exact.best]}`
        : '');
    if (exact) sub.classList.add(match ? 'good' : 'bad');
  }

  function renderQ() {
    const host = $('q-values');
    const plan = state.pendingPlan;
    if (!plan) {
      host.innerHTML = '<div class="q-empty">Run a search to see action values.</div>';
      $('plan-meta').textContent = '\u2014';
      return;
    }

    $('plan-meta').textContent = `${nf(plan.simulations)} sims \u00b7 ${msLabel(plan.ms)}`;

    let maxAbs = 1;
    for (const row of plan.rows) {
      if (row.Q !== null) maxAbs = Math.max(maxAbs, Math.abs(row.Q));
    }
    maxAbs = Math.max(maxAbs, 10);

    let html = '';
    for (const row of plan.rows) {
      const chosen = row.id === plan.action;
      const q = row.Q;
      const width = q === null ? 0 : Math.max(1.2, Math.min(46, (Math.abs(q) / maxAbs) * 46));
      const left = q === null ? 50 : q < 0 ? 50 - width : 50;
      const qText = q === null ? '\u2014' : signed(q, 1);
      html += `
        <div class="q-row ${chosen ? 'chosen' : ''}" title="${ACTION_LABEL[row.id]}: Q = ${q === null ? 'not sampled' : q.toFixed(2)}, visits = ${row.N}">
          <span class="q-name">${ACTION_SHORT[row.id]}</span>
          <span class="q-track">
            <span class="q-zero"></span>
            <span class="q-bar ${q !== null && q < 0 ? 'negative' : ''}" style="left:${left}%;width:${width}%"></span>
          </span>
          <span class="q-value">${qText}</span>
          <span class="q-n">N ${nf(row.N)}</span>
        </div>`;
    }
    host.innerHTML = html;
  }

  function renderTree() {
    const host = $('tree');
    const meta = $('tree-meta');
    const plan = state.pendingPlan;

    if (!plan || !plan.root || plan.root.N === 0) {
      meta.textContent = '\u2014';
      host.innerHTML =
        '<div class="tree-empty">The search tree appears after the first simulation.</div>';
      return;
    }

    const root = plan.root;
    const rootN = Math.max(1, plan.rootN || root.N);
    meta.textContent = `depth ${plan.depth} \u00b7 ${nf(plan.nodes)} nodes \u00b7 ${msLabel(plan.ms)}`;

    const visitBar = (n) => {
      const w = Math.max(2, Math.sqrt(n / rootN) * 100);
      return `<span class="tree-bar"><i style="width:${w}%"></i></span>`;
    };

    let html = `
      <div class="tree-row root">
        <span class="tree-action">root</span>
        <span class="tree-best">history h</span>
        <span class="tree-q">N ${nf(root.N)}</span>
        <span class="tree-n"></span>
      </div>`;

    plan.rows.forEach((row, i) => {
      const lastRow = i === plan.rows.length - 1;
      const chosen = row.id === plan.action;
      const branch = lastRow ? '\u2514\u2500' : '\u251c\u2500';
      const qText = row.Q === null ? '\u2014' : signed(row.Q, 1);
      html += `
        <div class="tree-row ${chosen ? 'chosen' : ''}">
          <span class="tree-branch">${branch}</span>
          <span class="tree-action">${ACTION_SHORT[row.id]}</span>
          <span class="tree-q">Q ${qText}</span>
          ${visitBar(row.N)}
          <span class="tree-n">N ${nf(row.N)}</span>
        </div>`;

      if (row.id === 'listen') {
        for (const obs of ['L', 'R']) {
          const child = root.children.get('listen|' + obs);
          if (!child || child.N === 0) continue;
          const best = bestActionAt(child);
          const prefix = lastRow ? '   ' : '\u2502  ';
          html += `
            <div class="tree-row">
              <span class="tree-branch">${prefix}\u2514</span>
              <span class="tree-obs">heard ${obs === 'L' ? 'L' : 'R'}</span>
              <span class="tree-best">\u2192 ${best ? ACTION_SHORT[best.id] : '\u2014'}</span>
              <span class="tree-q"></span>
              ${visitBar(child.N)}
              <span class="tree-n">N ${nf(child.N)}</span>
            </div>`;
        }
      }
    });

    host.innerHTML = html;
  }

  function renderExact() {
    const exact = state.exact;
    if (!exact) return;
    const b = state.phase === 'ended' ? state.decisionBelief : state.belief;
    const values = exact.valuesAt(b);

    $('exact-badge').textContent = ACTION_LABEL[values.best].toLowerCase();

    const makeVal = (id, value) => {
      const best = values.best === id;
      return `<span class="val ${best ? 'best' : ''}">${ACTION_SHORT[id]} <b>${signed(value, 1)}</b></span>`;
    };
    $('exact-values').innerHTML =
      makeVal('listen', values.listen) +
      makeVal('open-left', values.openLeft) +
      makeVal('open-right', values.openRight);

    const compare = $('exact-compare');
    compare.classList.remove('match', 'miss');
    const plan = state.pendingPlan;
    if (plan) {
      if (plan.action === values.best) {
        compare.textContent = 'POMCP matches';
        compare.classList.add('match');
      } else {
        compare.textContent = 'POMCP differs';
        compare.classList.add('miss');
      }
    } else {
      compare.textContent = '\u2014';
    }

    $('threshold-note').textContent =
      `open when P(left) ${'\u2265'} ${pct(exact.threshold)} (or ${'\u2264'} ${pct(1 - exact.threshold)})`;

    renderValueChart(b, values);
  }

  function renderValueChart(b, values) {
    const exact = state.exact;
    const W = 380;
    const H = 154;
    const ML = 33;
    const MR = 10;
    const MT = 10;
    const MB = 23;
    const plotW = W - ML - MR;
    const plotH = H - MT - MB;

    const curve = exact.curve(201);
    let vMin = Infinity;
    let vMax = -Infinity;
    for (const p of curve) {
      if (p.v < vMin) vMin = p.v;
      if (p.v > vMax) vMax = p.v;
    }
    const yMin = Math.floor(vMin) - 0.6;
    const yMax = Math.ceil(vMax) + 0.4;

    const x = (v) => ML + v * plotW;
    const y = (v) => MT + ((yMax - v) / (yMax - yMin)) * plotH;
    const thr = exact.threshold;

    let grid = '';
    const ticks = 3;
    for (let i = 0; i < ticks; i++) {
      const val = yMin + ((yMax - yMin) * i) / (ticks - 1);
      const yy = y(val);
      grid += `<line x1="${ML}" y1="${yy}" x2="${W - MR}" y2="${yy}" stroke="#1b2735" stroke-width="1"/>`;
      grid += `<text x="${ML - 6}" y="${yy + 3}" text-anchor="end" font-size="8" fill="#637184" font-family="monospace">${val.toFixed(0)}</text>`;
    }

    const zoneLeft = x(1 - thr);
    const zoneRight = x(thr);
    const zones = `
      <rect x="${ML}" y="${MT}" width="${Math.max(0, zoneLeft - ML)}" height="${plotH}" fill="rgba(245,182,74,0.055)"/>
      <rect x="${zoneLeft}" y="${MT}" width="${Math.max(0, zoneRight - zoneLeft)}" height="${plotH}" fill="rgba(45,212,191,0.055)"/>
      <rect x="${zoneRight}" y="${MT}" width="${Math.max(0, W - MR - zoneRight)}" height="${plotH}" fill="rgba(245,182,74,0.055)"/>`;

    let line = '';
    let area = '';
    curve.forEach((p, i) => {
      line += `${i === 0 ? 'M' : 'L'}${x(p.b).toFixed(1)},${y(p.v).toFixed(1)} `;
    });
    area =
      `M${x(0).toFixed(1)},${y(yMin).toFixed(1)} ` +
      line.slice(1) +
      `L${x(1).toFixed(1)},${y(yMin).toFixed(1)} Z`;

    const xMarker = x(b);
    const yMarker = y(values.value);
    const markerLabelX = Math.min(W - MR - 22, Math.max(ML + 16, xMarker));

    const svg = `
      <defs>
        <linearGradient id="vgrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2dd4bf" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${zones}
      ${grid}
      <path d="${area}" fill="url(#vgrad)"/>
      <path d="${line}" fill="none" stroke="#2dd4bf" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>
      <line x1="${zoneLeft}" y1="${MT}" x2="${zoneLeft}" y2="${MT + plotH}" stroke="rgba(231,238,247,0.35)" stroke-width="1" stroke-dasharray="3 3"/>
      <line x1="${zoneRight}" y1="${MT}" x2="${zoneRight}" y2="${MT + plotH}" stroke="rgba(231,238,247,0.35)" stroke-width="1" stroke-dasharray="3 3"/>
      <line x1="${xMarker}" y1="${MT}" x2="${xMarker}" y2="${MT + plotH}" stroke="rgba(245,182,74,0.62)" stroke-width="1" stroke-dasharray="2 3"/>
      <circle cx="${xMarker}" cy="${yMarker}" r="3.6" fill="#f5b64a" stroke="#080b10" stroke-width="1.5"/>
      <text x="${markerLabelX}" y="${MT + 9}" text-anchor="middle" font-size="8.5" fill="#f5b64a" font-family="monospace">b=${b.toFixed(3)}</text>
      <text x="${ML}" y="${H - 7}" font-size="8" fill="#637184" font-family="monospace">0</text>
      <text x="${x(0.5)}" y="${H - 7}" text-anchor="middle" font-size="8" fill="#637184" font-family="monospace">0.5</text>
      <text x="${W - MR}" y="${H - 7}" text-anchor="end" font-size="8" fill="#637184" font-family="monospace">1</text>`;

    $('value-chart').innerHTML = svg;
  }

  function renderStats() {
    const episodes = state.returns.length;
    $('stat-episodes').textContent = nf(episodes);

    if (episodes) {
      const sum = state.returns.reduce((a, b) => a + b, 0);
      $('stat-avg').textContent = signed(sum / episodes, 2);
    } else {
      $('stat-avg').textContent = '\u2014';
    }

    $('stat-safe').textContent = state.opens
      ? `${Math.round((100 * state.safeOpens) / state.opens)}%`
      : '\u2014';
    $('stat-listens').textContent = episodes
      ? (state.totalListens / episodes).toFixed(2)
      : '\u2014';
    $('stat-agree').textContent = state.decisions
      ? `${Math.round((100 * state.agreements) / state.decisions)}%`
      : '\u2014';
    $('stat-sims').textContent = state.totalSims >= 1000000
      ? (state.totalSims / 1000000).toFixed(2) + 'M'
      : state.totalSims >= 10000
      ? (state.totalSims / 1000).toFixed(1) + 'k'
      : nf(state.totalSims);
  }

  function renderReturnChart() {
    const svg = $('return-chart');
    const data = state.returns.slice(-30);
    const W = 680;
    const H = 180;
    const ML = 42;
    const MR = 14;
    const MT = 14;
    const MB = 26;
    const plotW = W - ML - MR;
    const plotH = H - MT - MB;
    const yMin = -112;
    const yMax = 18;

    const y = (v) => MT + ((yMax - v) / (yMax - yMin)) * plotH;

    let grid = '';
    for (const tick of [10, 0, -50, -100]) {
      const yy = y(tick);
      grid += `<line x1="${ML}" y1="${yy}" x2="${W - MR}" y2="${yy}" stroke="${tick === 0 ? 'rgba(231,238,247,0.22)' : '#1b2735'}" stroke-width="1" ${tick === 0 ? '' : 'stroke-dasharray="2 4"'}/>`;
      grid += `<text x="${ML - 7}" y="${yy + 3}" text-anchor="end" font-size="8.5" fill="#637184" font-family="monospace">${tick > 0 ? '+' + tick : tick}</text>`;
    }

    if (!data.length) {
      svg.innerHTML =
        grid +
        `<text x="${ML + plotW / 2}" y="${MT + plotH / 2}" text-anchor="middle" font-size="10" fill="#637184" font-style="italic">Complete an episode to see its return.</text>`;
      return;
    }

    const slot = plotW / data.length;
    const barW = Math.max(3, Math.min(18, slot * 0.62));
    let bars = '';
    data.forEach((v, i) => {
      const cx = ML + slot * (i + 0.5);
      const zero = y(0);
      const top = y(Math.max(v, 0));
      const h = Math.max(1.5, Math.abs(y(v) - zero));
      const color = v >= 0 ? '#4ade80' : '#f87171';
      const opacity = 0.45 + 0.55 * ((i + 1) / data.length);
      const title = `Episode ${state.returns.length - data.length + i + 1}: ${signed(v, 0)}`;
      bars += `<rect x="${(cx - barW / 2).toFixed(1)}" y="${top.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="${Math.min(3, barW / 3).toFixed(1)}" fill="${color}" opacity="${opacity.toFixed(2)}"><title>${title}</title></rect>`;
    });

    const avgY = y(state.exactAverage);
    const avgLabelX = W - MR - 4;
    const avg = `
      <line x1="${ML}" y1="${avgY}" x2="${W - MR}" y2="${avgY}" stroke="#f5b64a" stroke-width="1.1" stroke-dasharray="5 4" opacity="0.75"/>
      <text x="${avgLabelX}" y="${avgY - 5}" text-anchor="end" font-size="8.5" fill="#f5b64a" font-family="monospace">exact policy ${signed(state.exactAverage, 1)}</text>`;

    svg.innerHTML =
      grid +
      avg +
      bars +
      `<text x="${ML}" y="${H - 7}" font-size="8" fill="#637184" font-family="monospace">older</text>
       <text x="${W - MR}" y="${H - 7}" text-anchor="end" font-size="8" fill="#637184" font-family="monospace">newer</text>`;

    $('returns-note').textContent = `last ${data.length} of ${state.returns.length}`;
  }

  function renderTimeline() {
    const host = $('timeline');
    if (!state.log.length) {
      host.innerHTML = '<span class="timeline-empty">No actions yet.</span>';
      return;
    }
    let html = '';
    for (const entry of state.log.slice(-14)) {
      if (entry.action === 'listen') {
        const side = entry.observation === 'L' ? 'L' : 'R';
        html += `<span class="chip listen">listen <em>\u2192</em> ${side} <em>${entry.reward}</em></span>`;
      } else {
        const good = entry.reward > 0;
        html += `<span class="chip open ${good ? 'good' : 'bad'}">${ACTION_SHORT[entry.action]} <em>${signed(entry.reward, 0)}</em></span>`;
      }
    }
    host.innerHTML = html;
  }

  function renderOutcome() {
    const el = $('outcome');
    if (state.phase !== 'ended' || !state.lastOutcome) {
      el.hidden = true;
      return;
    }
    const o = state.lastOutcome;
    el.hidden = false;
    el.classList.toggle('good', o.correct);
    el.classList.toggle('bad', !o.correct);
    $('outcome-title').textContent = o.correct
      ? `Safe! ${signed(o.reward, 0)}`
      : `Tiger! ${signed(o.reward, 0)}`;
    $('outcome-sub').textContent =
      `The tiger was behind the ${o.tiger === 'L' ? 'left' : 'right'} door. ` +
      `POMCP opened at belief ${pct(o.decisionBelief)} ` +
      (o.correct
        ? `(exact threshold ${pct(state.exact.threshold)}).`
        : `\u2014 below the exact threshold (${pct(state.exact.threshold)}).`);
  }

  function renderPlayback() {
    $('play-icon').textContent = state.auto ? '\u23F8' : '\u25B6';
    $('play-label').textContent = state.auto ? 'Pause' : 'Auto play';
    $('step').textContent = state.phase === 'ended' ? 'Next episode' : 'Step';
  }

  /* ------------------------------- Events -------------------------------- */
  function bind() {
    $('play').addEventListener('click', () => setAuto(!state.auto));

    $('step').addEventListener('click', () => {
      if (state.auto) setAuto(false);
      if (state.phase === 'ended') newEpisode();
      else executeStep();
      renderAll();
    });

    $('new-episode').addEventListener('click', () => {
      setAuto(false);
      newEpisode();
      renderAll();
    });

    $('next-episode').addEventListener('click', () => {
      if (state.auto) setAuto(false);
      newEpisode();
      renderAll();
    });

    $('reset-stats').addEventListener('click', restartExperiment);

    $('speed').addEventListener('input', (e) => {
      state.speed = Number(e.target.value);
    });

    const sims = $('sims');
    const exploration = $('exploration');
    const noise = $('noise');
    const gamma = $('gamma');

    sims.addEventListener('input', () => {
      $('sims-out').textContent = nf(Number(sims.value));
    });
    sims.addEventListener('change', () => {
      state.planner.simulations = Number(sims.value);
      if (state.phase !== 'ended') {
        state.planner.reset();
        think();
        renderAll();
      }
    });

    exploration.addEventListener('input', () => {
      $('exploration-out').textContent = exploration.value;
    });
    exploration.addEventListener('change', () => {
      state.planner.exploration = Number(exploration.value);
      if (state.phase !== 'ended') {
        state.planner.reset();
        think();
        renderAll();
      }
    });

    noise.addEventListener('input', () => {
      $('noise-out').textContent = Math.round(Number(noise.value) * 100) + '%';
    });
    noise.addEventListener('change', () => {
      state.env.noise = Number(noise.value);
      state.exact = solveExact(state.env, { points: 501 });
      state.exactAverage = simulateExactPolicyAverage(state.env, state.exact);
      restartExperiment();
    });

    gamma.addEventListener('input', () => {
      $('gamma-out').textContent = Number(gamma.value).toFixed(2);
    });
    gamma.addEventListener('change', () => {
      const v = Number(gamma.value);
      state.env.gamma = v;
      state.planner.gamma = v;
      state.exact = solveExact(state.env, { points: 501 });
      state.exactAverage = simulateExactPolicyAverage(state.env, state.exact);
      restartExperiment();
    });

    document.addEventListener('keydown', (e) => {
      if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === ' ' || e.code === 'Space') {
        e.preventDefault();
        setAuto(!state.auto);
      } else if (key === 's') {
        e.preventDefault();
        if (state.auto) setAuto(false);
        if (state.phase === 'ended') newEpisode();
        else executeStep();
        renderAll();
      } else if (key === 'n') {
        e.preventDefault();
        setAuto(false);
        newEpisode();
        renderAll();
      }
    });

    document.addEventListener('visibilitychange', () => {
      if (document.hidden && state.auto) setAuto(false);
    });
  }

  /* -------------------------------- Init --------------------------------- */
  function init() {
    const seed = (Date.now() ^ 0x9e3779b9) >>> 0;
    state.rng = new RNG(seed);
    state.env = new TigerEnv({ noise: 0.9, gamma: 0.95 });
    state.planner = new Planner(state.env, {
      simulations: 5000,
      exploration: 100,
      gamma: 0.95,
      maxDepth: 40,
      rng: state.rng,
    });
    state.exact = solveExact(state.env, { points: 501 });
    state.exactAverage = simulateExactPolicyAverage(state.env, state.exact);

    $('sims-out').textContent = nf(5000);
    $('exploration-out').textContent = '100';
    $('noise-out').textContent = '90%';
    $('gamma-out').textContent = '0.95';

    bind();
    newEpisode();
    renderAll();
  }

  init();
})();
