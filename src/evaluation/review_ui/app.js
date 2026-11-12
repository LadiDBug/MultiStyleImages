(function () {
  'use strict';
  const C = window.RQ1Review, config = JSON.parse(document.getElementById('review-config').textContent);
  const $ = id => document.getElementById(id), fields = Array.from(document.querySelectorAll('[data-question]'));
  const storageKey = 'rq1-review:' + config.dataset_id, prefsKey = storageKey + ':prefs';
  const labels = {content: 'Contenuto e struttura', style: 'Stile', artifacts: 'Artefatti', tradeoff: 'Compromesso'};
  let state = C.empty(config.dataset_id), reviewer = '', currentId = config.items[0].id;
  let blockedStorage = false, damagedStorage = '', lastVisible = [];
  const now = () => new Date().toISOString();
  const recordFor = (id, name = reviewer) => state.records.find(r => r.review_id === id && r.reviewer === name);
  const hasText = r => r && C.QUESTIONS.some(q => r.answers[q].trim());
  const currentItem = () => config.items.find(item => item.id === currentId);
  function message(text, error = false) {
    $('message').textContent = text; $('message').hidden = false;
    $('message').style.background = error ? '#fff0f0' : '#e5f3ef';
  }
  function savePrefs() {
    try {localStorage.setItem(prefsKey, JSON.stringify({reviewer, currentId,
      filters: Object.fromEntries(['part', 'style', 'cohort', 'status'].map(k => [k, $(k).value]))}));} catch (_) {}
  }
  function save() {
    if (blockedStorage) { $('storage-warning').hidden = false; $('save-state').textContent = 'Solo in memoria: esporta JSON'; return false; }
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
      $('storage-warning').hidden = true;
      $('save-state').textContent = 'Salvato nel browser · ' + new Date().toLocaleTimeString('it-IT');
      savePrefs(); return true;
    } catch (_) {
      $('storage-warning').hidden = false; $('save-state').textContent = 'Salvataggio locale non disponibile: esporta JSON';
      return false;
    }
  }
  function newRecord(item) {
    return {review_id: item.id, image_id: item.image_id, style: item.style, seed: item.seed,
      reviewer, updated_at: now(), completed: false,
      answers: Object.fromEntries(C.QUESTIONS.map(q => [q, '']))};
  }
  function activeRecord() {
    if (!reviewer) throw Error('Inserisci prima il tuo nome e premi Apri le mie risposte.');
    let r = recordFor(currentId);
    if (!r) {r = newRecord(currentItem()); state.records.push(r);}
    return r;
  }
  function makeText(tag, text) {const el = document.createElement(tag); el.textContent = text; return el;}
  function refreshProgress() {
    const assigned = config.items.filter(item => $('part').value === 'all' || item.part === $('part').value);
    const assignedIds = new Set(assigned.map(item => item.id));
    const mine = state.records.filter(r => r.reviewer === reviewer && r.completed && assignedIds.has(r.review_id)).length;
    const covered = new Set(state.records.filter(r => r.completed).map(r => r.review_id)).size;
    $('progress-text').textContent = reviewer ? `${mine}/${assigned.length} mie completate` : 'Inserisci il tuo nome per iniziare';
    $('team-progress').textContent = ` · ${covered}/${config.items.length} coperte dalle risposte caricate`;
    $('progress-bar').style.width = (100 * mine / Math.max(assigned.length, 1)) + '%';
    $('active-reviewer').textContent = reviewer ? 'Stai compilando come: ' + reviewer : '';
    $('reviewers').replaceChildren(...Array.from(new Set(state.records.map(r => r.reviewer))).sort().map(name => {
      const option = document.createElement('option'); option.value = name; return option;
    }));
  }
  function visibleItems() {
    return config.items.filter(item => {
      const r = recordFor(item.id), done = !!r?.completed;
      return ($('part').value === 'all' || $('part').value === item.part)
        && ($('style').value === 'all' || $('style').value === item.style)
        && ($('cohort').value === 'all' || item.sets.includes($('cohort').value))
        && ($('status').value === 'all' || ($('status').value === 'done' ? done : !done));
    });
  }
  function renderOtherReviewers() {
    const others = state.records.filter(r => r.review_id === currentId && r.reviewer !== reviewer && hasText(r));
    $('other-reviewers-title').textContent = `Risposte di altri revisori (${others.length})`;
    $('other-reviewers').replaceChildren();
    for (const r of others) {
      const box = document.createElement('div'); box.className = 'conflict';
      box.append(makeText('h3', r.reviewer + (r.completed ? ' · completata' : ' · bozza')));
      for (const q of C.QUESTIONS) {box.append(makeText('strong', labels[q]), makeText('p', r.answers[q] || 'Non compilata'));}
      $('other-reviewers').append(box);
    }
  }
  function renderCard() {
    const item = currentItem(); if (!item) return;
    const r = recordFor(item.id);
    $('case-title').textContent = `${item.number}/${config.items.length} · ${item.image_id} · ${item.style_label} · seed ${item.seed}`;
    $('case-reason').textContent = item.reasons.join(' · ');
    $('case-cohort').textContent = `Parte ${item.part} · ${item.sets.includes('random') ? 'Campione casuale' : 'Caso diagnostico'} · osserva tutti e quattro i metodi`;
    $('grid-link').href = item.grid; $('grid-image').src = item.grid;
    $('grid-image').alt = `Confronto di ${item.image_id}, stile ${item.style_label}: originale e quattro condizioni`;
    $('source-links').replaceChildren(makeText('span', 'PNG a piena risoluzione: '));
    for (const [condition, url] of Object.entries(item.sources)) {
      const link = makeText('a', condition); link.href = url; link.target = '_blank'; link.rel = 'noopener';
      $('source-links').append(link, document.createTextNode(' · '));
    }
    $('style-hint').textContent = item.style_hint;
    for (const f of fields) {f.value = r?.answers[f.dataset.question] || ''; f.disabled = !reviewer;}
    $('mark-complete').disabled = $('mark-draft').disabled = !reviewer;
    refreshCardStatus(); renderOtherReviewers();
  }
  function refreshCardStatus() {
    const r = recordFor(currentId), done = !!r?.completed;
    $('case-status').textContent = done ? 'Completata' : hasText(r) ? 'In bozza' : 'Da iniziare';
    $('case-status').className = 'badge' + (done ? ' done' : '');
    $('record-save').textContent = r ? 'Ultima modifica: ' + new Date(r.updated_at).toLocaleString('it-IT') : 'Le risposte appariranno qui dopo che inizi a scrivere.';
  }
  function renderNavigation() {
    lastVisible = visibleItems();
    $('case-select').replaceChildren(...lastVisible.map(item => {
      const option = document.createElement('option'), r = recordFor(item.id);
      option.value = item.id; option.textContent = `${r?.completed ? '✓ ' : ''}${item.number}. ${item.image_id} — ${item.style_label}`; return option;
    }));
    if (!lastVisible.some(item => item.id === currentId) && lastVisible.length) currentId = lastVisible[0].id;
    $('case-select').value = currentId;
    $('filter-count').textContent = `${lastVisible.length} griglie con questi filtri`;
    const index = lastVisible.findIndex(item => item.id === currentId);
    $('previous').disabled = index <= 0; $('next').disabled = index < 0 || index >= lastVisible.length - 1;
    $('review-card').hidden = !lastVisible.length; $('no-cases').hidden = !!lastVisible.length;
  }
  function render() {refreshProgress(); renderNavigation(); if (lastVisible.length) renderCard(); savePrefs();}
  function setReviewer() {
    const name = C.normalizeName($('reviewer').value);
    if (!name) return message('Scrivi un nome o pseudonimo per distinguere le tue risposte.', true);
    reviewer = name; $('reviewer').value = name; render();
    message('Risposte aperte per ' + name + '. Il testo si salva nel browser; esporta il JSON a fine sessione.');
  }
  $('apply-reviewer').addEventListener('click', setReviewer);
  $('reviewer').addEventListener('keydown', e => {if (e.key === 'Enter') setReviewer();});
  for (const f of fields) f.addEventListener('input', () => {
    try {
      const r = activeRecord(); r.answers[f.dataset.question] = f.value; r.completed = false; r.updated_at = now();
      save(); refreshProgress(); refreshCardStatus();
    } catch (e) {message(e.message, true);}
  });
  for (const id of ['part', 'style', 'cohort', 'status']) $(id).addEventListener('change', render);
  $('case-select').addEventListener('change', () => {currentId = $('case-select').value; render();});
  for (const [id, step] of [['previous', -1], ['next', 1]]) $(id).addEventListener('click', () => {
    const index = lastVisible.findIndex(item => item.id === currentId), next = lastVisible[index + step];
    if (next) {currentId = next.id; render(); $('case-title').scrollIntoView({behavior: 'smooth', block: 'start'});}
  });
  function mark(completed) {
    try {
      const r = activeRecord();
      if (completed && C.QUESTIONS.some(q => !r.answers[q].trim())) return message('Per completare la griglia rispondi a tutte e quattro le domande. Va bene anche dichiarare una parità, un dubbio o nessun difetto evidente.', true);
      r.completed = completed; r.updated_at = now(); save(); render();
      message(completed ? 'Griglia completata. Puoi passare alla successiva.' : 'Griglia conservata in bozza.');
    } catch (e) {message(e.message, true);}
  }
  $('mark-complete').addEventListener('click', () => mark(true));
  $('mark-draft').addEventListener('click', () => mark(false));
  function download(text, filename, type) {
    const url = URL.createObjectURL(new Blob([text], {type})), link = document.createElement('a');
    link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function fileStamp() {return now().replace(/[:.]/g, '-');}
  $('export-json').addEventListener('click', () => {
    const payload = {...state, exported_at: now()};
    if (damagedStorage) payload.recovery_raw_storage = damagedStorage;
    download(JSON.stringify(payload, null, 2), `rq1_risposte_${(reviewer || 'gruppo').replace(/[^\p{L}\p{N}_-]/gu, '_')}_${fileStamp()}.json`, 'application/json;charset=utf-8');
    message('Download JSON avviato: controlla che il file sia presente nei download. Include tutte le risposte caricate e le eventuali versioni alternative.');
  });
  $('export-csv').addEventListener('click', () => {
    download(C.toCSV(state), `rq1_risposte_${fileStamp()}.csv`, 'text/csv;charset=utf-8');
    message('Download CSV avviato. Per riunire lavori o conservare conflitti usa anche il JSON.');
  });
  $('import-button').addEventListener('click', () => $('import-files').click());
  $('import-files').addEventListener('change', async () => {
    try {
      const files = Array.from($('import-files').files);
      if (!files.length) return;
      if (files.some(f => f.size > 20 * 1024 * 1024)) throw Error('File oltre 20 MB: seleziona soltanto i JSON delle risposte.');
      const payloads = await Promise.all(files.map(async file => JSON.parse(await file.text())));
      const result = C.merge(state, payloads, config); state = result.state; save(); render(); renderConflicts();
      message(`Importazione: ${result.added} risposte aggiunte, ${result.duplicates} già presenti, ${result.conflicts} nuovi conflitti. I revisori restano separati.`);
    } catch (e) {message('Importazione non completata, risposte correnti conservate: ' + e.message, true);}
    finally {$('import-files').value = '';}
  });
  function renderConflicts() {
    const pending = state.alternatives.filter(a => a.status === 'pending');
    $('conflicts-section').hidden = !pending.length; $('conflicts').replaceChildren();
    for (const alternative of pending) {
      const box = document.createElement('div'); box.className = 'conflict';
      const item = config.items.find(i => i.id === alternative.incoming.review_id);
      box.append(makeText('h3', `${alternative.incoming.reviewer} · ${item.image_id} · ${item.style_label}`));
      const versions = document.createElement('div'); versions.className = 'versions';
      for (const [which, title] of [['current', 'Versione già caricata'], ['incoming', 'Versione importata']]) {
        const record = which === 'current' ? state.records.find(r => C.key(r) === C.key(alternative.incoming)) || alternative.current : alternative.incoming;
        const pane = document.createElement('div'); pane.append(makeText('strong', title));
        pane.append(makeText('pre', `${record.completed ? 'Completata' : 'Bozza'} · ${record.updated_at}\n\n` + C.QUESTIONS.map(q => labels[q] + ':\n' + record.answers[q]).join('\n\n')));
        const button = makeText('button', which === 'current' ? 'Usa la versione già caricata' : 'Usa la versione importata');
        button.addEventListener('click', () => {
          state = C.resolve(state, alternative.id, which, config); save(); render(); renderConflicts();
          message('Versione attiva scelta. Entrambe le versioni restano conservate nel JSON.');
        });
        pane.append(button); versions.append(pane);
      }
      box.append(versions); $('conflicts').append(box);
    }
  }
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw) {try {state = C.validate(JSON.parse(raw), config);} catch (e) {damagedStorage = raw; blockedStorage = true; throw e;}}
    const prefs = JSON.parse(localStorage.getItem(prefsKey) || '{}');
    reviewer = C.normalizeName(prefs.reviewer); $('reviewer').value = reviewer;
    if (config.items.some(item => item.id === prefs.currentId)) currentId = prefs.currentId;
    for (const id of ['part', 'style', 'cohort', 'status']) {
      if (prefs.filters && Array.from($(id).options).some(o => o.value === prefs.filters[id])) $(id).value = prefs.filters[id];
    }
    $('save-state').textContent = raw ? 'Risposte precedenti ripristinate dal browser' : 'Salvataggio automatico pronto';
  } catch (_) {
    $('storage-warning').hidden = false;
    if (damagedStorage) message('Archivio locale non leggibile: non lo sovrascriviamo. Il prossimo export JSON includerà anche il contenuto originale per recuperarlo.', true);
  }
  window.addEventListener('storage', event => {
    if (event.key !== storageKey || !event.newValue) return;
    try {
      const result = C.merge(state, [JSON.parse(event.newValue)], config);
      if (JSON.stringify(result.state) !== JSON.stringify(state)) {state = result.state; save(); render(); renderConflicts();}
    } catch (e) {message('Non posso riunire le modifiche da un’altra scheda: ' + e.message, true);}
  });
  render(); renderConflicts();
})();
