/* Persistenza e fusione delle risposte. Nessun accesso alla rete. */
(function (root) {
  'use strict';
  const FORMAT = 'rq1-grid-review';
  const VERSION = 1;
  const QUESTIONS = ['content', 'style', 'artifacts', 'tradeoff'];
  const normalizeName = name => String(name || '').normalize('NFKC').trim().replace(/\s+/g, ' ');
  const key = record => JSON.stringify([record.review_id, normalizeName(record.reviewer)]);
  const signature = record => JSON.stringify([record.completed, ...QUESTIONS.map(q => record.answers[q])]);
  const clone = value => JSON.parse(JSON.stringify(value));
  function empty(dataset) {
    return {format: FORMAT, version: VERSION, questionnaire_version: 1, dataset_id: dataset,
      records: [], alternatives: []};
  }
  function validateRecord(record, items) {
    if (!record || typeof record !== 'object' || !items.has(record.review_id)) throw Error('Griglia sconosciuta nel file.');
    const item = items.get(record.review_id);
    if (typeof record.reviewer !== 'string' || !normalizeName(record.reviewer) || record.reviewer.length > 120) throw Error('Nome revisore mancante o troppo lungo.');
    if (record.image_id !== item.image_id || record.style !== item.style || record.seed !== item.seed) throw Error('Foto/stile/seed non corrispondono alla griglia.');
    if (typeof record.completed !== 'boolean' || typeof record.updated_at !== 'string' || !Number.isFinite(Date.parse(record.updated_at))) throw Error('Stato o data della risposta non validi.');
    if (!record.answers || QUESTIONS.some(q => typeof record.answers[q] !== 'string' || record.answers[q].length > 20000)) throw Error('Risposte mancanti o troppo lunghe.');
    if (record.completed && QUESTIONS.some(q => !record.answers[q].trim())) throw Error('Una scheda completa contiene risposte vuote.');
    return {...record, reviewer: normalizeName(record.reviewer),
      answers: Object.fromEntries(QUESTIONS.map(q => [q, record.answers[q]]))};
  }
  function validate(payload, config) {
    if (!payload || payload.format !== FORMAT || payload.version !== VERSION || payload.questionnaire_version !== 1) throw Error('Formato delle risposte non supportato. Usa il JSON esportato dalla galleria.');
    if (payload.dataset_id !== config.dataset_id) throw Error('Il file appartiene a una galleria/selezione diversa.');
    if (!Array.isArray(payload.records) || !Array.isArray(payload.alternatives)) throw Error('Elenco delle risposte non valido.');
    if (payload.records.length > 10000 || payload.alternatives.length > 10000) throw Error('File con troppe risposte.');
    const items = new Map(config.items.map(item => [item.id, item]));
    const result = empty(config.dataset_id), seen = new Set(), seenAlternatives = new Set();
    for (const record of payload.records) {
      const valid = validateRecord(record, items), id = key(valid);
      if (seen.has(id)) throw Error('Risposte duplicate per la stessa griglia e lo stesso revisore.');
      seen.add(id); result.records.push(valid);
    }
    for (const alternative of payload.alternatives) {
      if (!alternative || !['pending', 'kept_current', 'accepted_incoming'].includes(alternative.status)) throw Error('Conflitto non valido.');
      const current = validateRecord(alternative.current, items), incoming = validateRecord(alternative.incoming, items);
      if (key(current) !== key(incoming)) throw Error('Le versioni del conflitto hanno revisori o griglie differenti.');
      const valid = {id: JSON.stringify([key(incoming), signature(incoming), signature(current)]),
        current, incoming, status: alternative.status};
      if (!seenAlternatives.has(valid.id)) {result.alternatives.push(valid); seenAlternatives.add(valid.id);}
    }
    return result;
  }
  function merge(current, payloads, config) {
    // Validazione atomica: nessuna modifica se uno dei file è incompatibile/corrotto.
    const base = validate(current, config), imported = payloads.map(p => validate(p, config));
    const result = clone(base), records = new Map(result.records.map(r => [key(r), r]));
    let added = 0, duplicates = 0, conflicts = 0;
    const addAlternative = candidate => {
      if (!result.alternatives.some(a => a.id === candidate.id)) {result.alternatives.push(candidate); return true;}
      return false;
    };
    for (const payload of imported) {
      for (const incoming of payload.records) {
        const id = key(incoming), existing = records.get(id);
        if (!existing) {records.set(id, incoming); added++;}
        else if (signature(existing) === signature(incoming)) duplicates++;
        else {
          const candidate = {id: JSON.stringify([id, signature(incoming), signature(existing)]),
            current: clone(existing), incoming: clone(incoming), status: 'pending'};
          if (addAlternative(candidate)) conflicts++;
        }
      }
      for (const alternative of payload.alternatives) addAlternative(alternative);
    }
    result.records = Array.from(records.values());
    return {state: result, added, duplicates, conflicts};
  }
  function resolve(state, alternativeId, choice, config) {
    if (!['current', 'incoming'].includes(choice)) throw Error('Scelta non valida.');
    const result = validate(state, config), alternative = result.alternatives.find(a => a.id === alternativeId);
    if (!alternative || alternative.status !== 'pending') throw Error('Conflitto già risolto o assente.');
    const index = result.records.findIndex(r => key(r) === key(alternative.incoming));
    if (choice === 'incoming') {
      // Conservare anche la versione attiva al momento della scelta, se è stata modificata nel frattempo.
      if (index >= 0) alternative.current = clone(result.records[index]);
      if (index >= 0) result.records[index] = clone(alternative.incoming);
      else result.records.push(clone(alternative.incoming));
    }
    alternative.status = choice === 'incoming' ? 'accepted_incoming' : 'kept_current';
    return result;
  }
  function toCSV(state) {
    const fields = ['image_id', 'style', 'seed', 'reviewer', 'completed', 'updated_at', ...QUESTIONS];
    const cell = value => {
      let text = String(value ?? '');
      if (/^[\s]*[=+\-@]/.test(text)) text = "'" + text;
      return '"' + text.replace(/"/g, '""') + '"';
    };
    return '\uFEFF' + [fields, ...state.records.map(r => fields.map(f => QUESTIONS.includes(f) ? r.answers[f] : r[f]))]
      .map(row => row.map(cell).join(',')).join('\r\n');
  }
  const api = {FORMAT, VERSION, QUESTIONS, normalizeName, key, signature, empty, validate, merge, resolve, toCSV};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.RQ1Review = api;
})(typeof globalThis === 'undefined' ? this : globalThis);
