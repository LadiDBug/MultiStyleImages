/* Eseguibile con Node: require('./tests/test_review_core.cjs')(core, assert). */
module.exports = function runCoreTests(C, assert) {
  const config = {dataset_id: 'test-gallery', items: [
    {id:'grid-a', image_id:'a.png', style:'gta5', seed:42},
    {id:'grid-b', image_id:'b.png', style:'pixel_art', seed:42}
  ]};
  const record = (reviewer = 'Diana', id = 'grid-a', text = 'Osservazione') => {
    const item = config.items.find(i => i.id === id);
    return {review_id:id,image_id:item.image_id,style:item.style,seed:item.seed,reviewer,
      completed:true,updated_at:'2026-11-12T12:00:00Z',
      answers:Object.fromEntries(C.QUESTIONS.map(q => [q,text]))};
  };
  const payload = (...records) => ({...C.empty(config.dataset_id), records});
  assert.equal(C.validate(C.empty(config.dataset_id), config).records.length, 0);
  let result = C.merge(payload(record()), [payload(record('Luca'), record('Diana','grid-b'))], config);
  assert.equal(result.state.records.length, 3, 'Revisori e griglie diversi devono restare separati');
  assert.equal(result.conflicts, 0);
  result = C.merge(payload(record()), [payload({...record(), updated_at:'2026-11-13T12:00:00Z'})], config);
  assert.equal(result.duplicates, 1, 'Stesso testo con data diversa non crea duplicati');
  result = C.merge(payload(record()), [payload(record('Diana','grid-a','Versione nuova'))], config);
  assert.equal(result.conflicts, 1);
  assert.equal(result.state.records[0].answers.content,'Osservazione','Nessuna sovrascrittura automatica');
  assert.equal(result.state.alternatives[0].incoming.answers.content,'Versione nuova');
  const roundTrip = C.validate(JSON.parse(JSON.stringify(result.state)), config);
  assert.equal(roundTrip.alternatives.length,1,'Il JSON deve conservare le versioni alternative');
  const accepted = C.resolve(roundTrip, roundTrip.alternatives[0].id, 'incoming', config);
  assert.equal(accepted.records[0].answers.content,'Versione nuova');
  assert.equal(accepted.alternatives[0].current.answers.content,'Osservazione');
  assert.equal(accepted.alternatives[0].status,'accepted_incoming');
  const kept = C.resolve(roundTrip, roundTrip.alternatives[0].id, 'current', config);
  assert.equal(kept.records[0].answers.content,'Osservazione');
  assert.equal(kept.alternatives[0].incoming.answers.content,'Versione nuova');
  const original = payload(record()), before = JSON.stringify(original);
  assert.throws(() => C.merge(original,[payload(record('Luca')),{...payload(),dataset_id:'other'}],config));
  assert.equal(JSON.stringify(original),before,'Importazione fallita atomica, dati originali intatti');
  assert.throws(() => C.validate(payload(record('Diana'),record('  Diana ')),config));
  assert.throws(() => C.validate(payload({...record(),answers:{...record().answers,style:''}}),config));
  assert.throws(() => C.validate(payload({...record(),reviewer:42}),config));
  assert.throws(() => C.validate({...payload(),version:2},config));
  assert.throws(() => C.validate(payload({...record(),review_id:'unknown'}),config));
  const csv = C.toCSV(payload(record('Diana','grid-a','=TEST("x")\nseconda riga')));
  assert.ok(csv.includes("'=TEST(")); assert.ok(csv.includes('""x""')); assert.ok(csv.includes('\nseconda riga'));
  return 'Core: merge revisori/griglie, duplicati, conflitti conservati e risolti, round-trip JSON, importazione atomica, validazione e CSV verificati.';
};
