/* Test dell'interfaccia in un DOM minimo isolato: nessun browser dell'utente. */
module.exports = async function runAppTests(coreSource, appSource, config, vm, assert) {
  class Element {
    constructor(tag='div'){this.tagName=tag;this.children=[];this.listeners={};this.value='';this.textContent='';this.dataset={};this.style={};this.hidden=false;this.disabled=false;this.files=[];}
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=[...nodes];}
    get options(){return this.children;}
    addEventListener(event,fn){(this.listeners[event] ||= []).push(fn);}
    async emit(event,data={}){for(const fn of this.listeners[event] || []) await fn(data);}
    click(){return this.emit('click');} remove(){} scrollIntoView(){}
  }
  const storage=new Map(), downloads=[];
  function setup(failStorage=false){
    const ids=['review-config','reviewer','reviewers','apply-reviewer','active-reviewer','progress-text','team-progress','progress-bar','save-state','storage-warning','export-json','export-csv','import-button','import-files','part','style','cohort','status','case-select','filter-count','previous','next','review-card','no-cases','case-title','case-status','case-reason','case-cohort','grid-link','grid-image','source-links','style-hint','mark-complete','mark-draft','record-save','other-reviewers-title','other-reviewers','conflicts-section','conflicts','message'];
    const nodes=Object.fromEntries(ids.map(id=>[id,new Element()]));
    nodes['review-config'].textContent=JSON.stringify(config);
    const options={part:['all','A','B'],style:['all','gta5','pixel_art','watercolor'],cohort:['all','random','diagnostic'],status:['all','todo','done']};
    for(const [id,values] of Object.entries(options)){nodes[id].value='all';nodes[id].children=values.map(v=>{const e=new Element('option');e.value=v;return e;});}
    const fields=['content','style','artifacts','tradeoff'].map(q=>{const e=new Element('textarea');e.dataset.question=q;return e;});
    const document={getElementById:id=>nodes[id],querySelectorAll:()=>fields,createElement:tag=>new Element(tag),
      createTextNode:text=>({textContent:text}),body:new Element('body')};
    const sandbox={document,console,localStorage:{getItem:k=>{if(failStorage)throw Error('blocked');return storage.get(k)||null;},setItem:(k,v)=>{if(failStorage)throw Error('blocked');storage.set(k,v);}},
      Blob:class{constructor(parts){this.text=parts.join('');}},URL:{createObjectURL:b=>{downloads.push(b.text);return 'blob:test';},revokeObjectURL:()=>{}},setTimeout:()=>{},addEventListener:()=>{}};
    sandbox.window=sandbox;vm.createContext(sandbox);vm.runInContext(coreSource,sandbox);vm.runInContext(appSource,sandbox);
    return {nodes,fields,sandbox};
  }
  let app=setup();assert.ok(app.fields.every(f=>f.disabled),'Campi disabilitati prima del nome');
  app.nodes.reviewer.value='Diana';await app.nodes['apply-reviewer'].click();
  assert.ok(app.fields.every(f=>!f.disabled));
  app.fields[0].value='Contenuto verificato';await app.fields[0].emit('input');
  const storageKey='rq1-review:'+config.dataset_id;
  assert.equal(JSON.parse(storage.get(storageKey)).records[0].answers.content,'Contenuto verificato');
  await app.nodes['mark-complete'].click();
  assert.equal(JSON.parse(storage.get(storageKey)).records[0].completed,false,'Richieste quattro risposte');
  for(let i=1;i<4;i++){app.fields[i].value='Risposta '+i;await app.fields[i].emit('input');}
  await app.nodes['mark-complete'].click();assert.equal(JSON.parse(storage.get(storageKey)).records[0].completed,true);
  app=setup();assert.equal(app.nodes.reviewer.value,'Diana');assert.equal(app.fields[0].value,'Contenuto verificato','Ripristino dopo reload');
  app.fields[0].value='Modifica dopo completamento';await app.fields[0].emit('input');
  assert.equal(JSON.parse(storage.get(storageKey)).records[0].completed,false,'Le modifiche riaprono la bozza');
  app.nodes.part.value='A';await app.nodes.part.emit('change');assert.equal(app.nodes['case-select'].children.length,15);
  app.nodes.part.value='B';await app.nodes.part.emit('change');assert.equal(app.nodes['case-select'].children.length,15);
  const item=config.items[1], incoming={format:'rq1-grid-review',version:1,questionnaire_version:1,dataset_id:config.dataset_id,alternatives:[],records:[{
    review_id:item.id,image_id:item.image_id,style:item.style,seed:item.seed,reviewer:'Luca',completed:true,
    updated_at:'2026-11-12T12:00:00Z',answers:{content:'Contenuto Luca',style:'Stile Luca',artifacts:'Artefatti Luca',tradeoff:'Scelta Luca'}}]};
  app.nodes['import-files'].files=[{size:1000,text:async()=>JSON.stringify(incoming)}];await app.nodes['import-files'].emit('change');
  assert.equal(JSON.parse(storage.get(storageKey)).records.length,2);
  await app.nodes['export-json'].click();assert.equal(JSON.parse(downloads.at(-1)).records.length,2,'Export comprende i due revisori');
  app.nodes['export-csv'].click();assert.ok(downloads.at(-1).includes('Contenuto Luca'));
  const before=storage.get(storageKey);
  app.nodes['import-files'].files=[{size:1000,text:async()=>JSON.stringify({...incoming,dataset_id:'wrong'})}];await app.nodes['import-files'].emit('change');
  assert.equal(storage.get(storageKey),before,'Import incompatibile non altera i dati');
  const denied=setup(true);denied.nodes.reviewer.value='Test';await denied.nodes['apply-reviewer'].click();
  denied.fields[0].value='In memoria';await denied.fields[0].emit('input');assert.equal(denied.nodes['storage-warning'].hidden,false);
  await denied.nodes['export-json'].click();assert.equal(JSON.parse(downloads.at(-1)).records[0].answers.content,'In memoria');
  return 'UI: nome richiesto, autosave/reload, quattro risposte obbligatorie, riapertura bozza, parti 15+15, import/export, file incompatibili e fallback senza localStorage verificati.';
};
