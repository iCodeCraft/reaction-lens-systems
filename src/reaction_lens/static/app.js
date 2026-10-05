'use strict';
const $ = id => document.getElementById(id);
let result = null, page = 0;
const pageSize = 10;
function text(tag, value, cls) { const e=document.createElement(tag); e.textContent=value; if(cls)e.className=cls; return e; }
function render() {
  if (!result) return;
  const needle=$('filter').value.toLowerCase();
  const rows=result.options.filter(r=>(!$('selected-only').checked||r.selected)&&(!needle||(r.reaction_id+' '+r.description).toLowerCase().includes(needle))).sort((a,b)=>a.rank-b.rank);
  const pages=Math.max(1,Math.ceil(rows.length/pageSize)); page=Math.min(page,pages-1);
  $('rows').replaceChildren();
  for(const r of rows.slice(page*pageSize,(page+1)*pageSize)) {
    const item=text('article','','reaction'), head=text('div','','reaction-head');
    head.append(text('span',String(r.rank).padStart(2,'0'),'rank'),text('span',r.reaction_id,'reaction-id'));
    if(r.selected)head.append(text('span','SELECTED','badge'));
    item.append(head,text('p',r.description),text('div',`Raw logit ${r.logit.toFixed(3)} · threshold ${result.raw_logit_threshold.toFixed(3)}`,'score'));
    $('rows').append(item);
  }
  if(!rows.length)$('rows').append(text('p','No reactions match this view.','fine'));
  $('page').textContent=`${rows.length.toLocaleString()} results · ${page+1} / ${pages}`;
  $('prev').disabled=page===0; $('next').disabled=page===pages-1;
}
function download(contents,type,name){const u=URL.createObjectURL(new Blob([contents],{type}));const a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
$('query').addEventListener('submit',async e=>{
 e.preventDefault();setEditingDisabled(true);$('error').hidden=true;$('run').disabled=true;$('run').textContent='Scoring every reaction…';
 const body={title:$('title').value,abstract:$('abstract').value};
 if($('threshold').value!=='')body.threshold=Number($('threshold').value);
 try {
  const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json(); if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Check the input and try again.');
  result=data;page=0;$('empty').hidden=true;$('results').hidden=false;
  $('scored').textContent=data.options_scored.toLocaleString();$('selected').textContent=data.proposed_reaction_ids.length.toLocaleString();$('elapsed').textContent=data.timings.request_seconds.toFixed(2)+'s';render();
 }catch(err){$('error').textContent=err.message;$('error').hidden=false;}
 finally{setEditingDisabled(false);$('run').disabled=false;$('run').textContent='Score the full catalog ↗';}
});
$('abstract').addEventListener('input',()=>{$('count').textContent=$('abstract').value.length.toLocaleString()+' characters';});
const examples = [
 {id:'mapk', category:'SIGNALING', name:'MAP kinase cascade', summary:'Follow RAF → MEK → ERK.', title:'Growth factor signaling through MAP kinase', abstract:'Growth factor stimulation activates receptor tyrosine kinases and downstream RAS signaling. RAF phosphorylates MEK, which activates ERK. Activated ERK regulates downstream proteins involved in cell proliferation.', note:'Inspect whether the ranking distinguishes the individual phosphorylation steps. Try removing the RAF sentence and compare the new ranking.'},
 {id:'pi3k-akt', category:'SIGNALING', name:'PI3K and AKT', summary:'Lipid signaling at the membrane.', title:'PI3K-dependent activation of AKT', abstract:'Following growth factor stimulation, PI3K converts phosphatidylinositol 4,5-bisphosphate into phosphatidylinositol 3,4,5-trisphosphate at the plasma membrane. AKT is recruited to the membrane through its pleckstrin homology domain. PDK1 phosphorylates AKT at Thr308.', note:'Look for reactions involving lipid conversion, membrane recruitment and AKT phosphorylation. Similar pathway names alone do not establish that a specific reaction is supported.'},
 {id:'apoptosis', category:'CELL DEATH', name:'Caspase activation', summary:'From cytochrome c to a protease cascade.', title:'Mitochondrial activation of the caspase cascade', abstract:'Cytochrome c released from mitochondria associates with APAF1 to promote apoptosome assembly. The apoptosome recruits procaspase-9 and promotes its activation. Caspase-9 subsequently cleaves and activates executioner caspases.', note:'Compare assembly reactions with proteolytic activation. The text names several steps, so several relevant proposals may appear.'},
 {id:'dna-repair', category:'DNA REPAIR', name:'Repairing a damaged base', summary:'A sequence of repair operations.', title:'Base excision repair of a damaged DNA base', abstract:'A DNA glycosylase removes a damaged base, leaving an abasic site. An AP endonuclease cleaves the DNA backbone at this site. DNA polymerase fills the resulting gap, and a DNA ligase seals the remaining nick.', note:'Inspect how the model handles a mechanism described by enzyme classes rather than named proteins. The text may be too broad to distinguish specific reaction variants.'},
 {id:'negation', category:'LIMITATION CHECK', name:'Same entities, opposite claim', summary:'A contrast to the MAPK example.', title:'No detected activation of the MAP kinase cascade', abstract:'We measured RAF, MEK and ERK after growth factor stimulation. We did not detect increased MEK phosphorylation or ERK activation. These observations do not support activation of the RAF–MEK–ERK cascade under the conditions examined.', note:'Compare this with the MAPK example. Negation handling has not been validated; related reactions may still rank highly. This checks a limitation, not an expected pass condition.'},
 {id:'off-topic', category:'LIMITATION CHECK', name:'Outside the biological domain', summary:'What happens with an astronomy text?', title:'Measuring the orbit of a distant planet', abstract:'Repeated observations of a distant star revealed periodic changes in its apparent brightness. A transit model was fitted to estimate the orbital period and the size of a candidate planet. Additional observations are needed to constrain the orbit.', note:'Inspect reaction ranks and score ranges for this illustrative astronomy text. No reviewed reaction labels are available for it.'}
];
function clearResults() {
 result=null;page=0;$('results').hidden=true;$('empty').hidden=false;
 $('filter').value='';$('selected-only').checked=false;$('error').hidden=true;
}
function markExample(id) {
 for(const button of document.querySelectorAll('.example-card'))button.setAttribute('aria-pressed',String(button.dataset.example===id));
}
function loadExample(id, scroll=true) {
 const item=examples.find(x=>x.id===id);if(!item)return;
 clearResults();$('title').value=item.title;$('abstract').value=item.abstract;$('threshold').value='';
 $('count').textContent=item.abstract.length.toLocaleString()+' characters';
 $('example-name').textContent=item.name;$('example-note').textContent=item.note;
 $('example-context').hidden=false;$('copy-status').textContent='';markExample(id);
 history.replaceState(null,'','#example='+encodeURIComponent(id));
 if(scroll){$('query').scrollIntoView({behavior:'smooth',block:'start'});$('abstract').focus({preventScroll:true});}
}
for(const item of examples) {
 const button=text('button','','example-card');button.type='button';button.dataset.example=item.id;button.setAttribute('aria-pressed','false');
 button.append(text('span',item.category,'example-category'),text('strong',item.name),text('span',item.summary,'example-summary'),text('span','Load text ↗','example-action'));
 button.addEventListener('click',()=>loadExample(item.id));$('example-grid').append(button);
}
$('example').onclick=()=>loadExample('mapk');
$('copy-example').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$('copy-status').textContent='Link copied';}catch{$('copy-status').textContent='Copy this page’s URL from the address bar.';}};
for(const id of ['title','abstract','threshold'])$(id).addEventListener('input',()=>{
 clearResults();
 if(id!=='threshold'){$('example-context').hidden=true;markExample(null);history.replaceState(null,'',location.pathname+location.search);}
});
const initialExample=new URLSearchParams(location.hash.slice(1)).get('example');
if(initialExample)loadExample(initialExample,false);
window.addEventListener('hashchange',()=>{const id=new URLSearchParams(location.hash.slice(1)).get('example');if(id)loadExample(id,false);});
function setEditingDisabled(disabled) {
 for(const e of document.querySelectorAll('#title,#abstract,#threshold,#example,.example-card'))e.disabled=disabled;
}
$('prev').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};
for(const id of ['filter','selected-only'])$(id).addEventListener('input',()=>{page=0;render();});
$('json').onclick=()=>download(JSON.stringify(result,null,2)+'\n','application/json','reaction-lens.json');
$('csv').onclick=()=>{
 const keys=['rank','reaction_id','description','logit','sigmoid_score','selected'];
 const cell=v=>{let s=String(v);if(typeof v==='string'&&/^[\s]*[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
 const lines=[keys.join(','),...result.options.slice().sort((a,b)=>a.rank-b.rank).map(r=>keys.map(k=>cell(r[k])).join(','))];
 download(lines.join('\r\n')+'\r\n','text/csv;charset=utf-8','reaction-lens.csv');
};
fetch('/api/model').then(async r=>{const data=await r.json();if(!r.ok)throw new Error(data.detail);$('status').textContent=`Local model ready · ${data.catalog_size.toLocaleString()} reactions · ${data.device.toUpperCase()}`;$('dot').style.background='#258879';$('identity').textContent=JSON.stringify(data,null,2);$('run').disabled=false;}).catch(e=>{$('status').textContent='Model setup required';$('identity').textContent=e.message;$('error').textContent=e.message;$('error').hidden=false;});

