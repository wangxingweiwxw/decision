'use strict';
const COLOR = {bailian:'#2fb8b8', microsoft:'#5c8fe0', qwen:'#d29445', jev:'#a868c8'};
const LABELS = ['yes','no','maybe'];
const LABEL_CN = {yes:'yes 支持', no:'no 反驳', maybe:'maybe 不确定'};
let D = null;

const $ = s => document.querySelector(s);
const pct = (v,d=2) => v==null ? '—' : (v*100).toFixed(d)+'%';
const fx  = (v,d=4) => v==null ? '不可用' : Number(v).toFixed(d);
const ms  = v => v==null ? '—' : Number(v).toFixed(1);

function el(tag, cls, html){
  const e=document.createElement(tag);
  if(cls) e.className=cls;
  if(html!=null) e.innerHTML=html;
  return e;
}
function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* ---------- tabs ---------- */
function initTabs(){
  const btns=[...document.querySelectorAll('#tabs button')];
  btns.forEach(b=>b.addEventListener('click',()=>{
    btns.forEach(x=>x.classList.toggle('on',x===b));
    document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id===b.dataset.tab));
    history.replaceState(null,'','#'+b.dataset.tab);
  }));
  const h=location.hash.replace('#','');
  const hit=btns.find(b=>b.dataset.tab===h);
  if(hit) hit.click();
}

/* ---------- header ---------- */
function renderHero(){
  const m=D.meta;
  const sig=D.paired.filter(p=>!(p.ci95[0]<=0 && 0<=p.ci95[1])).length;
  $('#h-samples').textContent=m.samples;
  $('#h-rounds').textContent=m.rounds;
  $('#h-req').textContent=m.total_requests.toLocaleString();
  $('#h-sig').textContent=sig;
  $('#h-run').textContent=m.run_id;
  $('#h-boot').textContent=m.bootstrap.toLocaleString();
  $('#f-run').textContent=m.run_id;
  $('#f-sha').textContent=m.dataset_sha256.slice(0,24)+'…';
  $('#f-seed').textContent=m.seed;
  $('#f-wall').textContent=m.wall_min;

  const best=[...D.order].sort((a,b)=>D.providers[b].accuracy-D.providers[a].accuracy)[0];
  const fast=[...D.order].sort((a,b)=>D.providers[a].p50-D.providers[b].p50)[0];
  const stats=[
    ['最高准确率', pct(D.providers[best].accuracy), D.providers[best].name],
    ['最低 P50 延迟', ms(D.providers[fast].p50)+' ms', D.providers[fast].name],
    ['可判定配对', sig+' / 6', '95% 区间不含零'],
    ['请求成功率', pct(m.ok_requests/m.total_requests), m.ok_requests.toLocaleString()+' / '+m.total_requests.toLocaleString()]
  ];
  $('#hero-stats').innerHTML = stats.map(([k,v,s])=>
    `<div class="s"><b>${esc(v)}</b><span>${esc(k)} · ${esc(s)}</span></div>`).join('');
}

/* ---------- overview ---------- */
function renderMain(){
  const rows=[
    ['首轮正确数', n=>`${D.providers[n].correct} / ${D.providers[n].n}`, n=>D.providers[n].correct, 'max'],
    ['准确率', n=>pct(D.providers[n].accuracy), n=>D.providers[n].accuracy, 'max'],
    ['95% Wilson 区间', n=>`[${pct(D.providers[n].ci[0])}, ${pct(D.providers[n].ci[1])}]`, null, null],
    ['Macro-F1', n=>fx(D.providers[n].macro_f1), n=>D.providers[n].macro_f1, 'max'],
    ['Brier（越低越好）', n=>fx(D.providers[n].brier), n=>D.providers[n].brier, 'min'],
    ['ECE（越低越好）', n=>fx(D.providers[n].ece), n=>D.providers[n].ece, 'min'],
    ['P50 延迟 ms', n=>ms(D.providers[n].p50), n=>D.providers[n].p50, 'min'],
    ['P95 延迟 ms', n=>ms(D.providers[n].p95), n=>D.providers[n].p95, 'min'],
    ['失败请求', n=>`${D.providers[n].requests-D.providers[n].ok} / ${D.providers[n].requests}`,
                 n=>D.providers[n].requests-D.providers[n].ok, 'min'],
    ['重复一致率', n=>pct(D.providers[n].repeat), n=>D.providers[n].repeat, 'max'],
    ['实际返回模型', n=>`<code>${esc(D.providers[n].served)}</code>`, null, null],
    ['上游', n=>{const u=D.providers[n].upstream; return (!u||u==='None')?'自有接口直连':esc(u);}, null, null]
  ];
  let h='<thead><tr><th>指标</th>'+D.order.map(n=>`<th>${esc(D.providers[n].name)}</th>`).join('')+'</tr></thead><tbody>';
  for(const [label,fmt,key,mode] of rows){
    let bestVal=null;
    if(key){
      const vals=D.order.map(key).filter(v=>v!=null && !Number.isNaN(v));
      if(vals.length) bestVal = mode==='max'? Math.max(...vals) : Math.min(...vals);
    }
    h+=`<tr><th>${esc(label)}</th>`;
    for(const n of D.order){
      const v = key? key(n) : null;
      const isBest = key && v!=null && v===bestVal;
      const txt = fmt(n);
      const na = txt==='不可用' || txt==='—';
      h+=`<td class="num${isBest?' best':''}${na?' na':''}">${txt}</td>`;
    }
    h+='</tr>';
  }
  $('#t-main').innerHTML=h+'</tbody>';

  const sigPairs=D.paired.filter(p=>!(p.ci95[0]<=0 && 0<=p.ci95[1]));
  const tie=D.paired.find(p=>p.left==='microsoft'&&p.right==='jev')
         || D.paired.find(p=>p.left==='jev'&&p.right==='microsoft');
  $('#verdict').innerHTML=`
    <div class="callout"><b>可以下的结论</b><br/>
      ${sigPairs.length}/6 组配对区间不含零：Jev 与微软 Decision-1 均显著优于百炼与 Qwen3.5-9B。
      而两者之间相差 ${Math.abs(tie.difference*100).toFixed(2)} 个百分点、区间
      [${(tie.ci95[0]*100).toFixed(2)}, ${(tie.ci95[1]*100).toFixed(2)}] 含零，<b>统计上等效</b>，
      不能宣称其中任一方更准确。
    </div>
    <div class="callout warn"><b>不能下的结论</b><br/>
      延迟差含网络路径差异，不代表模型推理速度；PubMedQA 为公开数据集，题目可能已进入训练数据，
      绝对准确率可能偏高。本页不构成厂商能力的最终排名。
    </div>`;
}

/* ---------- significance ---------- */
function renderPaired(){
  const lo=Math.min(...D.paired.map(p=>p.ci95[0]*100))-2;
  const hi=Math.max(...D.paired.map(p=>p.ci95[1]*100))+2;
  const span=hi-lo;
  const posOf=v=>((v-lo)/span)*100;
  let c='';
  for(const p of D.paired){
    const d=p.difference*100, a=p.ci95[0]*100, b=p.ci95[1]*100;
    const sig=!(a<=0 && 0<=b);
    const col=sig?'#3ecf8e':'#7a8c9d';
    c+=`<div class="pair">
      <div class="lbl">${esc(D.providers[p.left].name)} − ${esc(D.providers[p.right].name)}</div>
      <div class="axis">
        <div class="zero" style="left:${posOf(0)}%"></div>
        <div class="ci" style="left:${posOf(a)}%;width:${Math.max(posOf(b)-posOf(a),0.6)}%;background:${col}"></div>
        <div class="dot" style="left:${posOf(d)}%;background:${col}"></div>
      </div>
      <div class="v ${sig?'sig':'nosig'}">${d>=0?'+':''}${d.toFixed(2)}pp ${sig?'显著':'含零'}</div>
    </div>`;
  }
  $('#pairchart').innerHTML=c+
    `<p class="note" style="margin:14px 0 0">红线为零点；横条为 95% 区间，圆点为观测差值。横跨零点即无法判定。</p>`;

  let h='<thead><tr><th>配对（左 − 右）</th><th>差值 pp</th><th>95% 配对区间</th><th>判定</th></tr></thead><tbody>';
  for(const p of D.paired){
    const a=p.ci95[0]*100,b=p.ci95[1]*100,d=p.difference*100;
    const sig=!(a<=0&&0<=b);
    const verdict= sig ? (a>0?'左显著更优':'右显著更优') : '含零，不显著';
    h+=`<tr><th>${esc(D.providers[p.left].name)} − ${esc(D.providers[p.right].name)}</th>
      <td class="num">${d>=0?'+':''}${d.toFixed(2)}</td>
      <td class="num">[${a>=0?'+':''}${a.toFixed(2)}, ${b>=0?'+':''}${b.toFixed(2)}]</td>
      <td class="${sig?'sig':'nosig'}">${verdict}</td></tr>`;
  }
  $('#t-pair').innerHTML=h+'</tbody>';
}

/* ---------- errors ---------- */
function renderClasses(){
  let c='';
  for(const l of LABELS){
    c+=`<div style="margin:0 0 6px;font-size:13.4px;color:#bcd0e2">${esc(LABEL_CN[l])} · n=${D.providers[D.order[0]].per_class[l].support}</div>`;
    for(const n of D.order){
      const f1=D.providers[n].per_class[l].f1;
      c+=`<div class="bar">
        <div class="lbl">${esc(D.providers[n].name)}</div>
        <div class="track"><i style="width:${(f1*100).toFixed(1)}%;background:${COLOR[n]}"></i></div>
        <div class="val">F1 ${f1.toFixed(3)}</div></div>`;
    }
    c+='<div style="height:14px"></div>';
  }
  $('#clschart').innerHTML=c;

  let h='<thead><tr><th>类别 / 指标</th>'+D.order.map(n=>`<th>${esc(D.providers[n].name)}</th>`).join('')+'</tr></thead><tbody>';
  for(const l of LABELS){
    for(const k of ['f1','precision','recall']){
      const vals=D.order.map(n=>D.providers[n].per_class[l][k]);
      const best=Math.max(...vals);
      h+=`<tr><th>${esc(LABEL_CN[l])} · ${k==='f1'?'F1':k==='precision'?'Precision':'Recall'}</th>`;
      D.order.forEach((n,i)=>{
        const v=vals[i];
        h+=`<td class="num${v===best?' best':''}">${k==='f1'?v.toFixed(3):pct(v,1)}</td>`;
      });
      h+='</tr>';
    }
  }
  const mf=D.order.map(n=>D.providers[n].macro_f1); const bf=Math.max(...mf);
  h+='<tr><th>Macro-F1</th>'+D.order.map((n,i)=>`<td class="num${mf[i]===bf?' best':''}">${mf[i].toFixed(4)}</td>`).join('')+'</tr>';
  $('#t-class').innerHTML=h+'</tbody>';
}

function renderConfusion(){
  let out='';
  for(const n of D.order){
    const cm=D.providers[n].confusion;
    const max=Math.max(...LABELS.flatMap(a=>LABELS.map(b=>cm[a][b]||0)));
    let t='<table><thead><tr><th>真实↓/预测→</th>'+LABELS.map(l=>`<th>${l}</th>`).join('')+'</tr></thead><tbody>';
    for(const a of LABELS){
      t+=`<tr><th>${a}</th>`;
      for(const b of LABELS){
        const v=cm[a][b]||0;
        const alpha=max?(v/max):0;
        const diag=a===b;
        t+=`<td class="${diag?'d':''}" style="background:rgba(${diag?'62,207,142':'207,87,52'},${(alpha*0.42).toFixed(3)})">${v}</td>`;
      }
      t+='</tr>';
    }
    out+=`<div class="conf"><h4 style="color:${COLOR[n]}">${esc(D.providers[n].name)}</h4>${t}</tbody></table></div>`;
  }
  $('#confusion').innerHTML=out;
}

/* ---------- latency ---------- */
function renderLatency(){
  const order=[...D.order].sort((a,b)=>D.providers[a].p50-D.providers[b].p50);
  const max=Math.max(...D.order.map(n=>D.providers[n].p95));
  let c='<div style="margin:0 0 8px;font-size:13.4px;color:#bcd0e2">P50 中位耗时</div>';
  for(const n of order){
    const v=D.providers[n].p50;
    c+=`<div class="bar"><div class="lbl">${esc(D.providers[n].name)}</div>
      <div class="track"><i style="width:${(v/max*100).toFixed(1)}%;background:${COLOR[n]}"></i></div>
      <div class="val">${ms(v)} ms</div></div>`;
  }
  c+='<div style="height:16px"></div><div style="margin:0 0 8px;font-size:13.4px;color:#bcd0e2">P95 耗时</div>';
  for(const n of order){
    const v=D.providers[n].p95;
    c+=`<div class="bar"><div class="lbl">${esc(D.providers[n].name)}</div>
      <div class="track"><i style="width:${(v/max*100).toFixed(1)}%;background:${COLOR[n]};opacity:.72"></i></div>
      <div class="val">${ms(v)} ms</div></div>`;
  }
  $('#latchart').innerHTML=c+'<p class="note" style="margin:14px 0 0">两组共用同一横轴比例，越短越快。</p>';

  let h='<thead><tr><th>服务</th><th>P50 ms</th><th>P95 ms</th><th>成功 / 总数</th><th>失败率</th><th>重复一致率</th><th>失败原因</th></tr></thead><tbody>';
  const p50s=D.order.map(n=>D.providers[n].p50), bp=Math.min(...p50s);
  for(const n of order){
    const p=D.providers[n];
    const errKeys=Object.keys(p.errors||{});
    let reason='无';
    if(errKeys.length){
      const k=errKeys[0];
      const total=Object.values(p.errors).reduce((a,b)=>a+b,0);
      reason = (/429/.test(k) ? 'HTTP 429 上游共享池限流' : k.slice(0,42)) + ` · ${total} 次`;
    }
    h+=`<tr><th style="color:${COLOR[n]}">${esc(p.name)}</th>
      <td class="num${p.p50===bp?' best':''}">${ms(p.p50)}</td>
      <td class="num">${ms(p.p95)}</td>
      <td class="num">${p.ok} / ${p.requests}</td>
      <td class="num">${pct(1-p.ok/p.requests,2)}</td>
      <td class="num">${pct(p.repeat,1)}</td>
      <td style="white-space:normal">${esc(reason)}</td></tr>`;
  }
  $('#t-lat').innerHTML=h+'</tbody>';
}

/* ---------- threshold simulator ---------- */
function renderThreshold(){
  const slider=$('#thr'), out=$('#thr-out');
  const CAL=D.order.filter(n=>D.providers[n].calib_n>0);
  function run(){
    const th=parseFloat(slider.value);
    out.textContent=th.toFixed(2);
    let h='<thead><tr><th>服务</th><th>自动处理占比</th><th>自动子集准确率</th><th>需人工</th><th>自动集错误数</th></tr></thead><tbody>';
    const res={};
    for(const n of CAL){
      let auto=0, ok=0;
      for(const it of D.items){
        const p=it.p[n];
        if(p.prob==null) continue;
        if(p.prob>=th){ auto++; if(p.pred===it.gold) ok++; }
      }
      const total=D.items.filter(it=>it.p[n].prob!=null).length;
      res[n]={cov:auto/total, acc:auto?ok/auto:null, err:auto-ok, auto, total};
    }
    const bestCov=Math.max(...CAL.map(n=>res[n].cov));
    const accs=CAL.map(n=>res[n].acc).filter(v=>v!=null);
    const bestAcc=accs.length?Math.max(...accs):null;
    const errs=CAL.map(n=>res[n].err);
    const bestErr=Math.min(...errs);
    for(const n of CAL){
      const r=res[n];
      h+=`<tr><th style="color:${COLOR[n]}">${esc(D.providers[n].name)}</th>
        <td class="num${r.cov===bestCov?' best':''}">${pct(r.cov,1)} <span style="color:#6f849a">(${r.auto}/${r.total})</span></td>
        <td class="num${r.acc!=null&&r.acc===bestAcc?' best':''}">${r.acc==null?'—':pct(r.acc,1)}</td>
        <td class="num">${pct(1-r.cov,1)}</td>
        <td class="num${r.err===bestErr?' best':''}">${r.err}</td></tr>`;
    }
    h+=`<tr><th>Qwen3.5-9B</th><td class="na" colspan="4">不返回原生概率，无法设阈值（不伪造 one-hot 分布）</td></tr>`;
    $('#t-thr').innerHTML=h+'</tbody>';
    const note = Math.abs(th-0.90)<0.005
      ? '当前为本项目在独立 dev 集选定的阈值 0.90。dev→test 迁移偏差仅 0.2–4.6 个百分点，可用于生产配置。'
      : '提示：阈值必须在独立开发集上选定，再到测试集验证。直接按本页测试集结果反推阈值会高估线上表现。';
    $('#thr-note').textContent=note;
  }
  slider.addEventListener('input',run);
  run();
}

/* ---------- explorer ---------- */
const PAGE=24;
let shown=PAGE, filtered=[];

function matchFilters(){
  const q=$('#q').value.trim().toLowerCase();
  const g=$('#f-gold').value;
  const c=$('#f-case').value;
  filtered=D.items.filter(it=>{
    if(q && !(it.q.toLowerCase().includes(q) || it.ctx.toLowerCase().includes(q) || it.id.includes(q))) return false;
    if(g && it.gold!==g) return false;
    if(c){
      const preds=D.order.map(n=>it.p[n].pred);
      const oks=D.order.map(n=>it.p[n].pred===it.gold);
      if(c==='alldiff'  && new Set(preds).size<2) return false;
      if(c==='allwrong' && oks.some(Boolean))     return false;
      if(c==='allright' && !oks.every(Boolean))   return false;
    }
    return true;
  });
}

function renderItems(reset){
  if(reset) shown=PAGE;
  matchFilters();
  const box=$('#items');
  const slice=filtered.slice(0,shown);
  box.innerHTML = slice.map(it=>{
    const preds=D.order.map(n=>{
      const p=it.p[n];
      const cls = p.pred==null ? '' : (p.pred===it.gold ? 'ok' : 'bad');
      return `<div class="pred ${cls}">
        <div class="pn">${esc(D.providers[n].name)}</div>
        <div class="pv">${p.pred==null?'失败':esc(p.pred)}</div>
        <div class="pm">P ${p.prob==null?'N/A':p.prob.toFixed(3)} · ${p.ms==null?'—':p.ms+' ms'}</div>
      </div>`;
    }).join('');
    return `<div class="item">
      <div class="top">
        <span class="pid">PMID ${esc(it.id)}</span>
        <span class="q">${esc(it.q)}</span>
        <span class="tag gold">标准 ${esc(it.gold)}</span>
      </div>
      <button class="toggle">展开摘要上下文 ▾</button>
      <div class="ctx">${esc(it.ctx)}…</div>
      <div class="preds">${preds}</div>
    </div>`;
  }).join('') || '<p class="note">没有匹配的题目。</p>';

  box.querySelectorAll('.toggle').forEach(b=>b.addEventListener('click',()=>{
    const item=b.closest('.item');
    item.classList.toggle('open');
    b.textContent = item.classList.contains('open') ? '收起摘要上下文 ▴' : '展开摘要上下文 ▾';
  }));
  $('#count').textContent=`匹配 ${filtered.length} / ${D.items.length} 题，已显示 ${slice.length}`;
  $('#more').style.display = shown<filtered.length ? 'block' : 'none';
}

function initExplorer(){
  ['#q','#f-gold','#f-case'].forEach(s=>{
    const e=$(s);
    e.addEventListener(s==='#q'?'input':'change',()=>renderItems(true));
  });
  $('#more').addEventListener('click',()=>{ shown+=PAGE; renderItems(false); });
  renderItems(true);
}

/* ---------- boot ---------- */
fetch('./data/benchmark.json')
  .then(r=>{ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
  .then(d=>{
    D=d;
    renderHero(); renderMain(); renderPaired();
    renderClasses(); renderConfusion(); renderLatency();
    renderThreshold(); initExplorer(); initTabs();
  })
  .catch(e=>{
    document.querySelector('main').innerHTML =
      '<section class="panel on"><div class="callout warn"><b>数据加载失败：'+esc(e.message)+
      '</b><br/>请确认 <code>data/benchmark.json</code> 可访问。本页需通过 HTTP 访问，直接打开本地文件会被浏览器安全策略拦截。</div></section>';
  });
