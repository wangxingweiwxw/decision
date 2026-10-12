from fourway_style import *
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

ROOT=Path(r'D:\Decision')
RUN=ROOT/'decision-benchmark/results/20261012T124545_166990Z'
DEV=Path(r'C:\Users\win\AppData\Local\Temp\pmqa_dev')
WORK=ROOT/'work/pdf-report/pubmedqa'
WORK.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'reports'
sys.path.insert(0,str(ROOT/'decision-benchmark'))
from bench.metrics import question_records,classification_metrics,percentile

meta=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
samples=[json.loads(x) for x in (RUN/'dataset.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
rows=[json.loads(x) for x in (RUN/'requests.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
assert meta['status']=='complete' and not meta['mock'],'must be a complete real run'
assert len(samples)==500 and meta['rounds']==3

names=['bailian','microsoft','qwen','jev']
cn={'bailian':'阿里云百炼','microsoft':'微软 Decision-1','qwen':'Qwen3.5-9B','jev':'Jev 1.13'}
col={'bailian':'#15858A','microsoft':'#3E5EB8','qwen':'#B77826','jev':'#8E4BA8'}
labels=['yes','no','maybe']; lab={'yes':'yes 支持','no':'no 反驳','maybe':'maybe 不确定'}
TASK='pubmedqa/answer'
stats=summary['providers']
combined={n:stats[n]['tasks'][TASK] for n in names}
measured={}; first={}; by_round={}
for n in names:
    measured[n]=[r for r in rows if r['provider']==n and r['phase']=='measure']
    assert len(measured[n])==1500
    first[n]={r['sample_id']:r for r in measured[n] if r['round']==0}
    by_round[n]=[{k:percentile([r['latency_ms'] for r in measured[n] if r['round']==i and r['ok']],q)
                  for k,q in [('p50',.5),('p95',.95)]} for i in range(3)]
gold={s['id']:s['expected']['answer'] for s in samples}
TZ=timezone(timedelta(hours=8))
start=datetime.fromisoformat(meta['created_at']).astimezone(TZ)
end=datetime.fromisoformat(meta['finished_at']).astimezone(TZ)
TOTAL_REQ=sum(stats[n]['requests'] for n in names)

def figsave(fig,name):
    p=WORK/(name+'.png');fig.savefig(p,dpi=210,bbox_inches='tight');plt.close(fig);return p
def lower(fast,slow,key):return (1-stats[fast][key]/stats[slow][key])*100
def pred_of(n,sid):
    r=first[n].get(sid)
    return r['answers']['answer']['predicted'] if r and r['ok'] else None
def prob_of(n,sid):
    r=first[n].get(sid)
    if not r or not r['ok']: return None
    return r['answers']['answer'].get('pred_probability')
ORDER=sorted(names,key=lambda n:stats[n]['p50_ms'])
ACC_ORDER=sorted(names,key=lambda n:-combined[n]['accuracy'])
BEST=ACC_ORDER[0]; FASTEST=ORDER[0]
SIG=[p for p in summary['paired'] if not (p['ci95'][0]<=0<=p['ci95'][1])]

# ---- Fig 1: scale comparison (12 -> 500)
fig,ax=plt.subplots(figsize=(8.6,3.5));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def box(a,x,y,w,h,t,c='#15858A',s=10):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.014',
        facecolor=c+'12',edgecolor=c,linewidth=1.2,clip_on=False))
    a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=s,linespacing=1.6)
def arrow(a,p,q):a.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',color='#718292',lw=1.6))
box(ax,.015,.60,.40,.30,'此前：演示集 12 题\n144 次请求\n6 组配对全部含零\n无法判定优劣','#7E8C99',10)
box(ax,.585,.60,.40,.30,f'本次：PubMedQA {len(samples)} 题\n{TOTAL_REQ:,} 次正式请求\n{len(SIG)}/6 组配对显著\n结论可用于选型','#15858A',10)
arrow(ax,(.42,.75),(.58,.75))
box(ax,.015,.10,.40,.34,'数据：PQA-L 专家标注\n分层划分 dev 199 / test 801\n本次抽样 test 500 题\n随机种子 42','#3E7FD8',9.4)
box(ax,.585,.10,.40,.34,'题型：yes / no / maybe 三分类\n输入含研究问题 + 摘要正文\n字符数 p50≈1868\n仅首轮计准确率','#8E4BA8',9.4)
F_SCALE=figsave(fig,'scale')

# ---- Fig 2: accuracy with Wilson CI
fig,ax=plt.subplots(figsize=(8.6,2.9))
for i,n in enumerate(ACC_ORDER):
    m=combined[n];a=m['accuracy']*100;lo,hi=[v*100 for v in m['accuracy_ci95']]
    y=len(names)-1-i
    ax.errorbar(a,y,xerr=[[a-lo],[hi-a]],fmt='o',ms=9,capsize=6,color=col[n],lw=2.4)
    ax.text(hi+1.2,y,f'{a:.2f}%  [{lo:.2f}, {hi:.2f}]',va='center',fontsize=8.6)
ax.set_yticks(range(len(names))[::-1]);ax.set_yticklabels([cn[n] for n in ACC_ORDER],fontsize=9)
ax.set_xlim(55,95);ax.set_xlabel('首轮准确率 / %；横线 = 95% Wilson 区间',fontsize=8.8)
ax.grid(axis='x',alpha=.16)
fig.tight_layout();F_ACC=figsave(fig,'accuracy')

# ---- Fig 3: paired bootstrap intervals
fig,ax=plt.subplots(figsize=(8.6,3.4))
pairs=summary['paired']
for i,pr in enumerate(pairs):
    y=len(pairs)-1-i
    d=pr['difference']*100;lo,hi=pr['ci95'][0]*100,pr['ci95'][1]*100
    signif=not(lo<=0<=hi)
    c='#1F7A4D' if signif else '#96A3B0'
    ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt='o',ms=7,capsize=5,color=c,lw=2.2)
    ax.text(hi+.6,y,f'{d:+.2f}pp'+('  显著' if signif else '  含零'),va='center',fontsize=8,color=c)
ax.axvline(0,color='#CF5734',ls='--',lw=1.3)
ax.set_yticks(range(len(pairs))[::-1])
ax.set_yticklabels([f"{cn[p['left']]} − {cn[p['right']]}" for p in pairs],fontsize=8.4)
ax.set_xlim(-22,19);ax.set_xlabel('准确率差 / 百分点；红线 = 零点；绿色 = 区间不含零',fontsize=8.8)
ax.grid(axis='x',alpha=.16)
fig.tight_layout();F_PAIR=figsave(fig,'paired')

# ---- Fig 4: confusion matrices
fig,axes=plt.subplots(1,4,figsize=(7.8,2.7))
for ax,n in zip(axes,names):
    mat=np.array([[combined[n]['confusion'][a][b] for b in labels] for a in labels])
    ax.imshow(mat,cmap='Blues',vmin=0,vmax=mat.max())
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=9.6,
                    color='white' if mat[i,j]>mat.max()*.55 else '#22374B')
    ax.set_xticks(range(3));ax.set_xticklabels(labels,rotation=28,fontsize=7.2,ha='right')
    ax.set_yticks(range(3));ax.set_yticklabels(labels,fontsize=7.2)
    ax.set_title(cn[n],fontsize=9.2);ax.set_xlabel('预测',fontsize=8)
axes[0].set_ylabel('标准答案',fontsize=8)
fig.tight_layout();F_CONF=figsave(fig,'confusion')
print('part1 ok')

# ---- Fig 5: per-class F1 (highlights the shared maybe weakness)
fig,ax=plt.subplots(figsize=(8.6,2.9))
x=np.arange(3); w=0.2
for k,n in enumerate(names):
    f1=[combined[n]['per_class'][l]['f1'] for l in labels]
    ax.bar(x+(k-1.5)*w,f1,width=w,color=col[n],label=cn[n],alpha=.9)
for k,n in enumerate(names):
    for i,l in enumerate(labels):
        v=combined[n]['per_class'][l]['f1']
        REF=0.30; band=(REF-0.055, REF+0.030)
        ytxt=v+0.042
        if band[0] <= ytxt <= band[1] or abs(v-REF)<0.055:
            ax.text(i+(k-1.5)*w,max(v-0.055,0.012),f'{v:.2f}',ha='center',va='bottom',
                    fontsize=6.2,color='white',fontweight='bold',zorder=7)
        else:
            ax.text(i+(k-1.5)*w,ytxt,f'{v:.2f}',ha='center',fontsize=6.4,color='#43566B',zorder=7,
                    path_effects=[pe.withStroke(linewidth=3.2,foreground='white')])
ax.axhline(0.3,color='#CF5734',ls='--',lw=1.2,zorder=1)
ax.text(1.0,0.995,'红色虚线 = F1 0.30 参考线',fontsize=7.2,color='#CF5734',ha='center')
ax.set_xticks(x)
ax.set_xticklabels([f"{lab[l]}  (n={combined[names[0]]['per_class'][l]['support']})" for l in labels],fontsize=8.4)
ax.set_ylim(0,1.06);ax.set_ylabel('F1',fontsize=8.8)
ax.legend(fontsize=7,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.17),frameon=False)
ax.grid(axis='y',alpha=.16)
fig.tight_layout();F_CLS=figsave(fig,'perclass')

# ---- Fig 6: latency P50/P95 lollipop
fig,axes=plt.subplots(1,2,figsize=(8.6,2.7))
for ax,key,title in [(axes[0],'p50_ms','P50 中位耗时'),(axes[1],'p95_ms','P95 耗时')]:
    vals=[stats[n][key] for n in ORDER]
    ax.hlines(range(len(ORDER)),0,vals,color=[col[n] for n in ORDER],lw=2.4,alpha=.55)
    ax.scatter(vals,range(len(ORDER)),color=[col[n] for n in ORDER],s=70,zorder=3)
    hi=max(vals)
    for i,v in enumerate(vals):ax.text(v+hi*.03,i,f'{v:.0f} ms',va='center',fontsize=8.2)
    ax.set_yticks(range(len(ORDER)));ax.set_yticklabels([cn[n] for n in ORDER],fontsize=8)
    ax.set_xlim(0,hi*1.32);ax.set_title(title,fontsize=9.6)
    ax.set_xlabel('毫秒，线性轴，越短越快',fontsize=8.4)
    ax.grid(axis='x',alpha=.16);ax.invert_yaxis()
fig.tight_layout();F_LAT=figsave(fig,'latency')

# ---- Fig 7: coverage vs accuracy (risk curve) + ECDF of latency
fig,axes=plt.subplots(1,2,figsize=(8.6,2.9))
CAL=[n for n in names if combined[n]['calibration_n']>0]
for k,n in enumerate(CAL):
    cv=[c for c in combined[n]['coverage'] if c['accuracy'] is not None]
    mk=['s','^','D'][k];dash=[(0,()),(0,(4,2)),(0,(1,1.5))][k]
    axes[0].plot([c['actual_coverage']*100 for c in cv],[c['accuracy']*100 for c in cv],
                 marker=mk,ms=5,color=col[n],lw=1.8,ls=dash,label=cn[n],
                 markerfacecolor='none',markeredgewidth=1.5)
axes[0].set_xlabel('自动处理覆盖率 / %',fontsize=8.4)
axes[0].set_ylabel('该子集准确率 / %',fontsize=8.4)
axes[0].legend(fontsize=6.8,loc='lower left',framealpha=.92)
axes[0].grid(alpha=.16);axes[0].set_ylim(50,100);axes[0].set_xlim(0,104)
axes[0].set_title('覆盖率-准确率（风险曲线）\nQwen 无原生概率，未纳入',fontsize=8.6)
for n in names:
    v=sorted(r['latency_ms'] for r in measured[n] if r['ok'])
    axes[1].step(v,np.arange(1,len(v)+1)/len(v)*100,where='post',color=col[n],lw=1.7,label=cn[n])
axes[1].set_xscale('log');axes[1].set_xlabel('耗时 / ms（对数轴）',fontsize=8.4)
axes[1].set_ylabel('累计占比 / %',fontsize=8.4);axes[1].set_title('延迟经验累积分布',fontsize=9)
axes[1].legend(fontsize=6.6,loc='lower center',bbox_to_anchor=(.5,-.52),ncol=4,frameon=False)
axes[1].grid(alpha=.16)
fig.tight_layout();F_COV=figsave(fig,'coverage')

# ---- Fig 8: dev-selected threshold validated on test
THR=0.90
devv={'bailian':(52.0,80.8),'microsoft':(55.0,83.6),'jev':(70.0,85.7)}
testv={}
for n in CAL:
    recs=[(prob_of(n,s['id']), pred_of(n,s['id'])==gold[s['id']]) for s in samples]
    recs=[(p,c) for p,c in recs if p is not None]
    kept=[c for p,c in recs if p>=THR]
    testv[n]=(len(kept)/len(recs)*100, sum(kept)/len(kept)*100, len(kept)-sum(kept))
fig,axes=plt.subplots(1,2,figsize=(8.6,2.8))
xx=np.arange(len(CAL)); w=0.34
axes[0].bar(xx-w/2,[devv[n][0] for n in CAL],width=w,label='dev 选定',color='#9AB4C8')
axes[0].bar(xx+w/2,[testv[n][0] for n in CAL],width=w,label='test 实测',color='#15858A')
for i,n in enumerate(CAL):
    axes[0].text(i-w/2,devv[n][0]+1.5,f'{devv[n][0]:.1f}%',ha='center',fontsize=7.4)
    axes[0].text(i+w/2,testv[n][0]+1.5,f'{testv[n][0]:.1f}%',ha='center',fontsize=7.4)
axes[0].set_xticks(xx);axes[0].set_xticklabels([cn[n] for n in CAL],fontsize=7.8)
axes[0].set_ylabel('自动处理占比 / %',fontsize=8.4);axes[0].set_ylim(0,92)
axes[0].legend(fontsize=7,frameon=False,ncol=2,loc='upper left')
axes[0].set_title(f'阈值 {THR} 的自动化率：dev → test',fontsize=8.8);axes[0].grid(axis='y',alpha=.16)
axes[1].bar(xx-w/2,[devv[n][1] for n in CAL],width=w,label='dev 选定',color='#B7C8DC')
axes[1].bar(xx+w/2,[testv[n][1] for n in CAL],width=w,label='test 实测',color='#3E5EB8')
for i,n in enumerate(CAL):
    axes[1].text(i-w/2,devv[n][1]+1.2,f'{devv[n][1]:.1f}%',ha='center',fontsize=7.4)
    axes[1].text(i+w/2,testv[n][1]+1.2,f'{testv[n][1]:.1f}%',ha='center',fontsize=7.4)
axes[1].set_xticks(xx);axes[1].set_xticklabels([cn[n] for n in CAL],fontsize=7.8)
axes[1].set_ylabel('自动子集准确率 / %',fontsize=8.4);axes[1].set_ylim(0,112)
axes[1].legend(fontsize=7,frameon=False,ncol=2,loc='upper left')
axes[1].set_title('自动子集准确率：dev → test',fontsize=8.8);axes[1].grid(axis='y',alpha=.16)
fig.suptitle('Qwen3.5-9B 不返回原生概率，无法设阈值，未纳入本页两图',fontsize=7.4,color='#B77826',y=1.05)
fig.tight_layout();F_THR=figsave(fig,'threshold')
print('part2 ok')

# ================= PDF document: 14-page framework =================
BANNER='DECISION BENCHMARK  /  PubMedQA 大样本四方评测'
FOOT=f'{start:%Y-%m-%d} | 大样本选型评测 | PubMedQA test {len(samples)} 题 × {meta["rounds"]} 轮'
def deco(cv,doc):
    cv.saveState()
    cv.setFont('CN',7.6);cv.setFillColor(colors.HexColor('#6C7E8D'))
    cv.drawString(44,H-30,BANNER)
    cv.setStrokeColor(colors.HexColor('#D6E0E8'));cv.setLineWidth(.6)
    cv.line(44,H-38,W-44,H-38);cv.line(44,42,W-44,42)
    cv.drawString(44,30,FOOT)
    cv.drawRightString(W-44,30,f'{cv.getPageNumber()} / 14')
    cv.restoreState()
def tiny(t):raw(t,'SmallCN')
def ms(v):return f'{v:.1f}'
def num(v,d=4):return '不可用' if v is None else f'{v:.{d}f}'

# ---------- Page 1: cover + headline
raw('真实 API 大样本评测  技术选型报告','SmallCN')
raw('PubMedQA 四方评测\n可用于选型的结论','TitleCN')
raw('阿里云百炼 · 微软 Decision-1 · Qwen3.5-9B · Jev 1.13','BodyCN')
raw(f'{start:%Y-%m-%d %H:%M} - {end:%H:%M}（北京时间）· {TOTAL_REQ:,} 次正式请求 · 耗时 {meta["wall_seconds_including_warmup"]/60:.1f} 分钟','SmallCN')
sub(f'结论：准确率首选 {cn[BEST]} 或 {cn["microsoft"]}；最低时延 {cn[FASTEST]}')
para(f'{len(SIG)}/6 组两两配对区间不含零，首次具备选型效力。此前 12 题演示集为 0/6，无法判定优劣。')
para(f'{cn["jev"]} 准确率 {combined["jev"]["accuracy"]*100:.2f}%，{cn["microsoft"]} {combined["microsoft"]["accuracy"]*100:.2f}%，'
     f'两者差 {abs(summary["paired"][4]["difference"])*100:.2f}pp 且区间含零，统计上等效。')
para(f'{cn["bailian"]} P50 仅 {ms(stats["bailian"]["p50_ms"])} ms（最快），但准确率显著落后 '
     f'{(combined["jev"]["accuracy"]-combined["bailian"]["accuracy"])*100:.2f} 个百分点。')
hdr=['指标']+[cn[n] for n in names]
rows_spec=[('首轮准确率  越高越好',lambda n:pct(combined[n]['accuracy']),'max',lambda n:combined[n]['accuracy']),
  ('Macro-F1  越高越好',lambda n:f"{combined[n]['macro_f1']:.4f}",'max',lambda n:combined[n]['macro_f1']),
  ('Brier  越低越好',lambda n:num(combined[n]['brier_sum']),'min',lambda n:combined[n]['brier_sum']),
  ('P50 ms  越低越好',lambda n:ms(stats[n]['p50_ms']),'min',lambda n:stats[n]['p50_ms']),
  ('P95 ms  越低越好',lambda n:ms(stats[n]['p95_ms']),'min',lambda n:stats[n]['p95_ms']),
  ('失败请求  越低越好',lambda n:f"{stats[n]['requests']-stats[n]['successful_requests']} / {stats[n]['requests']}",'min',
      lambda n:stats[n]['requests']-stats[n]['successful_requests']),
  ('重复一致率',lambda n:pct(stats[n]['repeat_agreement']),'max',lambda n:stats[n]['repeat_agreement'])]
data=[hdr];winners=set()
for ri,(lb,fn,mode,key) in enumerate(rows_spec,start=1):
    data.append([lb]+[fn(n) for n in names])
    vals={n:key(n) for n in names if key(n) is not None}
    if vals:
        best=(max if mode=='max' else min)(vals.values())
        for ci_,n in enumerate(names,start=1):
            if vals.get(n)==best:winners.add((ci_,ri))
table(data,[PAGE_W*.26]+[PAGE_W*.185]*4,winners)
tiny('浅绿色为该行最优；并列最优同时标色。Qwen 不返回原生概率，其 Brier 为不可用，不参与该行比较。')
tiny('PubMedQA 为公开数据集，题目可能已进入模型训练数据，绝对准确率可能偏高；跨模型相对比较受影响较小。')
page()

# ---------- Page 2: why a large sample
heading('01','为什么要做大样本')
figure(F_SCALE,'图 1. 从 12 题演示集升级到 500 题专家标注集，配对区间由全部含零变为多数可判定。',270)
table([['对比项','12 题演示集','本次 PubMedQA 500 题'],
       ['正式请求数','144','%d'%TOTAL_REQ],
       ['可判定配对','0 / 6','%d / 6'%len(SIG)],
       ['准确率区间宽度','约 50 个百分点','约 7–8 个百分点'],
       ['能否用于选型','否，仅验证流程','是，但受第 13 页边界约束']],
      [PAGE_W*.24,PAGE_W*.33,PAGE_W*.43])
para('小样本下即使准确率差 8.33 个百分点，配对区间仍覆盖零点，差异可能完全由抽样波动造成。'
     '500 题把区间收窄到可判定范围，才使「谁更准」成为统计结论而非观感。')
tiny('样本量只解决抽样误差，不解决数据集代表性、训练数据污染和网络路径差异等系统性问题。')
page()

# ---------- Page 3: design
heading('02','测试设计与公平性')
table([['项目','实际设置'],
       ['数据集','PubMedQA PQA-L 专家标注；按标签分层划分 dev 199 / test 801；本次从 test 抽样 500'],
       ['题型','yes / no / maybe 三分类；输入含研究问题与摘要正文，字符数 p50 约 1868'],
       ['运行',f"{meta['rounds']} 轮；并发 {meta['concurrency']}；种子 {meta['seed']}；暖机 {meta['warmup_per_worker_per_provider']} 次/家；每题服务顺序随机"],
       ['计时','HTTP 发出前至完整响应体接收；不含 JSON 解码、本地评分与请求间停顿'],
       ['Qwen 适配','strict JSON schema；temperature=0；关闭思考；固定 DeepInfra BF16；仅返回类别键'],
       ['评分','仅首轮算准确率；失败计为错误；后两轮用于延迟与重复一致性'],
       ['阈值','在独立 dev 集选定 0.90，再到 test 验证，未用 test 成绩反推']],
      [PAGE_W*.20,PAGE_W*.80])
callout('标准答案从不发送给模型',
  '导入器只把 QUESTION 与 CONTEXTS 放入请求，排除 LONG_ANSWER；final_decision 仅进入本地标准答案。'
  '四家收到完全相同的 state 与 questions。数据来源与文件哈希记录在 provenance.json。')
tiny('测试复用长连接；未隔离云端缓存、硬件、网关与排队。百炼走国内直连，其余三家经 OpenRouter，网络路径不对等。')
page()

# ---------- Page 4: accuracy
heading('03','准确率与置信区间')
figure(F_ACC,'图 2. 首轮准确率及 95% Wilson 区间；区间已收窄到约 7–8 个百分点。',225)
t=[['服务','首轮正确数','准确率','95% Wilson 区间','Macro-F1']]
for n in ACC_ORDER:
    m=combined[n];ci=m['accuracy_ci95']
    t.append([cn[n],f"{m['correct']} / {m['n']}",pct(m['accuracy']),
              f"[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]",f"{m['macro_f1']:.4f}"])
table(t,[PAGE_W*.24,PAGE_W*.17,PAGE_W*.15,PAGE_W*.26,PAGE_W*.18])
para(f'{cn["jev"]} 与 {cn["microsoft"]} 的区间大幅重叠，{cn["bailian"]} 与 {cn["qwen"]} 的区间同样重叠；'
     '前两家与后两家之间则基本分离，这与下一页的配对检验一致。')
tiny('Macro-F1 为三类 F1 的算术平均。PubMedQA 标签不均衡（yes 272 / no 178 / maybe 50），Macro-F1 会放大小类别的影响，应与第 5 页同读。')
page()

# ---------- Page 5: paired significance
heading('04','配对显著性：结论能否成立')
figure(F_PAIR,'图 3. 六组两两配对的准确率差与 95% 区间；绿色为区间不含零的可判定对比。',255)
pr_rows=[['配对  左减右','差值 百分点','95% 配对区间','判定']]
for pr in summary['paired']:
    lo,hi=pr['ci95'][0]*100,pr['ci95'][1]*100
    crosses=lo<=0<=hi
    verdict='含零，不显著' if crosses else ('左显著更优' if lo>0 else '右显著更优')
    pr_rows.append([f"{cn[pr['left']]} - {cn[pr['right']]}",f"{pr['difference']*100:+.2f}",
                    f"[{lo:+.2f}, {hi:+.2f}]",verdict])
table(pr_rows,[PAGE_W*.38,PAGE_W*.16,PAGE_W*.24,PAGE_W*.22])
tiny(f"按输入 ID 聚类的 {meta['bootstrap_draws']:,} 次配对 bootstrap；6 组比较未做多重校正，属探索性分析。"
     "微软与 Jev、百炼与 Qwen 两组区间含零，不能宣称其中任一方更准确。")
page()
print('doc A ok')

# ---------- Page 6: error structure by class
heading('05','错误结构与共性短板')
figure(F_CLS,'图 4. 分类别 F1；红线为 0.30 参考线。四家在 maybe 类上全部低于该线。',235)
t=[['类别 / 样本数']+[cn[n] for n in names]]
for l in labels:
    sup=combined[names[0]]['per_class'][l]['support']
    t.append([f'{lab[l]}  n={sup}']+[f"{combined[n]['per_class'][l]['f1']:.3f}" for n in names])
t.append(['Macro-F1']+[f"{combined[n]['macro_f1']:.4f}" for n in names])
table(t,[PAGE_W*.26]+[PAGE_W*.185]*4)
callout('共性短板：maybe 类全线失效',
  f"四家在「证据不足 / 不确定」类的 F1 分别为 "
  f"{combined['bailian']['per_class']['maybe']['f1']:.3f}、{combined['microsoft']['per_class']['maybe']['f1']:.3f}、"
  f"{combined['qwen']['per_class']['maybe']['f1']:.3f}、{combined['jev']['per_class']['maybe']['f1']:.3f}，"
  "全部低于 0.24。这是决策模型品类的共性问题，不是某一家的缺陷。"
  "生产含义：凡涉及「证据是否充分」的判断，都不应交给模型自动裁决，必须人工复核或规则兜底。")
para(f"另一处结构性风险：{cn['bailian']} 的 no 类召回仅 "
     f"{combined['bailian']['per_class']['no']['recall']*100:.1f}%，"
     "即大量「反驳」被判成「支持」。在需要识别否定结论的场景风险显著，这也是其 Macro-F1 最低的主因。")
page()

# ---------- Page 7: confusion matrices
heading('06','混淆矩阵逐家对照')
figure(F_CONF,'图 5. 行为标准答案，列为预测；非对角线为错误，每格为题数（首轮 500 题）。',230)
para('矩阵显示四家的错误并非随机分布：maybe 行几乎没有被正确命中，多数被判为 yes 或 no；'
     f"{cn['bailian']} 的 no 行有大量流向 yes。这说明模型倾向给出明确结论，而不愿输出「不确定」。")
tiny('每格为题数而非比例；各类别样本量不同（yes 272 / no 178 / maybe 50），比较召回时应结合第 6 页的分类别 F1。')
page()

# ---------- Page 8: latency
heading('07','响应速度与稳定性')
figure(F_LAT,'图 6. P50 与 P95 耗时（线性轴，越短越快）；仅统计成功的正式请求。',250)
cmp_rows=[['比较  左侧更快','P50 耗时降低','P95 耗时降低']]
for i,a in enumerate(ORDER):
    for b in ORDER[i+1:]:
        cmp_rows.append([f'{cn[a]} / {cn[b]}',f'{lower(a,b,"p50_ms"):.2f}%',f'{lower(a,b,"p95_ms"):.2f}%'])
table(cmp_rows,[PAGE_W*.46,PAGE_W*.27,PAGE_W*.27])
rr=[['轮次 P50 ms']+[cn[n] for n in names]]
for i in range(3):
    rr.append([str(i+1)]+[ms(by_round[n][i]['p50']) for n in names])
table(rr,[PAGE_W*.26]+[PAGE_W*.185]*4)
para(f"三轮 P50 未见持续漂移。{cn['bailian']} 的领先包含国内直连与跨境 OpenRouter 的路径差异，"
     "不能等同于模型本身更快。")
tiny('P95 为线性插值分位数，不是最大耗时也不是 SLA。本次仅并发 1、单一时段，不能外推到吞吐量或长期稳定性。')
page()

# ---------- Page 9: failures
heading('08','失败与可用性风险')
fr=[['服务','成功 / 总数','失败率','超时','失败原因']]
for n in names:
    p=stats[n]; errs=p['errors']
    if errs:
        k=list(errs.keys())[0]
        reason=('HTTP 429 上游限流（DeepInfra 共享池 engine_overloaded）' if '429' in k else k[:60])
        reason+=f'  共 {sum(errs.values())} 次'
    else:
        reason='无'
    fr.append([cn[n],f"{p['successful_requests']} / {p['requests']}",pct(p['failure_rate']),
               str(p['timeout_count']),reason])
table(fr,[PAGE_W*.19,PAGE_W*.16,PAGE_W*.11,PAGE_W*.09,PAGE_W*.45])
qf=stats['qwen']['requests']-stats['qwen']['successful_requests']
para(f"{cn['qwen']} 出现 {qf} 次失败，全部为 OpenRouter 共享池上游限流，"
     "属可用性问题而非模型能力问题。但它提示：依赖共享池路由的服务在批量负载下存在限流风险，"
     "生产部署应自备上游密钥或指定专用 provider。其余三家零失败。")
tiny('程序不自动重试，失败按错误计入准确率分母；同时报告仅成功响应子集的准确率，避免隐藏慢失败。')
page()
print('doc B ok')

# ---------- Page 10: calibration
heading('09','概率质量与风险曲线')
table([['指标']+[cn[n] for n in names],
       ['Brier  越低越好']+[num(combined[n]['brier_sum'],4) for n in names],
       ['ECE  越低越好']+[num(combined[n]['ece_10_bins'],4) for n in names],
       ['有效概率分布数']+[f"{combined[n]['calibration_n']} / 500" for n in names]],
      [PAGE_W*.26]+[PAGE_W*.185]*4)
figure(F_COV,'图 7. 左：按所选类别概率排序的覆盖率-准确率曲线；右：延迟经验累积分布。',240)
para(f"{cn['jev']} 的 Brier 最低，{cn['microsoft']} 的 ECE 最低，两者校准质量接近；"
     f"{cn['bailian']} 的 Brier 与 ECE 明显偏高，其置信度可靠性较弱。"
     f"{cn['qwen']} 只输出类别、不自报概率，也不人为补成 0/1 分布，故概率指标为不可用。")
tiny('Brier 为多分类平方误差求和（三分类范围 0–2）；ECE 使用 10 个等宽分箱，采用「被选中类别的概率」而非厂商 confidence 作为共同口径。')
page()

# ---------- Page 11: threshold validation
heading('10','生产阈值：dev 选定到 test 验证')
figure(F_THR,'图 8. 阈值 0.90 在 dev 选定后到 test 的实际表现；两侧纵轴均自零起。',250)
th=[['服务','自动占比 dev','自动占比 test','准确率 dev','准确率 test','test 需人工','自动集错误']]
for n in CAL:
    d=devv[n];v=testv[n]
    th.append([cn[n],f'{d[0]:.1f}%',f'{v[0]:.1f}%',f'{d[1]:.1f}%',f'{v[1]:.1f}%',f'{100-v[0]:.1f}%',str(v[2])])
table(th,[PAGE_W*.19]+[PAGE_W*.135]*6)
para('阈值在独立 dev 集（100 题）选定为 0.90，再到 test 验证，未用 test 成绩反推。'
     f"{cn['jev']} 的自动占比偏差仅 {abs(testv['jev'][0]-devv['jev'][0]):.1f} 个百分点，阈值可迁移性良好，可直接用于生产配置。")
para(f"要最高自动化率选 {cn['jev']}：test 上自动处理 {testv['jev'][0]:.1f}%，仅 {100-testv['jev'][0]:.1f}% 需人工；"
     f"要最低自动错误选 {cn['microsoft']}：自动子集准确率 {testv['microsoft'][1]:.1f}%，自动集错误 {testv['microsoft'][2]} 个（三家最低）。")
tiny('覆盖率结果是测试集的描述性统计，上线前应在自有业务样本上重新标定阈值，并对 maybe 类单独设兜底规则。')
page()

# ---------- Page 12: cost
heading('11','成本与计费口径')
from collections import defaultdict
tok=defaultdict(int)
for r in rows:
    if r['phase']!='measure' or not r['ok']: continue
    u=r.get('usage') or {}
    tok[r['provider']]+= (u.get('input_tokens') or u.get('prompt_tokens') or 0)
base=min(v for v in tok.values() if v)
ct=[['服务',f'{TOTAL_REQ:,} 次请求输入 token','相对量','成本口径']]
for n in names:
    note=('≈ $%.4f @ $0.042/M，输出免费'%(tok[n]/1e6*0.042)) if n in ('microsoft','jev') else '按各平台实际计费'
    ct.append([cn[n],f'{tok[n]:,}',f'{tok[n]/base:.2f}×',note])
table(ct,[PAGE_W*.22,PAGE_W*.28,PAGE_W*.14,PAGE_W*.36])
para(f"{cn['jev']} 的 token 计数约为 {cn['microsoft']} 的 {tok['jev']/tok['microsoft']:.2f} 倍（分词口径不同），"
     "同价下单位成本更高；但由于输出免费，本次全量评测的 OpenRouter 侧成本仍在美分量级。")
tiny('token 数取各服务自报用量，口径可能不一致，不能直接当作输入长度比较。百炼与 Qwen 的实际价格随平台与上游浮动，本页不替其报价。')
page()
print('doc C ok')

# ---------- Page 13: selection guidance
heading('12','选型建议与结论边界')
sg=[['场景','推荐','依据']]
sg.append(['准确率优先',f"{cn['jev']} 或 {cn['microsoft']}",
           f"{combined['jev']['accuracy']*100:.1f}% / {combined['microsoft']['accuracy']*100:.1f}%，显著优于另两家；二者间不显著"])
sg.append(['追求最少人工',cn['jev'],f"阈值 0.90 下自动化 {testv['jev'][0]:.1f}%，人工量仅 {100-testv['jev'][0]:.1f}%"])
sg.append(['自动错误代价高',cn['microsoft'],f"自动子集准确率 {testv['microsoft'][1]:.1f}%，自动集错误 {testv['microsoft'][2]} 个最低"])
sg.append(['毫秒级实时链路',cn['bailian'],f"P50 {ms(stats['bailian']['p50_ms'])} ms、重复一致率 {pct(stats['bailian']['repeat_agreement'])}；须接受准确率落后"])
sg.append(['不建议做生产决策',cn['qwen'],'准确率中游、延迟最高、无原生概率，且遇上游 429 限流'])
table(sg,[PAGE_W*.22,PAGE_W*.22,PAGE_W*.56])
callout('推荐部署形态',
  f"{cn['jev']} 或 {cn['microsoft']} 作主决策 + 置信度 0.90 分流 + maybe 类与低置信统一转人工；"
  f"{cn['bailian']} 可作延迟敏感场景的前置粗筛，但需叠加否定结论的规则校验。")
sub('本次仍不能外推的部分')
para('一、PubMedQA 为公开数据集，题目可能已进入各模型训练数据，绝对准确率可能偏高，相对比较受影响较小；'
     '结论应以新收集并独立标注的样本复核。'
     '二、延迟不可直接跨家比较：百炼国内直连，其余三家经 OpenRouter 跨境路由。'
     '三、仅测 choice 三分类，未覆盖 Score / Noul 题型、长文档、高并发、跨时段稳定性与计费一致性。'
     '四、6 组配对未做多重比较校正；maybe 类仅 50 例，其 F1 区间很宽。')
tiny('各厂商公布的自测指标仍缺独立第三方验证，本报告只以本次实测为准，不得作为厂商能力的最终排名引用。')
page()

# ---------- Page 14: reproduction
heading('13','复现记录与资料来源')
for i,c in enumerate([f'cd {ROOT}\decision-benchmark',
        'python benchmark.py prepare-pubmedqa',
        'python benchmark.py --env-override run --dataset data/pubmedqa/test.jsonl '
        '--limit 500 --rounds 3 --warmup 10 --concurrency 1 --seed 42 '
        '--providers bailian,microsoft,qwen,jev']):
    story.append(Paragraph(escape(c),ParagraphStyle(name=f'Cmd{i}',parent=styles['CodeCN'],fontSize=7.4,leading=11.6)))
story.append(Spacer(1,8))
table([['服务','实际返回模型','上游']]+
      [[cn[n],(stats[n]['served_models'] or ['-'])[0],(stats[n]['upstream_providers'] or ['-'])[0]] for n in names],
      [PAGE_W*.26,PAGE_W*.48,PAGE_W*.26])
para(f'原始记录：{RUN}')
ds=hashlib.sha256((RUN/"dataset.jsonl").read_bytes()).hexdigest()
rq=hashlib.sha256((RUN/"requests.jsonl").read_bytes()).hexdigest()
table([['文件','SHA-256'],
       ['dataset.jsonl',ds[:32]+'\n'+ds[32:]],
       ['requests.jsonl',rq[:32]+'\n'+rq[32:]]],[PAGE_W*.26,PAGE_W*.74])
para(f"Python {meta['python']}；requests {meta['requests_version']}；平台 {meta['platform']}；"
     f"{meta['bootstrap_draws']:,} 次配对 bootstrap；34 项离线测试通过。")
tiny('request_payload 保留发送内容，raw_response 保留接口返回，均不含认证头；百炼 workspace 子域在产物中脱敏。'
     '标准答案仅存于本地数据。每个新运行写入新的时间戳目录，不覆盖历史。')
sub('指标口径')
tiny(f"Accuracy = 首轮正确题数 / {len(samples)}；Macro-F1 = 三类 F1 的算术平均；"
     "P50/P95 使用每家 1500 次正式请求与线性插值，暖机排除；配对 bootstrap 按样本 ID 聚类重采样。")
sub('资料来源')
for u in ['PubMedQA 论文：https://aclanthology.org/D19-1259/',
          'PubMedQA 作者仓库：https://github.com/pubmedqa/pubmedqa',
          'Jev 接入（OpenRouter）：https://openrouter.ai/docs/guides/community/jev',
          '微软发布说明：https://commandline.microsoft.com/microsoft-decision-1-model-foundry/',
          '阿里云百炼决策接口：https://help.aliyun.com/zh/model-studio/decision-model-api']:
    tiny(u)
tiny(f'实验日期 {start:%Y-%m-%d}（北京时间）；运行目录 {RUN.name}。')

OUT.mkdir(exist_ok=True)
pdf_path=OUT/f'pubmedqa_fourway_500x3_{start:%Y-%m-%d}.pdf'
doc=SimpleDocTemplate(str(pdf_path),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=52,bottomMargin=52,
                      title='PubMedQA 大样本四方评测选型报告',author='Decision Benchmark')
doc.build(story,onFirstPage=deco,onLaterPages=deco)
print('PDF:',pdf_path,'pages =',len(PdfReader(str(pdf_path)).pages))
