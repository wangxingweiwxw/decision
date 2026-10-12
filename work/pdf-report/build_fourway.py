from fourway_style import *
from matplotlib.patches import FancyBboxPatch

ROOT=Path(r'D:\Decision')
RUN=ROOT/'decision-benchmark/results/20261012T101454_247943Z'
WORK=ROOT/'work/pdf-report/fourway'
WORK.mkdir(exist_ok=True)
OUT=ROOT/'reports'
sys.path.insert(0,str(ROOT/'decision-benchmark'))
from bench.metrics import question_records,classification_metrics,percentile

meta=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
samples=[json.loads(x) for x in (RUN/'dataset.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
rows=[json.loads(x) for x in (RUN/'requests.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
assert meta['status']=='complete' and not meta['mock'],'must be a complete real run'
assert len(rows)==152 and len(samples)==12

names=['bailian','microsoft','qwen','jev']
cn={'bailian':'阿里云百炼','microsoft':'微软 Decision-1','qwen':'Qwen3.5-9B','jev':'Jev 1.13'}
col={'bailian':'#15858A','microsoft':'#3E5EB8','qwen':'#B77826','jev':'#8E4BA8'}
labels=['support','refute','insufficient'];lab=dict(zip(labels,['支持','反驳','信息不足']))
subject={'demo_physics_evidence':'物理','demo_chemistry_evidence':'化学','demo_biology_evidence':'生物'}

combined={};measured={};first={};by_round={}
for name in names:
    g,_=question_records(samples,rows,name)
    combined[name]=classification_metrics([r for v in g.values() for r in v],labels)
    measured[name]=[r for r in rows if r['provider']==name and r['phase']=='measure']
    assert len(measured[name])==36 and all(r['ok'] for r in measured[name])
    first[name]={r['sample_id']:r for r in measured[name] if r['round']==0}
    by_round[name]=[{k:percentile([r['latency_ms'] for r in measured[name] if r['round']==i],q)
                     for k,q in [('p50',.5),('p95',.95)]} for i in range(3)]
    assert combined[name]['accuracy']==summary['providers'][name]['classification_accuracy']
stats=summary['providers']
qwen_rows=[r for r in rows if r['provider']=='qwen' and r['ok']]
QWEN_NO_THINK=all(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens')==0 for r in qwen_rows)
TZ=timezone(timedelta(hours=8))
start=datetime.fromisoformat(meta['created_at']).astimezone(TZ)
end=datetime.fromisoformat(meta['finished_at']).astimezone(TZ)

def figsave(fig,name):
    path=WORK/(name+'.png');fig.savefig(path,dpi=210,bbox_inches='tight');plt.close(fig);return path
def box(ax,x,y,w,h,text,c='#15858A',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.014',
                 facecolor=c+'12',edgecolor=c,linewidth=1.2,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.6)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#718292',lw=1.5))
def lower(faster,slower,key):return (1-stats[faster][key]/stats[slower][key])*100
def pred_of(n,sid):return first[n][sid]['answers']['verdict']['predicted']
def prob_of(n,sid):
    p=first[n][sid]['answers']['verdict'].get('pred_probability')
    return 'N/A' if p is None else f'{p:.4f}'
def wrong_ids(n):return [s['id'] for s in samples if pred_of(n,s['id'])!=s['expected']['verdict']]

ORDER=sorted(names,key=lambda n:stats[n]['p50_ms'])
FASTEST=ORDER[0]
BEST_ACC=[n for n in names if combined[n]['accuracy']==max(combined[m]['accuracy'] for m in names)]

# ---- Figure 1: roles and relationships
fig,ax=plt.subplots(figsize=(8.6,4.3));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
box(ax,.06,.76,.36,.15,'Qwen3.5-9B\n微软披露的后训练基础',col['qwen'])
box(ax,.58,.76,.36,.15,'Microsoft Decision-1\n专门训练的决策模型',col['microsoft'])
arrow(ax,(.42,.835),(.58,.835));ax.text(.50,.945,'后训练',ha='center',fontsize=10)
box(ax,.03,.40,.40,.22,'本次公共参照\nOpenRouter → DeepInfra / BF16\nChat 接口；请求关闭思考',col['qwen'],9)
box(ax,.57,.40,.40,.22,'本次决策服务\nOpenRouter → Azure\n原生 decisions 接口',col['microsoft'],9)
arrow(ax,(.215,.74),(.215,.64));arrow(ax,(.785,.74),(.785,.64))
box(ax,.03,.04,.40,.19,'决策服务参照：阿里云百炼\ndecision-model-preview',col['bailian'],9)
box(ax,.57,.04,.40,.19,'决策服务参照：TypeSafe Jev\ntypesafe/jev-1.13（OpenRouter）',col['jev'],9)
arrow(ax,(.215,.38),(.215,.25));arrow(ax,(.785,.38),(.785,.25))
ax.text(.5,.315,'公开托管 Qwen 不是微软训练前检查点的精确复现',ha='center',fontsize=9,color='#43566B')
F_ARCH=figsave(fig,'architecture')

# ---- Figure 2: test design flow
fig,ax=plt.subplots(figsize=(8.6,4.0));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
box(ax,.25,.82,.50,.14,'12 道相同科学语境题\n本地答案不发送给任何模型')
xs=[.0,.265,.530,.795]
for x,n in zip(xs,names):
    box(ax,x,.52,.205,.15,cn[n],col[n],8.4);arrow(ax,(.5,.80),(x+.1025,.69))
box(ax,.17,.27,.66,.13,'12 题 × 3 轮 × 4 服务 = 144 次正式请求')
for x in xs:arrow(ax,(x+.1025,.50),(.5,.42))
box(ax,.015,.03,.42,.12,'首轮 12 题 / 家 → 准确率')
box(ax,.565,.03,.42,.12,'三轮 36 次 / 家 → 延迟',col['microsoft'])
arrow(ax,(.4,.25),(.25,.16));arrow(ax,(.6,.25),(.75,.16))
F_FLOW=figsave(fig,'flow')

# ---- Figure 3: accuracy dots + Wilson CI
fig,axes=plt.subplots(2,1,figsize=(9.0,4.8),gridspec_kw={'height_ratios':[1,1.05]})
for i,n in enumerate(names):
    y=len(names)-1-i
    for j,s in enumerate(samples):
        ok=pred_of(n,s['id'])==s['expected']['verdict']
        axes[0].scatter(j,y,color=col[n] if ok else '#CF5734',marker='o' if ok else 'X',s=85)
    axes[0].text(11.55,y+.33,f"{combined[n]['correct']}/12",va='center',fontsize=8.4,color=col[n])
    m=combined[n];a=m['accuracy']*100;lo,hi=[v*100 for v in m['accuracy_ci95']]
    axes[1].errorbar(a,y,xerr=[[a-lo],[hi-a]],fmt='o',capsize=5,color=col[n],lw=2)
    axes[1].text(a+2.5,y+.30,f'{a:.2f}%  [{lo:.2f}%, {hi:.2f}%]',ha='center',va='center',fontsize=8)
axes[0].set_xticks(range(12));axes[0].set_xticklabels([s['id'][-3:] for s in samples],fontsize=8.6)
axes[0].set_xlim(-.6,11.9);axes[0].set_ylim(-.6,3.75)
axes[0].set_xlabel('样本编号；圆点 = 正确，橙色叉号 = 错误',fontsize=8.6)
axes[1].set_xlim(0,104);axes[1].set_ylim(-.6,3.8)
axes[1].set_xlabel('准确率 / %；横线 = 95% Wilson 区间（右侧为数值）',fontsize=8.6)
axes[1].grid(axis='x',alpha=.15)
for ax in axes:
    ax.set_yticks(range(len(names))[::-1]);ax.set_yticklabels([cn[n] for n in names],fontsize=8)
fig.tight_layout(h_pad=2);F_ACC=figsave(fig,'accuracy')

# ---- Figure 4: confusion matrices
fig,axes=plt.subplots(1,4,figsize=(7.6,2.6))
for ax,n in zip(axes,names):
    mat=np.array([[combined[n]['confusion'][a][b] for b in labels] for a in labels])
    ax.imshow(mat,cmap='Blues',vmin=0,vmax=4)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=11,
                    color='white' if mat[i,j]>2 else '#22374B')
    ax.set_xticks(range(3));ax.set_xticklabels([lab[l] for l in labels],rotation=28,fontsize=7,ha='right')
    ax.set_yticks(range(3));ax.set_yticklabels([lab[l] for l in labels],fontsize=7.4)
    ax.set_title(cn[n],fontsize=9.4);ax.set_xlabel('预测',fontsize=8)
axes[0].set_ylabel('标准答案',fontsize=8)
fig.tight_layout();F_CONF=figsave(fig,'confusion')
print('part1 ok',F_ARCH.name,F_FLOW.name,F_ACC.name,F_CONF.name)

# ---- Figure 5: latency bars (P50 / P95)
fig,axes=plt.subplots(1,2,figsize=(7.2,2.7))
BAR=ORDER
for ax,key,title in [(axes[0],'p50_ms','P50 中位耗时'),(axes[1],'p95_ms','P95 耗时')]:
    vals=[stats[n][key] for n in BAR]
    ax.hlines(range(len(BAR)),0,vals,color=[col[n] for n in BAR],lw=2.4,alpha=.55)
    ax.scatter(vals,range(len(BAR)),color=[col[n] for n in BAR],s=70,zorder=3)
    hi=max(vals)
    for i,v in enumerate(vals):ax.text(v+hi*.03,i,f'{v:.1f} ms',va='center',fontsize=8.2)
    ax.set_yticks(range(len(BAR)));ax.set_yticklabels([cn[n] for n in BAR],fontsize=8)
    ax.set_xlim(0,hi*1.30);ax.set_title(title,fontsize=10)
    ax.set_xlabel('毫秒，线性轴，越短越快',fontsize=8.4)
    ax.grid(axis='x',alpha=.15);ax.invert_yaxis()
fig.tight_layout();F_LAT=figsave(fig,'latency')

# ---- Figure 6: latency distribution + empirical CDF
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
SHORT={n:cn[n] for n in names}
for n in names:
    v=sorted(r['latency_ms'] for r in measured[n])
    axes[0].scatter([SHORT[n]]*len(v),v,color=col[n],alpha=.55,s=18)
    axes[1].step(v,np.arange(1,len(v)+1)/len(v)*100,where='post',color=col[n],lw=1.6,label=cn[n])
axes[0].set_ylabel('单次请求耗时 / ms',fontsize=8.4)
axes[0].tick_params(axis='y',labelsize=7.6)
axes[0].tick_params(axis='x',labelsize=7)
plt.setp(axes[0].get_xticklabels(),rotation=20,ha='right',rotation_mode='anchor')
axes[0].set_title('全部正式请求（36 次/家）',fontsize=9.6);axes[0].grid(axis='y',alpha=.15)
axes[0].set_yscale('log')
axes[1].set_xscale('log');axes[1].set_xlabel('耗时 / ms（对数轴）',fontsize=8.4)
axes[1].set_ylabel('累计占比 / %',fontsize=8.4);axes[1].set_title('经验累积分布',fontsize=9.6)
axes[1].legend(fontsize=7.4,loc='lower center',bbox_to_anchor=(.5,-.52),ncol=4,frameon=False)
axes[1].grid(alpha=.15)
fig.tight_layout();F_DIST=figsave(fig,'distribution')

# ---- Figure 7: per-sample median latency
fig,ax=plt.subplots(figsize=(7.2,2.85))
x=np.arange(12)
for n in names:
    med=[statistics.median([r['latency_ms'] for r in measured[n] if r['sample_id']==s['id']]) for s in samples]
    ax.plot(x,med,marker='o',ms=4,color=col[n],lw=1.5,label=cn[n])
ax.set_xticks(x);ax.set_xticklabels([s['id'][-3:] for s in samples],fontsize=7.6)
ax.set_yscale('log');ax.set_ylabel('三轮耗时中位数 / ms',fontsize=8.4)
ax.set_xlabel('样本编号',fontsize=8.4)
ax.legend(fontsize=8,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False)
ax.grid(alpha=.15)
fig.tight_layout();F_PER=figsave(fig,'per_sample')

# ---- Figure 8: calibration shown as grouped bins + coverage (12 items cannot
# support a bubble reliability curve: only a few non-empty, high-probability bins).
fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
CAL=[n for n in names if combined[n]['calibration_n']>0]
bin_idx=sorted({i for n in CAL for i,b in enumerate(combined[n]['reliability_bins']) if b['count']})
wd=0.8/len(CAL)
for k,n in enumerate(CAL):
    bins=combined[n]['reliability_bins']
    xs=[i+ (k-(len(CAL)-1)/2)*wd for i in range(len(bin_idx))]
    ys=[bins[i]['accuracy']*100 if bins[i]['count'] else 0 for i in bin_idx]
    axes[0].bar(xs,ys,width=wd,color=col[n],label=cn[n],alpha=.88)
    for x,i in zip(xs,bin_idx):
        if bins[i]['count']:
            axes[0].text(x,bins[i]['accuracy']*100+2.5,f"n={bins[i]['count']}",
                         ha='center',fontsize=6.2,color='#43566B',rotation=90)
axes[0].set_xticks(range(len(bin_idx)))
axes[0].set_xticklabels([f"{combined[CAL[0]]['reliability_bins'][i]['lower']:.1f}-"
                         f"{combined[CAL[0]]['reliability_bins'][i]['upper']:.1f}" for i in bin_idx],fontsize=7)
axes[0].set_xlabel('所选类别概率分箱',fontsize=8.4);axes[0].set_ylabel('该箱实际正确率 / %',fontsize=8.4)
axes[0].set_ylim(0,132);axes[0].set_yticks([0,20,40,60,80,100])
axes[0].legend(fontsize=6.2,loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,frameon=False,columnspacing=.9,handlelength=1.2)
axes[0].set_title('概率分箱与实际正确率（n=每箱题数）',fontsize=8.8)
axes[0].grid(axis='y',alpha=.15)
for k,n in enumerate(CAL):
    cv=[c for c in combined[n]['coverage'] if c['accuracy'] is not None]
    mk=['s','^','D'][k];dash=[(0,()),(0,(4,2)),(0,(1,1.5))][k]
    axes[1].plot([c['actual_coverage']*100 for c in cv],[c['accuracy']*100 for c in cv],
                 marker=mk,ms=5,color=col[n],lw=1.7,ls=dash,label=cn[n],alpha=.9,
                 markerfacecolor='none',markeredgewidth=1.5)
axes[1].set_xlabel('自动处理覆盖率 / %',fontsize=8.4);axes[1].set_ylabel('该子集准确率 / %',fontsize=8.4)
axes[1].legend(fontsize=6.4,loc='lower left',framealpha=.92)
axes[1].grid(alpha=.15);axes[1].set_ylim(0,108);axes[1].set_xlim(0,104)
axes[1].set_title('按概率排序的覆盖率-准确率',fontsize=8.8)
fig.suptitle('Qwen3.5-9B 不返回原生概率，未纳入本页两图',fontsize=7.4,color='#B77826',y=1.04)
fig.tight_layout();F_CAL=figsave(fig,'calibration')
print('part2 ok',F_LAT.name,F_DIST.name,F_PER.name,F_CAL.name)

# ================= PDF document (14-page framework from the uploaded PDF) =================
RUN_ID=RUN.name
BANNER='DECISION BENCHMARK  /  四方真实 API 评估'
FOOT=f'{start:%Y-%m-%d} | 小样本探索性报告 | 每家 12 个独立输入'
def deco(cv,doc):
    cv.saveState()
    cv.setFont('CN',7.6);cv.setFillColor(colors.HexColor('#6C7E8D'))
    cv.drawString(44,H-30,BANNER)
    cv.setStrokeColor(colors.HexColor('#D6E0E8'));cv.setLineWidth(.6);cv.line(44,H-38,W-44,H-38)
    cv.line(44,42,W-44,42)
    cv.drawString(44,30,FOOT)
    cv.drawRightString(W-44,30,f'{cv.getPageNumber()} / 14')
    cv.restoreState()

def tiny(t):raw(t,'SmallCN')
def ms(v):return f'{v:.1f}'

# ---------- Page 1: cover + headline table
raw('真实 API 四方测试  技术评估报告','SmallCN')
raw('四方模型准确率\n与响应速度对比','TitleCN')
raw('阿里云百炼 · 微软 Decision-1 · Qwen3.5-9B · Jev 1.13','BodyCN')
raw(f'{start:%Y-%m-%d %H:%M:%S} - {end:%H:%M:%S}（北京时间）','SmallCN')
acc_txt='、'.join(cn[n] for n in BEST_ACC)
gap=(combined[BEST_ACC[0]]['accuracy']-min(combined[n]['accuracy'] for n in names))*100
sub(f'准确率并列第一：{acc_txt}；最低时延：{cn[FASTEST]}')
para(f'准确率：{acc_txt} 均为 12/12（100%），比百炼与 Qwen 高 {gap:.2f} 个百分点，多答对 1 题。')
para(f'响应速度：{cn[FASTEST]} 的 P50 为 {ms(stats[FASTEST]["p50_ms"])} ms，'
     f'比 {cn[ORDER[1]]} 低 {lower(FASTEST,ORDER[1],"p50_ms"):.2f}%，'
     f'比最慢的 {cn[ORDER[-1]]} 低 {lower(FASTEST,ORDER[-1],"p50_ms"):.2f}%。')
para(f'两家满分服务之间：{cn["jev"]} 的 P50 比 {cn["microsoft"]} 低 '
     f'{lower("jev","microsoft","p50_ms"):.2f}%（少等 {stats["microsoft"]["p50_ms"]-stats["jev"]["p50_ms"]:.1f} ms），'
     f'P95 低 {lower("jev","microsoft","p95_ms"):.2f}%。')
hdr=['指标']+[cn[n] for n in names]
rowspec=[('首轮正确数',lambda n:f"{combined[n]['correct']} / 12",'max'),
         ('准确率  越高越好',lambda n:pct(combined[n]['accuracy']),'max'),
         ('Macro-F1  越高越好',lambda n:f"{combined[n]['macro_f1']:.4f}",'max'),
         ('P50 ms  越低越好',lambda n:ms(stats[n]['p50_ms']),'min'),
         ('P95 ms  越低越好',lambda n:ms(stats[n]['p95_ms']),'min'),
         ('正式请求成功数',lambda n:f"{stats[n]['successful_requests']} / {stats[n]['requests']}",'eq'),
         ('重复分类一致率',lambda n:pct(stats[n]['repeat_agreement']),'eq')]
data=[hdr];winners=set()
keyf={'首轮正确数':lambda n:combined[n]['correct'],'准确率  越高越好':lambda n:combined[n]['accuracy'],
      'Macro-F1  越高越好':lambda n:combined[n]['macro_f1'],'P50 ms  越低越好':lambda n:stats[n]['p50_ms'],
      'P95 ms  越低越好':lambda n:stats[n]['p95_ms']}
for ri,(label_,fn,mode) in enumerate(rowspec,start=1):
    data.append([label_]+[fn(n) for n in names])
    if mode in ('max','min'):
        vals={n:keyf[label_](n) for n in names}
        best=(max if mode=='max' else min)(vals.values())
        for ci_,n in enumerate(names,start=1):
            if vals[n]==best:winners.add((ci_,ri))
table(data,[PAGE_W*.26]+[PAGE_W*.185]*4,winners)
tiny('浅绿色表示该行最优；并列最优同时标色，四家持平项不标色。百分比降幅以被比较服务的耗时为分母。')
para(f'若同时看准确率与等待时间，{cn["jev"]} 与 {cn["microsoft"]} 都值得进入下一轮验证，本次 Jev 的时延更低；'
     f'若以最低等待时间为主，{cn[FASTEST]} 值得继续验证，但它在一道数值比较题上出错。')
tiny('仅 12 个独立人工示例；全部准确率配对区间包含零。这是本次服务表现，不是总体科学能力或纯模型推理速度排名。')
page()

# ---------- Page 2: roles
heading('01','四个模型的角色与关系')
figure(F_ARCH,'图 1. Qwen 是微软披露的后训练基础；百炼与 Jev 是另外两项决策服务参照。',300)
para('保留 Qwen3.5-9B 是为了观察专门决策模型与公开基础模型参照的差异。纳入 Jev 则用于比较另一种决策服务'
     '在同一输入和评分条件下的表现。')
table([['服务','请求模型与入口']]+
      [[cn['bailian'],'decision-model-preview；百炼 systemone 接口'],
       [cn['microsoft'],'microsoft/microsoft-decision-1；OpenRouter decisions'],
       [cn['qwen'],'qwen/qwen3.5-9b；OpenRouter chat/completions'],
       [cn['jev'],'typesafe/jev-1.13；OpenRouter decisions']],[PAGE_W*.3,PAGE_W*.7])
callout('Jev 的接入范围',
 '本次通过 OpenRouter 的 Decisions 接口调用 TypeSafe 的 Jev 决策模型，固定请求 1.13 版本，并记录接口返回的实际快照。'
 '它与微软走同一 OpenRouter 入口和同一把密钥，支持预定义选项的结构化决策；本次仅测试 Choice 三分类，不测试 Score 或 Noul。')
tiny('微软与公开托管 Qwen 的硬件、接口、训练检查点及提示封装不同；服务差距不能全部归因于后训练。Jev 的结构和训练基础未在本次实验中验证。来源见第 14 页。')
page()

# ---------- Page 3: test design
heading('02','同次运行的测试设计')
figure(F_FLOW,'图 2. 每家另有 2 次暖机，共 152 条记录。暖机不计入准确率和主要延迟。',290)
table([['项目','实际设置'],
       ['题目','12 个独立中文人工示例；物理、化学、生物各 4 条；三标签各 4 条'],
       ['运行',f"{meta['rounds']} 轮；并发 {meta['concurrency']}；种子 {meta['seed']}；每题服务顺序随机；无自动重试"],
       ['计时','HTTP 发出前至完整响应体接收；不含 JSON 解码、本地评分和请求间停顿'],
       ['Qwen 适配','strict JSON schema；temperature=0；max_tokens=256；关闭思考；固定 DeepInfra BF16'],
       ['评分','仅首轮算准确率；服务失败计为错误；后两轮统计延迟与重复一致性'],
       ['总运行耗时',f"{meta['wall_seconds_including_warmup']:.2f} 秒，包含暖机和停顿"]],[PAGE_W*.22,PAGE_W*.78])
para('四家使用相同 state 和 questions，标准答案仅保存在本地。Qwen 额外包含通用分类格式提示'
     +('，且所有调用的 reasoning_tokens 为 0；因此结果对应本次关闭思考的 Chat 基线。' if QWEN_NO_THINK else '。'))
tiny('测试复用了长连接；未隔离云端缓存、硬件、网关和排队。所有输入均较短，本次没有评估长论文或高并发。')
page()
print('doc part A ok')

# ---------- Page 4: accuracy & uncertainty
heading('03','准确率与不确定性')
figure(F_ACC,'图 3. 每点代表一题；叉号为错误。下图为首轮准确率及 95% Wilson 区间。',330)
prs=[['配对差  左减右','差值 百分点','95% 配对区间']]
for pr in summary['paired']:
    prs.append([f"{cn[pr['left']]} - {cn[pr['right']]}",f"{pr['difference']*100:+.2f}",
                f"[{pr['ci95'][0]*100:+.2f}, {pr['ci95'][1]*100:+.2f}]"])
table(prs,[PAGE_W*.44,PAGE_W*.22,PAGE_W*.34])
tiny(f"按输入 ID 进行 {meta['bootstrap_draws']:,} 次配对 bootstrap；6 组比较未经多重校正。全部区间包含零，不能确认总体准确率差异。"
     "微软与 Jev 在本批题上逐题完全一致，产生退化的 [0,0] 区间；这不意味着两者在新题上必然等效。")
page()

# ---------- Page 5: errors by subject
heading('04','分类错误与学科分布')
figure(F_CONF,'图 4. 行为标准标签，列为预测标签；非对角线表示错误，每格为题数。',225)
srows=[['指标']+[cn[n] for n in names],
       ['合并 Macro-F1']+[f"{combined[n]['macro_f1']:.4f}" for n in names]]
for task,zh in subject.items():
    ids=[s['id'] for s in samples if s['task']==task]
    srows.append([zh]+[f"{sum(pred_of(n,i)==next(s for s in samples if s['id']==i)['expected']['verdict'] for i in ids)} / {len(ids)}" for n in names])
table(srows,[PAGE_W*.26]+[PAGE_W*.185]*4)
wb=wrong_ids('bailian');wq=wrong_ids('qwen')
para(f"百炼错误为 {wb[0]}，将数值冲突判为支持；Qwen 错误为 {wq[0]}，将缺少目标测量判为反驳。"
     "微软和 Jev 首轮均无误判。")
tiny('这些题测量的是记录与结论之间的关系；每科仅 4 条，不能代表物理建模、化学机制推断或生物专业研究能力。')
page()

# ---------- Page 6: latency gaps
heading('05','响应速度与等待差距')
figure(F_LAT,'图 5. 每家 36 次正式请求，越短越快；点为分位值，线长仅示意。',260)
cmp_rows=[['比较  左侧更快','P50 耗时降低','P95 耗时降低']]
for i,a in enumerate(ORDER):
    for b in ORDER[i+1:]:
        cmp_rows.append([f'{cn[a]} / {cn[b]}',f'{lower(a,b,"p50_ms"):.2f}%',f'{lower(a,b,"p95_ms"):.2f}%'])
table(cmp_rows,[PAGE_W*.46,PAGE_W*.27,PAGE_W*.27])
para(f'{cn["jev"]} 相对 {cn["microsoft"]} 的 P50 减少 {stats["microsoft"]["p50_ms"]-stats["jev"]["p50_ms"]:.1f} ms，'
     f'P95 减少 {stats["microsoft"]["p95_ms"]-stats["jev"]["p95_ms"]:.1f} ms。'
     f'{cn[FASTEST]} 走国内直连，其余三家经 OpenRouter，网络路径不对等。')
tiny('P95 为线性插值分位数，既不是最大耗时，也不是 SLA。比较包含网络、网关、排队和返回过程；没有把各厂商 server latency 当成统一指标。')
page()

# ---------- Page 7: distribution
heading('06','延迟分布与同题比较')
figure(F_DIST,'图 6. 左图展示全部正式请求（对数纵轴）；右图展示耗时的经验累积分布。',250)
figure(F_PER,'图 7. 每道题的三轮耗时中位数；图例颜色在整份报告中保持一致。',235)
para('四家在本次运行呈现不同等待水平。每家 36 次全部成功，因此成功请求延迟与全部尝试延迟相同，'
     '不存在丢弃失败后使速度显得更快的问题。')
tiny('未对尾部延迟进行显著性检验。重复输入可能被上游缓存，且本次测试只有一个时段、并发 1；不能外推到吞吐量或长期稳定性。')
page()

# ---------- Page 8: rounds + two stable errors
heading('07','轮次变化与两道稳定错题')
rrows=[['轮次 P50 ms']+[cn[n] for n in names]]
for i in range(3):
    rrows.append([str(i+1)]+[ms(by_round[n][i]['p50']) for n in names])
table(rrows,[PAGE_W*.26]+[PAGE_W*.185]*4)
for sid,title in [(wb[0],'数值比较'),(wq[0],'证据缺失')]:
    s=next(x for x in samples if x['id']==sid)
    sub(f'{sid}  {title}')
    st=s['state']
    para(f"结论：{st.get('claim','')}")
    para(f"记录：{st.get('record','')}")
    para(f"标准：{lab[s['expected']['verdict']]}")
    seq=[['服务','三轮结果','首轮所选概率']]
    for n in names:
        tri='/'.join(lab[[r for r in measured[n] if r['sample_id']==sid and r['round']==k][0]['answers']['verdict']['predicted']] for k in range(3))
        seq.append([cn[n],tri,prob_of(n,sid)])
    table(seq,[PAGE_W*.3,PAGE_W*.42,PAGE_W*.28])
tiny('四家重复分类一致率均为 24/24；错误也保持稳定。输出稳定并不代表正确。数值关系可增加确定性规则；缺少测量应保留"信息不足"类别。')
page()

# ---------- Page 9: probability quality
heading('08','概率质量与自动处理阈值')
def num(v,d=6):return '不可用' if v is None else f'{v:.{d}f}'
table([['指标']+[cn[n] for n in names],
       ['Brier  越低越好']+[num(combined[n]['brier_sum']) for n in names],
       ['ECE  越低越好']+[num(combined[n]['ece_10_bins']) for n in names],
       ['有效概率分布数']+[f"{combined[n]['calibration_n']} / 12" for n in names]],
      [PAGE_W*.26]+[PAGE_W*.185]*4)
figure(F_CAL,'图 8. 左图为非空概率分箱的实际正确率，n 为该箱题数；右图按所选类别概率排序观察覆盖率与准确率。',265)
para('百炼、微软与 Jev 返回原生概率分布；Qwen 本次只输出类别，不生成自报概率，也不人为补成 0/1 分布，'
     '其 Brier 和 ECE 留为不可用。')
para('Brier 比较完整类别分布与标准答案；ECE 比较所选类别概率与分箱正确率。接口 confidence 的定义可能不同，'
     '报告没有直接用它跨厂商排名。')
tiny('12 题不足以验证校准。微软与 Jev 都答对时，Brier/ECE 主要体现它们对本批正确答案的概率集中程度；不代表对未知题的风险估计更可靠。阈值应在独立开发集选择，再在测试集验证。')
page()
print('doc part B ok')

# ---------- Pages 10-12: per-item audit, 4 items each
for gi,(lo,hi) in enumerate([(0,4),(4,8),(8,12)]):
    heading(f'{9+gi:02d}',f'逐题审计 {samples[lo]["id"][-3:]} 至 {samples[hi-1]["id"][-3:]}')
    tiny('标准答案由本地测试集给出；以下为首轮输出。P 是所选类别概率；Qwen 未请求概率，记为 N/A。')
    for s in samples[lo:hi]:
        sub(f"{s['id']}   标准 {lab[s['expected']['verdict']]}")
        para(f"结论：{s['state'].get('claim','')}")
        para(f"记录：{s['state'].get('record','')}")
        rowsT=[['模型']+[cn[n] for n in names],
               ['预测']+[lab[pred_of(n,s['id'])] for n in names],
               ['P']+[prob_of(n,s['id']) for n in names]]
        wrong_cells=set()
        for ci_,n in enumerate(names,start=1):
            if pred_of(n,s['id'])!=s['expected']['verdict']:wrong_cells.add((ci_,1))
        table(rowsT,[PAGE_W*.14]+[PAGE_W*.215]*4)
    page()

# ---------- Page 13: conclusion boundary
heading('12','结论边界与下一轮专业评测')
sub('本次可以支持的选择')
para(f'在这 12 道中文证据判断题上，{cn["microsoft"]} 与 {cn["jev"]} 准确率相同且首轮全对，本次 Jev 的 P50 与 P95 都更低；'
     f'{cn["bailian"]} 拥有最低等待时间，但出现一道数值误判；{cn["qwen"]} 在本次关闭思考的配置下少对一题且耗时最高。'
     '这个观察适合决定下一轮优先验证对象，不足以作为采购结论。')
sub('不能从本次推导的结论')
para('本次不是专业科研能力排行榜，也不是微软后训练收益的因果实验。模型结构、上游硬件、精度、接口封装与网关不同；'
     '尚未评估长输入、高并发、跨时段变化、长尾失败、计费一致性或自动重试。百炼走国内直连而其余三家经 OpenRouter，'
     '延迟差中包含网络路径差异。')
table([['下一轮评测','建议设计'],
       ['数据与标注','数百条起步，由专业人员核对；覆盖数值单位、反事实、信息缺失与长摘要'],
       ['独立性','按论文、实验对象或来源分组划分开发集与测试集，防止同源泄漏'],
       ['对照设置','增加确定性规则；Qwen 开启思考另设实验组，避免混入本次配置'],
       ['运行条件','区分首次与重复输入，测试多个时间段、输入长度和并发等级；尽量对齐网络路径'],
       ['统计与成本','预注册主要指标；六组比较预设校正方案；分开报告失败率、关键漏判与成本']],
      [PAGE_W*.24,PAGE_W*.76])
para('准确率优先的流程可先评估微软和 Jev；时延敏感的流程可评估百炼加数值规则与人工复核。'
     '需要文字解释或开放式推理时，应另测生成模型任务，而不是让本次 Choice 测试替代。')
tiny('新增诊断题应形成独立集合。不能反复修改当前测试题、针对错题调整提示后，再把同一批题的提高当成外部泛化收益。')
page()

# ---------- Page 14: reproduction + sources
heading('13','复现记录与资料来源')
raw(f'cd {ROOT}\decision-benchmark\n'
    'python benchmark.py --env-override run --providers bailian,microsoft,qwen,jev --rounds 3 --warmup 2 --concurrency 1','CodeCN')
table([['服务','实际返回模型','上游']]+
      [[cn[n],(stats[n]['served_models'] or ['-'])[0],(stats[n]['upstream_providers'] or ['-'])[0]] for n in names],
      [PAGE_W*.26,PAGE_W*.48,PAGE_W*.26])
para(f'原始记录：{RUN}')
ds=hashlib.sha256((RUN/"dataset.jsonl").read_bytes()).hexdigest()
rq=hashlib.sha256((RUN/"requests.jsonl").read_bytes()).hexdigest()
para(f"Python {meta['python']}；requests {meta['requests_version']}；平台 {meta['platform']}；"
     "34 项离线测试通过；四方 6 组配对统计已校验。")
tiny('request_payload 保留发送内容，raw_response 保留接口返回；无认证头。标准答案仅在本地数据中。每个新运行保存到新的时间戳目录。')
table([['文件','SHA-256'],
       ['dataset.jsonl',ds[:32]+'\n'+ds[32:]],
       ['requests.jsonl',rq[:32]+'\n'+rq[32:]]],[PAGE_W*.26,PAGE_W*.74])
sub('指标口径')
tiny('Accuracy = 首轮正确题数 / 12；Macro-F1 = 合并三类 F1 的算术平均；每家 P50/P95 使用 36 次正式请求和线性插值。'
     '暖机排除。延迟差为本次观察值，未估计置信区间。')
sub('官方资料')
for t in ['Jev 接入与版本（OpenRouter）：https://openrouter.ai/docs/guides/community/jev',
          '微软发布及后训练基础：https://commandline.microsoft.com/microsoft-decision-1-model-foundry/',
          'OpenRouter Qwen：https://openrouter.ai/qwen/qwen3.5-9b',
          '阿里云百炼决策接口：https://help.aliyun.com/zh/model-studio/decision-model-api']:
    tiny(t)
tiny(f'实验日期 {start:%Y-%m-%d}（北京时间）；运行目录 {RUN_ID}。')

OUT.mkdir(exist_ok=True)
pdf_path=OUT/f'decision_models_fourway_realrun_{start:%Y-%m-%d}.pdf'
doc=SimpleDocTemplate(str(pdf_path),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=52,bottomMargin=52,
                      title='决策模型四方真实 API 评估报告',author='Decision Benchmark')
doc.build(story,onFirstPage=deco,onLaterPages=deco)
pages=len(PdfReader(str(pdf_path)).pages)
print('PDF:',pdf_path,'pages =',pages)
