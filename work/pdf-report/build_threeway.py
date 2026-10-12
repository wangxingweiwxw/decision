from threeway_style import *
from matplotlib.patches import FancyBboxPatch

ROOT=Path(r'C:\decision')
RUN=ROOT/'decision-benchmark/results/20261011T112337_746433Z'
WORK=ROOT/'work/pdf-report/threeway'
WORK.mkdir(exist_ok=True)
OUT=ROOT/'reports'
sys.path.insert(0,str(ROOT/'decision-benchmark'))
from bench.metrics import question_records,classification_metrics,percentile
meta=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
samples=[json.loads(x) for x in (RUN/'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
rows=[json.loads(x) for x in (RUN/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
assert meta['status']=='complete' and not meta['mock'] and len(rows)==114
names=['bailian','microsoft','qwen']
cn={'bailian':'阿里云百炼','microsoft':'微软 Decision-1','qwen':'Qwen3.5-9B'}
col={'bailian':'#15858A','microsoft':'#3E5EB8','qwen':'#B77826'}
labels=['support','refute','insufficient'];lab=dict(zip(labels,['支持','反驳','信息不足']))
combined={};measured={};first={};by_round={}
for name in names:
    g,_=question_records(samples,rows,name)
    combined[name]=classification_metrics([r for v in g.values() for r in v],labels)
    measured[name]=[r for r in rows if r['provider']==name and r['phase']=='measure']
    assert len(measured[name])==36 and all(r['ok'] for r in measured[name])
    first[name]={r['sample_id']:r for r in measured[name] if r['round']==0}
    by_round[name]=[{k:percentile([r['latency_ms'] for r in measured[name] if r['round']==i],q) for k,q in [('p50',.5),('p95',.95)]} for i in range(3)]
    assert combined[name]['accuracy']==summary['providers'][name]['classification_accuracy']
stats=summary['providers']
qwen_rows=[r for r in rows if r['provider']=='qwen']
assert all(r['usage']['completion_tokens_details']['reasoning_tokens']==0 for r in qwen_rows)
start=datetime.fromisoformat(meta['created_at']).astimezone(timezone(timedelta(hours=8)))
end=datetime.fromisoformat(meta['finished_at']).astimezone(timezone(timedelta(hours=8)))

def figsave(fig,name):
    path=WORK/(name+'.png');fig.savefig(path,dpi=210,bbox_inches='tight');plt.close(fig);return path

def box(ax,x,y,w,h,text,c='#15858A',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.014',facecolor=c+'12',edgecolor=c,linewidth=1.2,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.6)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#718292',lw=1.5))
def lower(faster,slower,key):return (1-stats[faster][key]/stats[slower][key])*100

# 1: training provenance vs independently hosted services.
fig,ax=plt.subplots(figsize=(7,4.0));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
box(ax,.08,.75,.34,.15,'Qwen3.5-9B\n微软披露的后训练基础',col['qwen'])
box(ax,.59,.75,.34,.15,'Microsoft Decision-1\n专门训练的决策模型',col['microsoft'])
arrow(ax,(.43,.825),(.57,.825));ax.text(.50,.95,'后训练',ha='center',fontsize=10)
box(ax,.04,.37,.39,.23,'本次公共参照\nOpenRouter → DeepInfra / BF16\nChat 接口；请求关闭思考',col['qwen'],9)
box(ax,.57,.37,.39,.23,'本次决策服务\nOpenRouter → Azure\n原生 decisions 接口',col['microsoft'],9)
arrow(ax,(.25,.73),(.24,.62));arrow(ax,(.76,.73),(.76,.62))
box(ax,.23,.03,.54,.19,'第三方决策服务：阿里云百炼\ndecision-model-preview',col['bailian'])
ax.text(.5,.285,'公开托管 Qwen 不是微软训练前检查点的精确复现',ha='center',fontsize=10,color='#566878')
architecture=figsave(fig,'architecture')

fig,ax=plt.subplots(figsize=(7,3.8));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
box(ax,.25,.81,.50,.14,'12 道相同科学语境题\n本地答案不发送给任何模型')
for x,n in zip([.02,.355,.69],names):
    box(ax,x,.51,.29,.15,cn[n],col[n]);arrow(ax,(.5,.79),(x+.145,.68))
box(ax,.17,.26,.66,.13,'12 题 × 3 轮 × 3 服务 = 108 次正式请求')
for x in [.165,.5,.835]:arrow(ax,(x,.49),(.5,.41))
box(ax,.04,.02,.41,.12,'首轮 12 题 / 家 → 准确率')
box(ax,.55,.02,.41,.12,'三轮 36 次 / 家 → 延迟',col['microsoft'])
arrow(ax,(.4,.24),(.25,.15));arrow(ax,(.6,.24),(.75,.15))
flow=figsave(fig,'flow')

fig,axes=plt.subplots(2,1,figsize=(7,4.9),gridspec_kw={'height_ratios':[1,1.05]})
for i,n in enumerate(names):
    for j,s in enumerate(samples):
        ok=first[n][s['id']]['answers']['verdict']['predicted']==s['expected']['verdict']
        axes[0].scatter(j,2-i,color=col[n] if ok else '#CF5734',marker='o' if ok else 'X',s=95)
    axes[0].text(12,2-i,f"{combined[n]['correct']}/12",va='center',fontsize=10)
    m=combined[n];a=m['accuracy']*100;lo,hi=[v*100 for v in m['accuracy_ci95']]
    axes[1].errorbar(a,2-i,xerr=[[a-lo],[hi-a]],fmt='o',capsize=5,color=col[n],lw=2)
    axes[1].text((lo+hi)/2,2-i+.23,f'{a:.2f}% [{lo:.2f}%, {hi:.2f}%]',ha='center',fontsize=8)
axes[0].set_xticks(range(12),[s['id'][-3:] for s in samples],fontsize=8);axes[0].set_xlim(-.5,13)
axes[0].set_ylim(-.5,2.6);axes[0].set_xlabel('样本编号；圆点 = 正确，橙色叉号 = 错误')
axes[1].set_xlim(0,105);axes[1].set_ylim(-.6,2.7);axes[1].set_xlabel('准确率 / %；横线 = 95% Wilson 区间')
axes[1].grid(axis='x',alpha=.15)
for ax in axes:ax.set_yticks([2,1,0],[cn[n] for n in names],fontsize=8)
fig.tight_layout(h_pad=2);accuracy=figsave(fig,'accuracy')

fig,axes=plt.subplots(1,3,figsize=(7.2,2.55))
for ax,n in zip(axes,names):
    mat=np.array([[combined[n]['confusion'][a][b] for b in labels] for a in labels])
    ax.imshow(mat,cmap='Blues',vmin=0,vmax=4)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=12,color='white' if mat[i,j]>2 else '#22374B')
    ax.set_xticks(range(3),[lab[l] for l in labels],rotation=25,fontsize=8);ax.set_yticks(range(3),[lab[l] for l in labels],fontsize=8)
    ax.set_title(cn[n],fontsize=10);ax.set_xlabel('预测');ax.set_ylabel('标准答案')
fig.tight_layout(w_pad=1.1);confusion=figsave(fig,'confusion')

fig,axes=plt.subplots(1,2,figsize=(7,3.0))
for ax,key,title in zip(axes,['p50_ms','p95_ms'],['P50：典型等待水平','P95：较慢端的等待水平']):
    for i,n in enumerate(names):
        v=stats[n][key];ax.barh(2-i,v,color=col[n],height=.45);ax.text(v+25,2-i,f'{v:.1f}',va='center',fontsize=9)
    ax.set_yticks([2,1,0],[cn[n] for n in names],fontsize=8);ax.set_xlim(0,2250);ax.set_xlabel('客户端耗时 / ms');ax.set_title(title,fontsize=10)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();latency_bars=figsave(fig,'latency_bars')
fig,axes=plt.subplots(1,2,figsize=(7,2.8))
for i,n in enumerate(names):
    v=[r['latency_ms'] for r in measured[n]]
    axes[0].scatter(i+np.random.default_rng(9).uniform(-.12,.12,len(v)),v,color=col[n],alpha=.7,s=12)
    axes[1].step(np.sort(v),np.arange(1,37)/36,where='post',color=col[n],label=cn[n],lw=1.8)
axes[0].set_xticks(range(3),[cn[n] for n in names],fontsize=8);axes[0].set_ylabel('完整响应耗时 / ms')
axes[1].set_xlabel('完整响应耗时 / ms');axes[1].set_ylabel('已完成比例');axes[1].legend(frameon=False,fontsize=7)
fig.tight_layout();latency_scatter=figsave(fig,'latency_scatter')

fig,ax=plt.subplots(figsize=(7,3.0));x=np.arange(12)
for off,n in zip([-.25,0,.25],names):
    vals=[percentile([r['latency_ms'] for r in measured[n] if r['sample_id']==s['id']],.5) for s in samples]
    ax.bar(x+off,vals,width=.24,color=col[n],label=cn[n])
ax.set_xticks(x,[s['id'][-3:] for s in samples]);ax.set_xlabel('样本编号');ax.set_ylabel('三轮耗时中位数 / ms')
ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();sameinput=figsave(fig,'sameinput')

fig,axes=plt.subplots(1,2,figsize=(7,2.7))
for n in names[:2]:
    bins=[b for b in combined[n]['reliability_bins'] if b['count']]
    axes[0].scatter([b['mean_probability'] for b in bins],[b['accuracy'] for b in bins],s=[b['count']*22 for b in bins],color=col[n],label=cn[n],alpha=.8)
    cs=combined[n]['coverage'];axes[1].plot([c['actual_coverage']*100 for c in cs],[c['accuracy']*100 for c in cs],'-o',color=col[n],label=cn[n])
axes[0].plot([.6,1],[.6,1],'--',color='#aab4c0');axes[0].set_xlim(.6,1.04);axes[0].set_ylim(.6,1.06)
axes[0].set_xlabel('分箱内所选类别平均概率');axes[0].set_ylabel('实际准确率');axes[0].legend(frameon=False,fontsize=8,loc='lower right')
axes[1].set_xlabel('自动处理覆盖率 / %');axes[1].set_ylabel('子集准确率 / %');axes[1].set_ylim(40,105)
fig.tight_layout();calibration=figsave(fig,'calibration')

# Page 1
raw('真实 API 三方测试  |  技术评估报告','SubCN')
raw('决策模型与基础模型\n准确率、响应速度对比','TitleCN')
raw('阿里云百炼 · Microsoft Decision-1 · Qwen3.5-9B','BodyCN')
raw(start.strftime('%Y-%m-%d %H:%M:%S')+' - '+end.strftime('%H:%M:%S')+'（北京时间）','SmallCN')
callout('本次结果：微软准确率最高，阿里云百炼响应最快',
    '<b>准确率：</b>微软 12/12（100%），另外两家均 11/12（91.67%）；微软分别领先 <b>8.33 个百分点</b>，多答对 1 题。<br/>'
    +f'<b>相较 Qwen3.5-9B：</b>微软 P50 耗时降低 <b>{lower("microsoft","qwen","p50_ms"):.2f}%</b>，P95 降低 <b>{lower("microsoft","qwen","p95_ms"):.2f}%</b>。<br/>'
    +f'<b>速度最快：</b>阿里云百炼 P50 为 {stats["bailian"]["p50_ms"]:.1f} ms，比微软低 {lower("bailian","microsoft","p50_ms"):.2f}%，比 Qwen 低 {lower("bailian","qwen","p50_ms"):.2f}%。')
table([['核心指标']+[cn[n] for n in names],
       ['首轮正确数 / 独立题数']+[f"{combined[n]['correct']} / 12" for n in names],
       ['准确率（越高越好）']+[pct(combined[n]['accuracy']) for n in names],
       ['P50 ms（越低越好）']+[f(stats[n]['p50_ms']) for n in names],
       ['P95 ms（越低越好）']+[f(stats[n]['p95_ms']) for n in names],
       ['正式请求成功数']+['36 / 36']*3,
       ['重复分类一致率']+['100%（24 / 24）']*3],
      [171,112,112,112],winners=((2,1),(2,2),(1,3),(1,4)))
raw('浅绿色 = 本次指标最优；三家持平项不着色。准确率按独立题计算，延迟按正式请求计算。','SmallCN')
para('<b>如何使用：</b>本次微软相较 Qwen 既少错一题、又响应更快；阿里云百炼具有最短等待时间。重视准确性可进一步验证微软，重视时延可进一步验证阿里云百炼。')
raw('结论边界：仅 12 道人工示例；两两配对区间均包含零，不能证明总体准确率高低。Qwen 是公开托管基础模型参照，三家部署和接口不同，不能将差距全部归因于微软后训练。','SmallCN')
raw('阅读导航：第 2-3 页模型与设计；第 4-5 页准确率；第 6-7 页延迟；第 8 页错误案例；第 9 页概率；第 10-11 页逐题审计；第 12-13 页限制与复现。','SmallCN')
page()

# Page 2
heading('01','为什么引入 Qwen3.5-9B')
figure(architecture,'图 1. 上部为微软公开披露的模型关系，下部为本次实际调用链路。图示不表示两家托管运行环境相同。',max_h=300)
para('微软官方发布说明称 Microsoft-Decision-1 基于 Qwen3.5-9B 进行后训练，用于预定义选项的决策评分。因此，第三个模型采用公开托管的 Qwen3.5-9B，比引入无关通用模型更贴近“专门决策训练带来什么变化”的研究问题。')
table([['模型','本次角色 / 接口'],['阿里云百炼','外部决策服务参照；systemone 接口'],['Microsoft Decision-1','微软决策模型；OpenRouter alpha/decisions'],['Qwen3.5-9B','公开基础模型参照；OpenRouter chat/completions']],[145,362])
callout('关系确认，不等于因果隔离','微软没有在本次材料中提供可复现的训练前检查点、训练数据与同硬件对照。我们测量的是三个实际可用服务的表现，不能从这些结果计算“后训练本身贡献了多少准确率或速度”。')
raw('来源：微软官方 2026-10-09 发布文章；OpenRouter Qwen3.5-9B 模型页。链接见第 13 页。','SmallCN')
page()

# Page 3
heading('02','同次运行，保持可比的输入与评分')
figure(flow,'图 2. 三家每题随机先后调用。正式记录之外，每家暖机 2 次，共 6 次；总记录 114 条。',max_h=265)
table([['设置','本次实际配置'],['数据','12 条中文人工示例；物理、化学、生物各 4 条；三标签各 4 条'],['运行','3 轮、并发 1、种子 42；每家长连接 Session；无自动重试'],['计时','发出 HTTP 前至完整响应体接收；不含 JSON 解析、本地评分及 0.1 秒停顿'],['Qwen 参数','固定 DeepInfra / BF16；temperature=0；max_tokens=256；reasoning.enabled=false'],['输出适配','Qwen 使用 strict JSON schema 只输出类别键；三家采用相同 state/questions'],['耗时总计',f"{meta['wall_seconds_including_warmup']:.2f} 秒（包含暖机和停顿）"]],[107,400])
para('准确率仅使用首轮 12 个独立输入；后两轮用于延迟与重复稳定性。Qwen 的固定系统提示和 JSON schema 用于输出适配；它与决策 API 的提示封装和计算方式仍不同。')
raw('连通性检查不纳入正式记录。原始业务 JSON 为 262-284 字符；Qwen 额外携带格式提示，实际 Token 不强行对齐。Qwen 全部 38 次调用的 reasoning_tokens 均为 0。','SmallCN')
page()

# Page 4
heading('03','准确率：微软多对一题，仍需更多样本')
figure(accuracy,'图 3. 上图每点代表一题，下图展示首轮准确率的不确定性。阿里云百炼与 Qwen 错误发生在不同样本上。',max_h=335)
table([['配对差（左减右）','观察差 / 百分点','95% bootstrap 区间']]+[[cn[p['left']]+' - '+cn[p['right']],f"{p['difference']*100:+.2f}",f"[{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}]"] for p in summary['paired']],[240,115,152])
para('微软相对另外两家均多对 1 道题，差 8.33 个百分点。阿里云百炼与 Qwen 得分相同，但并不代表行为相同：两家各有一道对方答对的错题。')
raw('2,000 次 bootstrap 按输入 ID 配对重采样；区间均包含零。三个区间为探索性比较，未做多重比较校正。12 道人工选题不能代表真实科研总体分布，区间无法修正选题偏差。','SmallCN')
page()

# Page 5
heading('04','三类错误与学科表现')
figure(confusion,'图 4. 每家首轮 12 题。行表示标准答案，列表示模型预测；非对角线格子是误分类。',max_h=216)
table([['指标']+[cn[n] for n in names],['合并 Macro-F1']+[f"{combined[n]['macro_f1']:.4f}" for n in names],
       ['物理正确数','3 / 4','4 / 4','4 / 4'],['化学正确数','4 / 4','4 / 4','4 / 4'],['生物正确数','4 / 4','4 / 4','3 / 4']],
      [171,112,112,112],winners=((2,1),))
sub('阿里云百炼：把数值反驳判为支持')
para('demo-002 的 A 电阻为 2 Ω、B 为 7 Ω，结论“A 大于 B”应被反驳；阿里云百炼返回支持。其反驳类召回率为 3/4，支持类精确率为 4/5。')
sub('Qwen3.5-9B：把缺少证据判为反驳')
para('demo-012 没有测量目标基因表达，只有 pH 记录，标准答案为信息不足；Qwen 返回反驳。其信息不足类召回率为 3/4，反驳类精确率为 4/5。')
callout('不要把 4 道题当成专业能力排名','每个学科仅 4 条短记录，主要考察给定证据与结论的关系。这不是对物理建模、化学机制或生物研究能力的完整测试。')
page()

# Page 6
heading('05','响应速度：三个服务的等待差距')
figure(latency_bars,'图 5. 两图横轴从零开始、刻度一致；每家 36 次正式请求，越短越快。',max_h=213)
table([['更快服务 / 对照服务','P50 降低','P95 降低']]+[[cn[a]+' / '+cn[b],f"{lower(a,b,'p50_ms'):.2f}%",f"{lower(a,b,'p95_ms'):.2f}%"] for a,b in [('bailian','microsoft'),('bailian','qwen'),('microsoft','qwen')]],[277,115,115])
figure(latency_scatter,'图 6. 左图为全部正式请求；右图为经验累积分布。请求失败率三家均为 0%，因此成功子集与全部尝试一致。',max_h=202)
raw('百分比均以对照服务耗时为分母。P95 为线性插值分位数，不是最大耗时或 SLA。网络、网关、服务排队及生成过程均包含在客户端耗时内。','SmallCN')
page()

# Page 7
heading('06','同题延迟、轮次变化与暖机')
figure(sameinput,'图 7. 同一题三轮的耗时中位数；每组展示三家，便于观察速度差异是否只来自单一道题。',max_h=217)
table([['轮次 / P50-P95 ms']+[cn[n] for n in names]]+[[str(i+1)]+[f(by_round[n][i]['p50'])+' / '+f(by_round[n][i]['p95']) for n in names] for i in range(3)],[150,119,119,119])
table([['暖机 / ms','第 1 次','第 2 次']]+[[cn[n]]+[f(r['latency_ms']) for r in rows if r['provider']==n and r['phase']=='warmup'] for n in names],[227,140,140])
para('暖机全部排除主要延迟统计。微软首个暖机调用明显较慢，提醒我们连接或首次调用状态可能影响用户体验；仅凭这一条不能定位为模型加载、网络还是排队。')
raw('Qwen 采用固定上游且禁止回退；微软本次实际为 Azure。重复完全相同输入可能受云端缓存影响；客户端无缓存，但本次未隔离服务端缓存。','SmallCN')
page()

# Page 8
heading('07','两个稳定错误：错在不同环节')
for sid,title in [('demo-002','数值冲突：2 Ω 小于 7 Ω'),('demo-012','证据缺失：未测量基因表达')]:
    s=next(s for s in samples if s['id']==sid)
    callout(sid+' / '+title,'<b>结论：</b>'+escape(s['state']['claim'])+'<br/><b>记录：</b>'+escape(s['state']['record'])+'<br/><b>标准答案：</b>'+lab[s['expected']['verdict']])
    table([['服务','三轮分类输出','首轮所选概率']]+[[cn[n], ' / '.join(lab[r['answers']['verdict']['predicted']] for r in measured[n] if r['sample_id']==sid),
        '不可用' if first[n][sid]['answers']['verdict']['pred_probability'] is None else f"{first[n][sid]['answers']['verdict']['pred_probability']:.6f}"] for n in names],[146,241,120])
para('三家重复分类一致率均为 24/24。阿里云百炼在 demo-002、Qwen 在 demo-012 上三轮都错，说明“输出稳定”不能替代“输出正确”。')
raw('没有证据不等于结论被反驳。科学应用中应为缺失测量保留“信息不足”类别；可计算数值关系应配合确定性规则校验。原始接口输出不能用于反推出模型内部错误原因。','SmallCN')
page()

# Page 9
heading('08','概率质量：只比较实际返回的概率')
table([['指标 / 首轮 n=12']+[cn[n] for n in names],
       ['Brier（越低越好）']+[f"{combined[n]['brier_sum']:.6f}" if combined[n]['brier_sum'] is not None else '不可用' for n in names],
       ['ECE（越低越好）']+[f"{combined[n]['ece_10_bins']:.6f}" if combined[n]['ece_10_bins'] is not None else '不可用' for n in names],
       ['有效概率分布数','12 / 12','12 / 12','0 / 12']], [171,112,112,112])
figure(calibration,'图 8. 左图为非空概率分箱，右图按所选类别概率排序展示覆盖率；Qwen 没有分布，因此不画概率曲线。',max_h=222)
sub('为什么 Qwen 不显示 Brier 和 ECE')
para('本次 Chat 基线只请求类别键。模型自报的“我有 90% 把握”不等于原生类别概率；将输出类别转换成 0/1 分布也会制造不存在的确定性。因此 Qwen 的相关指标应留空，而不是填零或按错误计分。')
sub('怎样理解另外两家的概率')
para('Brier 使用完整类别分布与标准标签计算平方误差；ECE 使用所选类别的概率与分箱实际正确率对比。两家接口 confidence 未用于跨厂商校准比较。阿里云百炼的高概率错题仍会进入高置信自动处理子集。')
raw('只有 12 题，概率分箱和覆盖率曲线十分不稳定。阈值必须在独立开发集选择，再在独立测试集验证，不能据此确定科研自动审核规则。','SmallCN')
page()

# Pages 10-11
for batch in range(2):
    heading('09' if batch==0 else '10','逐题审计：'+('001-006' if batch==0 else '007-012'))
    raw('以下为原始输入、本地标准答案和三家首轮输出。P 表示所选类别概率；Qwen 未请求概率，记为 N/A。','SmallCN')
    for s in samples[batch*6:(batch+1)*6]:
        title=s['id']+' | 标准：'+lab[s['expected']['verdict']]
        detail='<b>结论：</b>'+escape(s['state']['claim'])+'<br/><b>记录：</b>'+escape(s['state']['record'])
        for n in names:
            a=first[n][s['id']]['answers']['verdict'];pv='N/A' if a['pred_probability'] is None else f"{a['pred_probability']:.4f}"
            detail+='<br/>'+cn[n]+'：'+lab[a['predicted']]+'；P='+pv
        elements=[Paragraph(title,styles['CellBold']),Spacer(1,4),Paragraph(detail,styles['Cell'])]
        t=Table([[elements]],colWidths=[PAGE_W]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#FFF2E9') if s['id'] in ['demo-002','demo-012'] else colors.HexColor('#F2F6FA')),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([KeepTogether([t,Spacer(1,8)])])
    page()

# Page 12
heading('11','结论边界与下一轮实验')
table([['已经观察到','还不能据此推断'],['微软本次准确率最高；阿里云百炼最快','整体科学能力排名，或任何专业场景下的稳定胜出'],['微软比 Qwen 少错一题且响应更快','训练数据、模型改造、接口、硬件分别贡献了多少'],['Qwen 请求 BF16 与关闭思考；返回 reasoning_tokens=0','所有未观测的服务端执行细节完全一致或可控'],['三家本次 108 次正式请求均成功','长期可用性、高并发吞吐量、SLA 或异常恢复能力'],['三家三轮输出均稳定','预测正确、概率已校准或可以无人复核上线']],[250,257])
sub('面向科学项目的正式评测')
para('<b>数据：</b>至少从数百条独立、专业标注样本开始；加入数值与单位、反事实对照、证据缺失、长文献摘要。按论文或实验对象分组划分开发集和测试集，避免同源样本泄漏。')
para('<b>对照：</b>保留三家真实服务测试，并增加规则基线。若要研究后训练的因果效果，需要同一硬件、精度、推理引擎和受控提示条件，以及可访问的对应权重或检查点。')
para('<b>运行：</b>分开测试短/长输入、首次出现/重复输入、多个时段及并发等级；另设 Qwen 开启思考实验，不与本次关闭思考结果混合。')
para('<b>统计：</b>按独立研究对象计算配对差和置信区间，预注册主要指标及关键漏判成本。三方多重比较需预先选择校正方案；总分之外同时看错误类型、失败率和延迟尾部。')
callout('建议优先补充的两类题','围绕 demo-002 扩展数值比较与单位变体；围绕 demo-012 扩展“没有证据”和“相反证据”的区分。新增变体单独作为诊断集，不回填当前测试集来夸大改进。')
page()

# Page 13
heading('12','复现、原始证据与参考资料')
raw('cd C:\\decision\\decision-benchmark\npython benchmark.py --env-override run --providers bailian,microsoft,qwen --rounds 3 --warmup 2 --concurrency 1','CodeCN')
table([['项目','记录'],['实际模型', '\n'.join(stats[n]['served_models'][0] for n in names)],['实际上游','阿里云百炼；微软 Azure；Qwen DeepInfra'],['环境',f"Python {meta['python']}；requests {meta['requests_version']}；并发 1"],['软件验证','26 项离线测试通过；覆盖答案隔离、严格输出校验、概率缺失、三组配对统计'],['三方原始记录',str(RUN)]],[105,402])
raw('request_payload 保存实际发送业务载荷，raw_response 保存接口返回；均不含认证头。Qwen 的系统提示、JSON schema 和路由参数可逐条审计；标准答案只在本地 dataset.jsonl 内。','SmallCN')
for fn in ['dataset.jsonl','requests.jsonl']:
    sha=hashlib.sha256((RUN/fn).read_bytes()).hexdigest();raw(fn+' SHA-256: '+sha[:32]+'\n'+sha[32:],'SmallCN')
sub('指标口径')
raw('Accuracy = 首轮正确题数 / 12；Macro-F1 为合并三类 F1 的算术平均；P50/P95 使用每家 36 次正式成功请求和线性插值。暖机排除。概率缺失按不可用处理，服务失败则计入准确率错误。','SmallCN')
sub('来源与接口文档')
refs=[('微软：Decision-1 后训练基础与发布说明','https://commandline.microsoft.com/microsoft-decision-1-model-foundry/'),('Qwen 官方模型卡','https://huggingface.co/Qwen/Qwen3.5-9B'),('OpenRouter Qwen3.5-9B','https://openrouter.ai/qwen/qwen3.5-9b'),('结构化输出','https://openrouter.ai/docs/guides/features/structured-outputs'),('阿里云百炼决策接口','https://help.aliyun.com/zh/model-studio/decision-model-api')]
for title,url in refs:para(escape(title)+'：<link href="'+url+'" color="#236AA0">'+escape(url)+'</link>','SmallCN')
raw('来源核查与实验日期：2026-10-11。厂商宣传基准未混入本地实测。此前双模型报告属于不同运行，不能把其中历史延迟与本报告拼接比较。','SmallCN')

PDF=OUT/'decision_models_threeway_2026-10-11.pdf'
def decor(c,doc):
    c.saveState();c.setStrokeColor(colors.HexColor('#15858A'));c.setLineWidth(2.5);c.line(44,H-31,W-44,H-31)
    c.setFont('CN',8);c.setFillColor(colors.HexColor('#718292'));c.drawString(44,H-46,'DECISION BENCHMARK / 三方真实 API 评估')
    c.setStrokeColor(colors.HexColor('#DDE5EC'));c.setLineWidth(.5);c.line(44,37,W-44,37)
    c.setFont('CN',7.5);c.drawString(44,24,'2026-10-11 | 小样本探索性报告 | 每家 12 个独立输入')
    c.drawRightString(W-44,24,f'{doc.page} / 13');c.restoreState()
doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=65,bottomMargin=51,title='三方测试：阿里云百炼、Microsoft Decision-1 与 Qwen3.5-9B',author='Decision Benchmark')
doc.build(story,onFirstPage=decor,onLaterPages=decor)
reader=PdfReader(PDF);texts=[p.extract_text() for p in reader.pages]
assert len(texts)==13,len(texts)
assert all(len(t)>200 for t in texts)
assert all(v in texts[0] for v in ['91.67%','100.00%','93.6','538.6','1137.2'])
assert '逐题审计' in texts[9] and '逐题审计' in texts[10]
assert not any('\ufffd' in t for t in texts)
(WORK/'extracted.txt').write_text('\n\n'.join(texts),encoding='utf-8')
(WORK/'report_audit.json').write_text(json.dumps({'source_run':str(RUN),'pages':len(texts),'combined_metrics':combined,'source_hashes':{fn:hashlib.sha256((RUN/fn).read_bytes()).hexdigest() for fn in ['run.json','dataset.jsonl','requests.jsonl','summary.json']}},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(texts),'bytes':PDF.stat().st_size},ensure_ascii=False))
