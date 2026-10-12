import os
os.environ.setdefault('MPLCONFIGDIR', r'C:\decision\work\pdf-report\mpl-cache')
import sys, json, math, hashlib, statistics
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(r'C:\decision')
RUN=ROOT/'decision-benchmark/results/20261011T103758_581181Z'
WORK=ROOT/'work/pdf-report'
OUT=ROOT/'reports'
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'decision-benchmark'))
from bench.metrics import classification_metrics, question_records, percentile, wilson
meta=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
saved=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
samples=[json.loads(l) for l in (RUN/'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
rows=[json.loads(l) for l in (RUN/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
assert meta['mock'] is False and meta['status']=='complete'
assert len(samples)==12 and len(rows)==76
names=['bailian','microsoft']
cn={'bailian':'阿里云百炼','microsoft':'微软 / OpenRouter'}
labels=['support','refute','insufficient']
lab={'support':'支持','refute':'反驳','insufficient':'信息不足'}
col={'bailian':'#15858A','microsoft':'#3E5EB8'}
combined={}; measured={}; first={}; by_round={}
for name in names:
    groups,_=question_records(samples,rows,name)
    combined[name]=classification_metrics([r for g in groups.values() for r in g],labels)
    measured[name]=[r for r in rows if r['provider']==name and r['phase']=='measure']
    assert len(measured[name])==36 and all(r['ok'] for r in measured[name])
    first[name]={r['sample_id']:r for r in measured[name] if r['round']==0}
    by_round[name]=[{k:percentile([r['latency_ms'] for r in measured[name] if r['round']==i],q) for k,q in [('p50',.5),('p95',.95)]} for i in range(3)]
    assert combined[name]['accuracy']==saved['providers'][name]['classification_accuracy']

font_path=r'C:\Windows\Fonts\msyh.ttc'
fontManager.addfont(font_path)
plt.rcParams.update({'font.family':FontProperties(fname=font_path).get_name(),'axes.unicode_minus':False,'font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#22374B','text.color':'#22374B',
                     'xtick.color':'#566878','ytick.color':'#566878','axes.edgecolor':'#CBD5E1','savefig.facecolor':'white'})
pdfmetrics.registerFont(TTFont('CN',font_path,subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNB',r'C:\Windows\Fonts\msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')

def figsave(fig,name):
    p=WORK/(name+'.png'); fig.savefig(p,dpi=210,bbox_inches='tight'); plt.close(fig); return p

# Figure 1: counts, never mix repeat calls into independent sample counts.
fig,axes=plt.subplots(1,2,figsize=(7.0,2.55))
for ax,name in zip(axes,names):
    mat=np.array([[combined[name]['confusion'][g][p] for p in labels] for g in labels])
    ax.imshow(mat,cmap='Blues',vmin=0,vmax=4)
    for i in range(3):
        for j in range(3): ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>=3 else '#17364B',fontsize=13)
    ax.set_xticks(range(3),[lab[l] for l in labels]); ax.set_yticks(range(3),[lab[l] for l in labels])
    ax.set_xlabel('预测标签'); ax.set_ylabel('标准标签'); ax.set_title(cn[name]+'（首轮 n=12）',pad=10)
fig.tight_layout(); confusion_fig=figsave(fig,'confusion')

fig,axes=plt.subplots(1,2,figsize=(7.0,2.9),gridspec_kw={'width_ratios':[1,1.4]})
vals=[[r['latency_ms'] for r in measured[n]] for n in names]
bp=axes[0].boxplot(vals,positions=[1,2],widths=.45,patch_artist=True,showfliers=False)
for box,n in zip(bp['boxes'],names): box.set_facecolor(col[n]); box.set_alpha(.2)
for i,n in enumerate(names,1):
    jitter=np.random.default_rng(7).uniform(-.12,.12,len(vals[i-1]))
    axes[0].scatter(i+jitter,vals[i-1],s=12,alpha=.6,color=col[n])
axes[0].set_xticks([1,2],['阿里云百炼','微软']); axes[0].set_ylabel('客户端完整响应耗时 / ms'); axes[0].set_title('36 次正式请求 / 服务')
for n in names:
    x=np.sort([r['latency_ms'] for r in measured[n]])
    axes[1].step(x,np.arange(1,len(x)+1)/len(x),where='post',label=cn[n],color=col[n],lw=2)
axes[1].set_xlabel('客户端完整响应耗时 / ms'); axes[1].set_ylabel('已完成请求比例'); axes[1].set_ylim(0,1.04)
axes[1].axhline(.95,ls=':',color='#aaa',lw=1); axes[1].legend(loc='lower right',frameon=False,fontsize=8)
axes[1].set_title('经验累积分布（ECDF）'); fig.tight_layout(); latency_fig=figsave(fig,'latency')

fig,ax=plt.subplots(figsize=(7.0,3.25))
x=np.arange(12)
for offset,n in zip([-.19,.19],names):
    med=[percentile([r['latency_ms'] for r in measured[n] if r['sample_id']==s['id']],.5) for s in samples]
    ax.bar(x+offset,med,width=.36,color=col[n],label=cn[n])
ax.set_xticks(x,[s['id'].split('-')[1] for s in samples]); ax.set_xlabel('样本 ID 后三位'); ax.set_ylabel('三轮中位数 / ms')
ax.legend(frameon=False,fontsize=8); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
fig.tight_layout(); sample_latency_fig=figsave(fig,'sample_latency')

fig,axes=plt.subplots(1,2,figsize=(7.0,2.65))
for n in names:
    bins=[b for b in combined[n]['reliability_bins'] if b['count']]
    axes[0].scatter([b['mean_probability'] for b in bins],[b['accuracy'] for b in bins],s=[b['count']*25 for b in bins],color=col[n],label=cn[n],alpha=.75)
    for b in bins: axes[0].annotate('n='+str(b['count']),(b['mean_probability'],b['accuracy']),xytext=(-12,-14 if n=='bailian' else 7),textcoords='offset points',fontsize=7)
axes[0].plot([0,1],[0,1],ls='--',color='#aab4c0'); axes[0].set_xlim(.65,1.04); axes[0].set_ylim(.65,1.07)
axes[0].set_xlabel('所选类别的平均概率'); axes[0].set_ylabel('实际准确率'); axes[0].set_title('非空概率分箱（仅 12 样本）')
for n in names:
    cs=combined[n]['coverage']; axes[1].plot([c['actual_coverage']*100 for c in cs],[c['accuracy']*100 for c in cs],marker='o',color=col[n],label=cn[n])
axes[1].set_xlabel('实际自动处理比例 / %'); axes[1].set_ylabel('所选子集准确率 / %'); axes[1].set_ylim(40,105)
axes[1].set_title('按所选类别概率排序'); axes[1].legend(frameon=False,fontsize=8); fig.tight_layout(); calibration_fig=figsave(fig,'calibration')


# Visual reading guide: derive every measured value from the same archived run.
from matplotlib.patches import FancyBboxPatch
def diagram_box(ax, x, y, w, h, text, color='#15858A', size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',
                              facecolor=color+'12',edgecolor=color,linewidth=1.2,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.7)
def arrow(ax, start, end):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.5,color='#718292'))

fig,ax=plt.subplots(figsize=(7,5.4));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
diagram_box(ax,.22,.82,.56,.14,'同一组 12 条独立输入\n物理 4 条 · 化学 4 条 · 生物 4 条')
diagram_box(ax,.03,.57,.42,.15,'阿里云百炼\n直连接口',col['bailian'])
diagram_box(ax,.55,.57,.42,.15,'微软 Decision-1\nOpenRouter → Azure',col['microsoft'])
arrow(ax,(.40,.81),(.24,.73));arrow(ax,(.60,.81),(.76,.73))
diagram_box(ax,.17,.32,.66,.15,'12 输入 × 3 轮 × 2 服务 = 72 次正式请求\n另加 4 次暖机；共保存 76 条记录')
arrow(ax,(.24,.56),(.36,.48));arrow(ax,(.76,.56),(.64,.48))
diagram_box(ax,.03,.05,.42,.16,'准确率与概率质量\n每家仅使用首轮 12 题')
diagram_box(ax,.55,.05,.42,.16,'响应速度与稳定性\n每家使用全部 36 次正式请求',col['microsoft'],10)
arrow(ax,(.36,.31),(.24,.22));arrow(ax,(.64,.31),(.76,.22))
flow_fig=figsave(fig,'guide_flow')

fig,axes=plt.subplots(2,1,figsize=(7,5.1),gridspec_kw={'height_ratios':[1.15,1]})
for i,n in enumerate(names):
    y=1-i
    for j,sample in enumerate(samples):
        a=first[n][sample['id']]['answers']['verdict']
        ok=a['predicted']==sample['expected']['verdict']
        axes[0].scatter(j,y,s=155,color=col[n] if ok else '#CE5935',marker='o' if ok else 'X')
    axes[0].text(12,y,str(sum(first[n][q['id']]['answers']['verdict']['predicted']==q['expected']['verdict'] for q in samples))+'/12',va='center',fontsize=11)
axes[0].set_yticks([1,0],[cn[n] for n in names]);axes[0].set_xticks(range(12),[q['id'][-3:] for q in samples],fontsize=8)
axes[0].set_ylim(-.55,1.7);axes[0].set_xlim(-.5,13);axes[0].set_xlabel('样本编号；圆点 = 正确，橙色叉号 = 错误')
axes[0].set_title('每个标记代表一道独立题目',loc='left',pad=12)
for i,n in enumerate(names):
    m=combined[n];a=m['accuracy']*100;lo,hi=[v*100 for v in m['accuracy_ci95']]
    axes[1].errorbar(a,1-i,xerr=[[a-lo],[hi-a]],fmt='o',color=col[n],capsize=6,lw=2,ms=7)
    axes[1].text((lo+hi)/2,1-i+.22,f'{a:.2f}%   [ {lo:.2f}%, {hi:.2f}% ]',ha='center',fontsize=9)
axes[1].set_yticks([1,0],[cn[n] for n in names]);axes[1].set_xlim(0,105);axes[1].set_ylim(-.5,1.6)
axes[1].set_xlabel('准确率 / %；点 = 实测，横线 = 95% Wilson 区间')
axes[1].grid(axis='x',alpha=.15);fig.tight_layout(h_pad=2.7);accuracy_guide_fig=figsave(fig,'guide_accuracy')

fig,axes=plt.subplots(1,2,figsize=(7,2.9))
for ax,q,title in zip(axes,[.5,.95],['P50：典型等待水平','P95：较慢一端的等待水平']):
    for i,n in enumerate(names):
        v=percentile([r['latency_ms'] for r in measured[n]],q)
        ax.barh(1-i,v,color=col[n],height=.42);ax.text(v+15,1-i,f'{v:.1f}',va='center',fontsize=10)
    ax.set_yticks([1,0],['阿里云百炼','微软 / OpenRouter'],fontsize=8)
    ax.set_xlim(0,1050);ax.set_ylim(-.6,1.6);ax.set_xlabel('毫秒；越短越快');ax.set_title(title,fontsize=10)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.5);latency_guide_fig=figsave(fig,'guide_latency')

fig,ax=plt.subplots(figsize=(7,2.35));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
steps=[('发起请求',.01,.17),('网络与网关',.25,.22),('排队与服务处理',.54,.22),('收完整响应',.82,.17)]
for t,x,w in steps:diagram_box(ax,x,.39,w,.27,t,size=9)
for a,b in zip(steps,steps[1:]):arrow(ax,(a[1]+a[2]+.005,.525),(b[1]-.005,.525))
ax.annotate('',xy=(.98,.20),xytext=(.02,.20),arrowprops=dict(arrowstyle='<->',color='#15858A',lw=1.5))
ax.text(.5,.05,'客户端端到端耗时：包含整条服务路径',ha='center',fontsize=11)
ax.text(.5,.86,'流程示意；框宽不表示时间占比，网络往返未单独测量',ha='center',fontsize=9,color='#718292')
timing_guide_fig=figsave(fig,'guide_timing')

fig,axes=plt.subplots(1,2,figsize=(7,3.05),gridspec_kw={'width_ratios':[1,1.1]})
axes[0].bar(['样品 A','样品 B'],[2,7],color=['#15858A','#3E5EB8'],width=.5)
for i,v in enumerate([2,7]):axes[0].text(i,v+.15,f'{v} Ω',ha='center',fontsize=12)
axes[0].set_ylim(0,8.3);axes[0].set_ylabel('同温度下的电阻 / Ω');axes[0].set_title('原始记录：2 < 7')
axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
axes[1].axis('off');axes[1].set_xlim(0,1);axes[1].set_ylim(0,1)
diagram_box(axes[1],.05,.71,.90,.23,'待判断：A 的电阻大于 B',size=11)
arrow(axes[1],(.5,.68),(.5,.53))
diagram_box(axes[1],.05,.25,.90,.26,'2 > 7 不成立\n标准答案：反驳','#CE5935',12)
axes[1].text(.5,.09,'判断只依赖给定记录',ha='center',fontsize=10)
fig.tight_layout();error_guide_fig=figsave(fig,'guide_error')

fig,ax=plt.subplots(figsize=(7,2.2));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
diagram_box(ax,.02,.30,.27,.48,'返回“支持”\n所选概率 1.0000\n阿里云百炼首轮','#CE5935',10)
diagram_box(ax,.37,.30,.27,.48,'对照原始记录\n检查 2 与 7 的关系\n本地标准答案',size=10)
diagram_box(ax,.72,.30,.26,.48,'判定错误\n三轮结果均相同\n稳定不等于正确','#CE5935',10)
arrow(ax,(.30,.54),(.36,.54));arrow(ax,(.65,.54),(.71,.54))
ax.text(.5,.08,'接口概率描述模型输出；不能替代证据核验',ha='center',fontsize=11)
confidence_guide_fig=figsave(fig,'guide_confidence')

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCN',fontName='CN',fontSize=9.6,leading=15.8,spaceAfter=7,textColor=colors.HexColor('#263B4E'),wordWrap='CJK'))
styles.add(ParagraphStyle(name='SmallCN',parent=styles['BodyCN'],fontSize=8,leading=12,spaceAfter=5,textColor=colors.HexColor('#5B6D7C')))
styles.add(ParagraphStyle(name='TitleCN',fontName='CNB',fontSize=25,leading=34,spaceAfter=14,textColor=colors.HexColor('#123047'),wordWrap='CJK'))
styles.add(ParagraphStyle(name='SectionCN',fontName='CNB',fontSize=18,leading=25,spaceAfter=12,textColor=colors.HexColor('#123047'),wordWrap='CJK'))
styles.add(ParagraphStyle(name='SubCN',fontName='CNB',fontSize=11.5,leading=17,spaceBefore=9,spaceAfter=6,textColor=colors.HexColor('#19787F'),wordWrap='CJK'))
styles.add(ParagraphStyle(name='Cell',fontName='CN',fontSize=8.2,leading=12.3,wordWrap='CJK',textColor=colors.HexColor('#263B4E')))
styles.add(ParagraphStyle(name='CellBold',parent=styles['Cell'],fontName='CNB'))
styles.add(ParagraphStyle(name='CodeCN',fontName='CN',fontSize=8.2,leading=13,spaceAfter=6,wordWrap='CJK',backColor=colors.HexColor('#F1F5F9'),borderPadding=7))
W,H=A4; PAGE_W=W-88
story=[]

def para(text,style='BodyCN'):
    story.append(Paragraph(text,styles[style]))

def raw(text,style='BodyCN'):
    para(escape(str(text)).replace('\n','<br/>'),style)

def heading(number,title):
    raw(number+'  /  '+title,'SectionCN')

def sub(title): raw(title,'SubCN')

def table(data,widths=None,winners=()):
    cells=[]
    for i,row in enumerate(data):
        cells.append([Paragraph(escape(str(v)).replace('\n','<br/>'),styles['CellBold' if i==0 or (j,i) in winners else 'Cell']) for j,v in enumerate(row)])
    t=Table(cells,colWidths=widths or [PAGE_W/len(data[0])]*len(data[0]),hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF5')),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
      ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#B9CAD8')),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#DDE5EC')),
      ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFC')])]))
    for col_idx,row_idx in winners:
        t.setStyle(TableStyle([('BACKGROUND',(col_idx,row_idx),(col_idx,row_idx),colors.HexColor('#DCF2DC'))]))
    story.extend([t,Spacer(1,10)])

def callout(title,text):
    cell=[Paragraph(escape(title),styles['SubCN']),Paragraph(text,styles['BodyCN'])]
    t=Table([[cell]],colWidths=[PAGE_W]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#EAF5F4')),
    ('BOX',(0,0),(-1,-1),.5,colors.HexColor('#BDDDDA')),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    story.extend([t,Spacer(1,10)])

def figure(path,caption,max_h=235):
    from PIL import Image as PImage
    iw,ih=PImage.open(path).size
    width=PAGE_W;height=width*ih/iw
    if height>max_h:width*=max_h/height;height=max_h
    story.append(Image(str(path),width=width,height=height,hAlign='CENTER'))
    raw(caption,'SmallCN')

def page(): story.append(PageBreak())

def pct(v): return f'{100*v:.2f}%'

def ci(v): return f'{100*v[0]:.2f}% - {100*v[1]:.2f}%'

def f(v): return f'{v:.1f}'

start=datetime.fromisoformat(meta['created_at']).astimezone(timezone(timedelta(hours=8)))
end=datetime.fromisoformat(meta['finished_at']).astimezone(timezone(timedelta(hours=8)))

# 1
raw('真实 API 小样本测试  |  技术评估报告','SubCN')
raw('决策模型准确率与\n响应速度对比','TitleCN')
raw('阿里云百炼 decision-model-preview\nMicrosoft Decision-1（通过 OpenRouter / Azure）','BodyCN')
raw('测试时间：'+start.strftime('%Y-%m-%d %H:%M:%S')+' - '+end.strftime('%H:%M:%S')+'（北京时间）','SmallCN')

p50_a=percentile([r['latency_ms'] for r in measured['bailian']],.5)
p50_m=percentile([r['latency_ms'] for r in measured['microsoft']],.5)
p95_a=percentile([r['latency_ms'] for r in measured['bailian']],.95)
p95_m=percentile([r['latency_ms'] for r in measured['microsoft']],.95)
gain50=(1-p50_a/p50_m)*100; gain95=(1-p95_a/p95_m)*100
callout('本次结果：微软更准，阿里云百炼更快',
    '<b>准确率：微软领先 8.33 个百分点。</b>微软 100.00%，阿里云百炼 91.67%，即微软多答对 1 道题（12 道题中）。<br/>'
    +f'<b>响应速度：阿里云百炼领先。</b>相较微软，P50 耗时降低 {gain50:.2f}%（少等 {p50_m-p50_a:.1f} ms），P95 耗时降低 {gain95:.2f}%（少等 {p95_m-p95_a:.1f} ms）。<br/>'
    +'<b>请求成功与重复稳定性：两家持平。</b>正式请求均无失败，后两轮分类均与首轮一致。')
table([['核心指标','阿里云百炼','微软 / OpenRouter','本次优势 / 差距'],
       ['首轮正确数 / 独立题数','11 / 12','12 / 12','微软多对 1 题'],
       ['首轮准确率（越高越好）','91.67%','100.00%','微软高 8.33 个百分点'],
       ['客户端 P50（越低越好）','107.7 ms','550.6 ms',f'阿里云百炼低 {gain50:.2f}%'],
       ['客户端 P95（越低越好）','368.1 ms','854.4 ms',f'阿里云百炼低 {gain95:.2f}%'],
       ['正式请求失败率（越低越好）','0%（0 / 36）','0%（0 / 36）','持平'],
       ['重复分类一致率（越高越好）','100%（24 / 24）','100%（24 / 24）','持平']],
      [174,103,112,118],winners=((2,1),(2,2),(1,3),(1,4)))
raw('浅绿色单元格 = 本次实测更优；持平项不着色。耗时降低比例以微软耗时为分母；准确率差使用百分点。','SmallCN')
para('<b>如何选择：</b>就本次表现看，重视判断正确性可优先进一步验证微软；重视响应速度可优先进一步验证阿里云百炼。两家各有优势，本报告不设未经验证的综合赢家。')
raw('适用范围：仅 12 条人工科学语境示例，准确率差异来自 1 题，配对区间包含零，尚不能证明总体准确率谁更高。速度比较覆盖客户端完整接入路径，不等于模型内部计算速度。','SmallCN')
sub('阅读导航')
raw('第 2-5 页：四组图解；第 6 页：设计与样本；第 7 页：准确率；第 8-9 页：延迟；第 10 页：概率与复核；第 11 页：错误案例；第 12-13 页：逐题审计；第 14 页：下一轮方案；第 15 页：复现与资料。','SmallCN')
raw('数据来源：本地真实请求记录。模型不是本地模拟模式；示例内容是人工构造，尚未运行 PubMedQA 正式评测。','SmallCN')
page()

# Four-page visual reading guide.
heading('图解 A','一次测试如何产生两类结论')
figure(flow_fig,'图 A. 同一组题目配对调用两家服务；评分与计时使用不同的统计口径。',max_h=385)
callout('先分清“题数”和“请求数”','12 道题重复三次，仍然只有 <b>12 个独立输入</b>。重复请求可以观察延迟与稳定性，但不能把准确率样本量写成 36。')
para('例如 demo-002 在三轮都回答错误，代表同一道题上的稳定错误；它在首轮准确率中只计一次。本报告先用相同输入保证可比，再按各指标的定义选择记录。')
raw('暖机用于预热连接与服务链路，其 4 条记录单列，不进入准确率和主要延迟统计。','SmallCN')
page()
heading('图解 B','把准确率还原成一道道题')
figure(accuracy_guide_fig,'图 B. 上图展示逐题正确性，下图显示首轮准确率与 Wilson 区间；均使用每家 12 个独立输入。',max_h=380)
callout('91.67% 与 100% 的差别来自一道题','阿里云百炼 11/12，微软 12/12。每增减一道正确题，准确率就变化 <b>8.33 个百分点</b>；本次观测差异全部来自 demo-002。')
para('横线较长说明小样本估计仍不精确。12/12 的实测结果不意味着模型以后不会出错，也不意味着在物理、化学、生物三个领域都达到 100% 准确率。')
raw('Wilson 区间展示单家比例的不确定性；两家优劣应查看同题配对比较。本次配对差的 95% bootstrap 区间包含零，详见第 7 页。','SmallCN')
page()
heading('图解 C','响应速度应当怎样阅读')
figure(latency_guide_fig,'图 C1. 横轴从零开始，两图使用同一刻度。每家 n=36 次正式请求，标注为实测分位数。',max_h=222)
table([['读数','通俗含义','本次分位数比值'],['P50','把耗时排序，位于中间的水平','微软 / 阿里云百炼 ≈ 5.11'],['P95','约 95% 请求耗时所在的上界水平','微软 / 阿里云百炼 ≈ 2.32']],[60,270,177])
raw('小样本分位数采用线性插值，因此 P95 不一定恰好是某个请求的耗时；它不是最长耗时或服务承诺。','SmallCN')
figure(timing_guide_fig,'图 C2. 计时边界示意：对比的是从当前客户端感受到的完整服务响应。',max_h=171)
para('两家走不同接入路径：阿里云百炼为直连，微软经 OpenRouter 转至 Azure。没有分段计时证据，因此不能把上述比值解释为模型内部计算速度相差多少倍。')
page()
heading('图解 D','高置信错误为什么值得关注')
figure(error_guide_fig,'图 D1. 将 demo-002 的文字记录画成柱形图，结论与证据的冲突就更容易看清。',max_h=242)
figure(confidence_guide_fig,'图 D2. 将模型输出与证据核验分开：很高的返回概率也可能伴随错误。',max_h=175)
para('阿里云百炼首轮将“支持”的概率返回为 1.0000，却与给定数值冲突；微软首轮返回“反驳”，所选类别概率约为 0.99857。这只描述本题上的表现，不能推广为任何模型永远可靠或不可靠。')
callout('如何用于科学项目','对数值大小、单位和阈值等可计算条件，先做确定性校验；对证据缺失或模型与规则冲突的案例，交由人工复核。阈值需在独立数据上验证，不能仅凭这 12 道题设置。')
page()

# 2
heading('01','测试设计与数据范围')
table([['项目','本次实际配置'],['独立输入 / 学科','12 条；物理、化学、生物各 4 条'],['标签分布','支持、反驳、信息不足各 4 条'],['请求结构','每个输入 1 个 choice 问题，固定 3 个选项'],['样本长度','state 与 questions 的 JSON 长度 262-284 字符'],['重复次数 / 并发','3 轮；并发 1；每个服务 2 次暖机'],['正式 / 暖机请求','72 次正式 + 4 次暖机 = 76 次已记录请求'],['随机与请求策略','种子 42；每轮随机样本顺序；每题随机服务先后'],['连接与重试','每个服务复用 Session；无自动重试；请求后停顿 0.1 秒'],['超时与计时','连接/读取等待 60 秒；完整 HTTP 响应体到达后停止计时']],[138,369])
sub('同题配对，而非比较各厂商宣传数字')
para('两家得到相同的业务状态、问题说明和选项描述。标准答案保存在 expected 字段，仅供本地评分；请求构造显式只发送 model、state、questions。本次 OpenRouter 未指定 provider.only，实际成功响应记录的上游为 Azure。')
sub('评分与延迟采用不同样本口径')
para('准确率、Macro-F1、混淆矩阵和校准指标只使用第 1 轮的 12 个独立输入。延迟使用全部 3 轮共 36 次正式请求 / 服务。后两轮只用于延迟和重复稳定性，不能将答题样本量扩大为 36。暖机与此前连接检查不计入正式结果。')
sub('样本代表性')
para('题目为中文短文本，重点是记录与结论的支持关系、简单数值大小比较、缺失证据和否定信息。它们并非经过独立专家验证的科学能力基准，也不覆盖长论文、多模态、开放式推理、noul、score 或高并发场景。')
raw('运行总墙钟时间约 38.85 秒，包含暖机与请求间停顿；不能用 72 / 38.85 作为任一家服务的最大吞吐量。','SmallCN')
page()
# 3
heading('02','准确率、分类误差与统计区间')
table([['指标（首轮 n=12）','阿里云百炼','微软 / OpenRouter'],['准确率',pct(combined['bailian']['accuracy']),pct(combined['microsoft']['accuracy'])],['95% Wilson 区间',ci(combined['bailian']['accuracy_ci95']),ci(combined['microsoft']['accuracy_ci95'])],['合并三类 Macro-F1',f"{combined['bailian']['macro_f1']:.4f}",'1.0000'],['物理 / 化学 / 生物正确数','3/4 · 4/4 · 4/4','4/4 · 4/4 · 4/4']],[177,165,165])
figure(confusion_fig,'图 1. 三个学科采用同一标签语义，因此可以合并混淆矩阵。唯一误分类为“反驳”被预测成“支持”。',max_h=208)
para('阿里云百炼“支持”类别精确率为 4/5 = 80%，召回率为 4/4 = 100%；“反驳”召回率为 3/4 = 75%；“信息不足”召回率为 4/4 = 100%。微软在本次样本中三类均全部正确。')
sub('不能仅凭 100% 宣称模型更强')
para('配对差定义为“阿里云百炼准确率 - 微软准确率”：观察值 <b>-8.33 个百分点</b>，按独立输入进行 2,000 次配对 bootstrap，95% 区间为 <b>[-25.00, 0.00] 个百分点</b>。区间包含零；只有 1 个分歧样本，检验效能很低。')
para('作为补充核查，两家分歧的配对计数为 0 对 1，双侧精确 McNemar 检验 p=1.00。该结果不证明两家能力相等，只说明这批样本不足以建立可靠的总体优劣结论。')
raw('合并 Macro-F1 按整体三类计算；它不同于先计算三个学科的 Macro-F1 再平均。Wilson 和 bootstrap 依赖抽样假设，无法修正人工选题产生的代表性偏差。','SmallCN')
page()
# 4
heading('03','客户端延迟与响应分布')
lat_table=[['统计量（ms；n=36 / 服务）','阿里云百炼','微软 / OpenRouter']]
for title,fn in [('最小值',min),('均值',statistics.mean),('P50',lambda v:percentile(v,.5)),('P90',lambda v:percentile(v,.9)),('P95',lambda v:percentile(v,.95)),('最大值',max)]:
    lat_table.append([title]+[f(fn([r['latency_ms'] for r in measured[n]])) for n in names])
table(lat_table,[207,150,150])
figure(latency_fig,'图 2. 左：逐请求散点与箱线图；右：经验累积分布。36 个请求来自重复输入，不等于 36 个独立业务样本。',max_h=226)
para('本次所有 36 次阿里云百炼正式请求均低于 389 ms；所有微软正式请求均高于 496 ms，两组观察范围没有重叠。这个现象说明当前客户端和接入条件下存在明显体验差异，但不足以推断其它地域、时段或负载下的结果。')
sub('计时边界')
para('起点在 Python 发起 HTTP 请求之前，终点在完整响应体接收之后，排除 JSON 解析、本地打分和请求后的 0.1 秒停顿。包括网络往返、可能的连接建立、网关、排队和服务处理。服务端 latency_ms 即便存在也不参与横向对比，不与客户端耗时相减估算纯网络耗时。')
raw('本次无失败和超时，成功请求与全部尝试的延迟分位数相同；后续若有失败，必须同时展示失败率，以免成功子集形成选择偏差。','SmallCN')
page()
# 5
heading('04','轮次、暖机与同题延迟')
rtable=[['轮次（每家 12 次）','阿里云百炼 P50 / P95 ms','微软 P50 / P95 ms']]
for i in range(3): rtable.append([str(i+1)]+[f(by_round[n][i]['p50'])+' / '+f(by_round[n][i]['p95']) for n in names])
table(rtable,[157,175,175])
figure(sample_latency_fig,'图 3. 同一输入在三轮中的耗时中位数。横轴对应附录样本编号；这不是专业领域整体速度排名。',max_h=242)
sub('观察到的轮次变化')
para('阿里云百炼首轮 P95 约 380.0 ms，后两轮分别约 112.2 和 118.2 ms；微软三轮 P50 分别约 562.4、548.2、539.2 ms。后续轮次的变化可能与连接状态、云端缓存、负载或路由有关；本次未控制这些因素，不能确认单一原因。')
table([['已记录暖机（ms）','第 1 次','第 2 次'],['阿里云百炼','680.4','108.6'],['微软 / OpenRouter','2562.7','911.0']],[207,150,150])
para('两家已预热的 Session 在正式请求中复用。暖机时间单列并排除主要延迟指标；暖机本身也不等于完全无缓存的首次调用，因为此前做过连接检查。重复完全相同的输入可能命中服务缓存，后续正式研究应分开设计首次出现输入与重复输入两种工作负载。')
page()
# 6
heading('05','概率质量与人工复核策略')
table([['首轮合并指标（n=12）','阿里云百炼','微软 / OpenRouter'],['Brier score（各类平方误差之和）',f"{combined['bailian']['brier_sum']:.6f}",f"{combined['microsoft']['brier_sum']:.6f}"],['ECE（10 个等宽分箱）',f"{combined['bailian']['ece_10_bins']:.6f}",f"{combined['microsoft']['ece_10_bins']:.6f}"],['概率分布有效数','12 / 12','12 / 12']],[247,130,130])
figure(calibration_fig,'图 4. 两项指标均使用“所选类别的概率”，不直接比较厂商自定义 confidence。分箱点大小与样本数有关。',max_h=204)
para('Brier 和 ECE 越低，说明在当前标签样本上预测概率与结果越一致。但本次只有 12 题、概率高度集中，ECE 对分箱很敏感；这些数值不能证明模型在真实科学数据上已完成可靠校准。阿里云百炼的高置信错误对 Brier 影响很大。')
table([['目标自动处理比例','实际纳入数 / 比例','阿里云百炼子集准确率','微软子集准确率'],['40%','4 / 12 = 33.33%','75.00%','100.00%'],['80%','9 / 12 = 75.00%','88.89%','100.00%'],['100%','12 / 12 = 100%','91.67%','100.00%']],[127,146,117,117])
para('小样本下按 floor(目标比例 × 12) 取整，因此“80%”实际只接受 9 条。阿里云百炼错误题的返回概率为 1.0，会排在高概率端；仅过滤低概率样本无法移除它。并列概率按样本 ID 排序，这会影响小样本覆盖率曲线，不能把曲线当成已验证的上线规则。')
raw('正式阈值需在独立开发集选择，再在未调参测试集验证；模型返回概率不是科学命题为真的客观概率。','SmallCN')
page()
# 7
heading('06','唯一错误案例：数值反驳被判为支持')
case=samples[1]
callout('样本 demo-002', '<b>结论：</b>'+escape(case['state']['claim'])+'<br/><b>记录：</b>'+escape(case['state']['record'])+'<br/><b>标准答案：</b>反驳（refute）。在相同测量条件下 2 小于 7，与“A 大于 B”直接冲突。')
table([['服务 / 首轮','输出','P(支持)','P(反驳)','confidence'],['阿里云百炼','支持','1.000000','0.000000','1.000000'],['微软','反驳','0.000261','0.998570','0.997855']],[111,61,110,110,115])
case_rows=[['轮次','阿里云百炼输出 / 正确性','微软输出 / 正确性','阿里云百炼 / 微软耗时 ms']]
for i in range(3):
    a=next(r for r in measured['bailian'] if r['sample_id']=='demo-002' and r['round']==i)
    b=next(r for r in measured['microsoft'] if r['sample_id']=='demo-002' and r['round']==i)
    case_rows.append([i+1,'支持 / 错误','反驳 / 正确',f(a['latency_ms'])+' / '+f(b['latency_ms'])])
table(case_rows,[50,141,141,175])
sub('重复稳定，不等于可靠正确')
para('两家后两轮相对首轮的分类一致率均为 24/24。阿里云百炼这道题的三次输出却都错误，因此“重复一致”只能描述输出稳定性，不能替代准确率，也不能将高 confidence 作为跳过核验的充分理由。返回的 1.0 是接口数值，仍可能涉及输出精度，不代表客观上的绝对确定。')
sub('可以推断什么，不能推断什么')
para('可以确认：当前模型、提示词和输入表示组合在该题上出现稳定误判。不能仅由答案推断模型内部推理过程，或断定是中文能力、数字处理、指令理解中的哪一环导致。接口没有提供可用于定位原因的书面推理。')
sub('下一轮有针对性的排查')
para('固定其它条件，构造 A&lt;B、A=B、A&gt;B 以及缺失 B 的平衡样本；交换实体名称、选项顺序和数值位置，分别测试中文与英文。对大小关系先用确定性代码计算，作为基线与约束；新增变体应作为诊断集，不回填当前测试集来夸大改进。')
page()
# 8-9 full audit
for batch in range(2):
    heading('07' if batch==0 else '08','逐题审计：'+('样本 001-006' if batch==0 else '样本 007-012'))
    raw('每题展示原始结论、记录、本地标准答案和首轮输出。P 为所选类别概率，C 为接口 confidence；二者含义不同。','SmallCN')
    for s in samples[batch*6:(batch+1)*6]:
        a=first['bailian'][s['id']]['answers']['verdict']; b=first['microsoft'][s['id']]['answers']['verdict']
        title=s['id']+'  |  '+{'demo_physics_evidence':'物理','demo_chemistry_evidence':'化学','demo_biology_evidence':'生物'}[s['task']]+'  |  标准：'+lab[s['expected']['verdict']]
        elements=[Paragraph(escape(title),styles['CellBold']),Spacer(1,4),Paragraph('<b>结论：</b>'+escape(s['state']['claim'])+'<br/><b>记录：</b>'+escape(s['state']['record']),styles['Cell']),Spacer(1,4),Paragraph('阿里云百炼：'+lab[a['predicted']]+f"；P={a['pred_probability']:.4f}，C={a['confidence']:.4f}"+'<br/>微软：'+lab[b['predicted']]+f"；P={b['pred_probability']:.4f}，C={b['confidence']:.4f}",styles['Cell'])]
        t=Table([[elements]],colWidths=[PAGE_W]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#FFF2E9') if s['id']=='demo-002' else colors.HexColor('#F2F6FA')),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
        story.extend([KeepTogether([t,Spacer(1,10)])])
    page()
# 10
heading('09','结论边界与下一轮正式评估')
sub('本报告已经支持的判断')
para('两条真实服务链路已跑通；在当前中文短文本、单问题、串行负载下，阿里云百炼客户端延迟更低；微软在这批示例上少错一道题。所有正式请求均成功返回有效结构，且分类结果在三轮内稳定。')
sub('尚未解决的问题')
table([['限制','影响'],['只有 12 条人工示例','不能估计真实科研任务的总体水平；3 个学科各 4 条更不足以排名'],['输入只有 262-284 字符','不能外推到长摘要、全文、多证据或复杂数值推理'],['一次运行、一个客户端','不能区分接入路径与模型计算；无法代表长期尾延迟或 SLA'],['同题重复、缓存未控制','后续轮次可能获得缓存或暖连接收益；不属于纯冷请求对比'],['概率高且类别简单','校准曲线不稳定，不能直接制定高风险自动处理阈值'],['模型别名与路由可能变化','需要保存实际 served_model、provider、时间和原始输出']],[160,347])
sub('建议的下一轮实验')
para('<b>数据：</b>使用 500 条独立、标注可靠的科学任务；文献判断可从 PubMedQA 起步，保留不含答案结论的上下文。按论文/实验对象分组划分开发集与测试集，补充近期未用于调参的专业样本。')
para('<b>控制：</b>固定提示词、选项语义、输入语言、阈值和版本；区分短/中/长输入。首次出现输入与重复输入分别统计，至少在多个时段复测；并发 1 与并发 4 分开运行。')
para('<b>指标：</b>准确率、Macro-F1、关键类别漏判、Brier、风险-覆盖率、P50/P95 和失败率。增加规则系统/专业分类器基线；配对比较按独立实验对象重采样。不要用测试集结果继续调提示词。')
raw('建议运行：python benchmark.py --env-override run --dataset data\\pubmedqa\\test.jsonl --limit 500 --rounds 3 --warmup 10 --concurrency 1','CodeCN')
raw('先执行 prepare-pubmedqa。该命令默认生成自定义分层划分，不能作为 PubMedQA 官方榜单成绩。以上是建议，尚未执行，也不属于本报告的测量结果。','SmallCN')
page()
# 11
heading('10','复现、指标定义与数据来源')
sub('复现本次小样本测试')
raw('cd C:\\decision\\decision-benchmark\npython benchmark.py --env-override run --rounds 3 --warmup 2 --concurrency 1','CodeCN')
para('本次连接检查发现旧进程环境变量覆盖了 .env 中的新阿里云百炼密钥；使用显式 --env-override 后成功。该选项只改变本次进程配置优先级，不修改系统环境变量。修复后 19 项自动测试通过。此过程不纳入 72 次正式请求统计。')
table([['复现信息','记录'],['实际微软模型','microsoft/microsoft-decision-1-20261009'],['实际阿里云百炼模型','decision-model-preview（返回未提供更细快照号）'],['地域 / 路由','阿里云百炼北京；微软由 OpenRouter 转 Azure，响应未给出具体推理地域'],['运行环境','Windows 10 build 19045；Python 3.12.7；requests 2.32.5'],['随机种子 / bootstrap','42 / 2,000 次；按输入 ID 配对重采样']],[140,367])
sub('指标定义')
para('Accuracy = 正确数 / 独立测试题数；Macro-F1 = 各标签 F1 的算术平均；Brier = 各类别预测概率与真实 one-hot 标签的平方差之和，再对样本取平均；ECE = 各概率分箱的概率-准确率绝对差按样本数加权。分位数采用排序后的线性插值。', 'SmallCN')
sub('原始证据与审计')
raw(str(RUN),'SmallCN')
para('run.json：运行配置与时间；dataset.jsonl：12 条实际输入和本地标准答案；requests.jsonl：76 条含暖机的完整记录；summary.json：原程序统计。PDF 中的合并 Macro-F1、合并校准、按轮次延迟和精确 McNemar 为基于同一记录的补充计算。','SmallCN')
for filename in ['dataset.jsonl','requests.jsonl']:
    sha=hashlib.sha256((RUN/filename).read_bytes()).hexdigest()
    raw(filename+' SHA-256: '+sha[:32]+'\n'+sha[32:],'SmallCN')
sub('方法与接口参考')
refs=[('阿里云百炼决策接口','https://help.aliyun.com/zh/model-studio/decision-model-api'),('OpenRouter 微软模型页','https://openrouter.ai/microsoft/microsoft-decision-1'),('PubMedQA（Jin 等，2019）','https://aclanthology.org/D19-1259/'),('概率校准（Guo 等，ICML 2017）','https://proceedings.mlr.press/v70/guo17a.html')]
for title,url in refs: para(escape(title)+'：<link href="'+url+'" color="#236AA0">'+escape(url)+'</link>','SmallCN')
raw('本报告未引用厂商宣传性能作为实测结果；所有比较数字均来自指定本地运行。未包含 API Key、.env 内容或账户密钥。','SmallCN')

PDF=OUT/'decision_models_benchmark_2026-10-11.pdf'
def page_decor(c,doc):
    c.saveState();c.setStrokeColor(colors.HexColor('#15858A'));c.setLineWidth(2.5);c.line(44,H-31,W-44,H-31)
    c.setFont('CN',8);c.setFillColor(colors.HexColor('#718292'));c.drawString(44,H-46,'DECISION BENCHMARK  /  真实 API 评估')
    c.setStrokeColor(colors.HexColor('#DDE5EC'));c.setLineWidth(.5);c.line(44,37,W-44,37)
    c.setFont('CN',7.5);c.drawString(44,24,'2026-10-11  |  小样本探索性报告  |  n=12')
    c.drawRightString(W-44,24,f'{doc.page} / 15');c.restoreState()
doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=65,bottomMargin=51,
                      title='阿里云百炼与微软 Decision-1：真实 API 准确率与响应速度对比',author='Decision Benchmark',subject='2026-10-11 小样本探索性测试；12 条独立示例')
doc.build(story,onFirstPage=page_decor,onLaterPages=page_decor)
reader=PdfReader(PDF)
assert len(reader.pages)==15, f'Expected 15 pages, got {len(reader.pages)}'
texts=[p.extract_text() for p in reader.pages]
assert all(len(t)>200 for t in texts)
assert '91.67%' in texts[0] and '100.00%' in texts[0]
assert 'demo-002' in texts[10]
assert '逐题审计' in texts[11] and '逐题审计' in texts[12]
assert not any('bailian' in t.lower() for t in texts)
assert sum('阿里云百炼' in t for t in texts) >= 12
assert not any('\ufffd' in t for t in texts)
(WORK/'extracted.txt').write_text('\n\n'.join(texts),encoding='utf-8')
extra={'source_run':str(RUN),'pdf':str(PDF),'pages':len(reader.pages),'combined_metrics':combined,'round_latency':by_round,
       'source_hashes':{fn:hashlib.sha256((RUN/fn).read_bytes()).hexdigest() for fn in ['run.json','dataset.jsonl','requests.jsonl','summary.json']}}
(WORK/'report_audit.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(reader.pages),'bytes':PDF.stat().st_size},ensure_ascii=False))