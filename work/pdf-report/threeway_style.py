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

font_path=r'C:\Windows\Fonts\msyh.ttc'
fontManager.addfont(font_path)
plt.rcParams.update({'font.family':FontProperties(fname=font_path).get_name(),'axes.unicode_minus':False,'font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#22374B','text.color':'#22374B',
                     'xtick.color':'#566878','ytick.color':'#566878','axes.edgecolor':'#CBD5E1','savefig.facecolor':'white'})
pdfmetrics.registerFont(TTFont('CN',font_path,subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNB',r'C:\Windows\Fonts\msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')

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

