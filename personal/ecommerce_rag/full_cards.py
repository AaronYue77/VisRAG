"""On-demand official photos and source-faithful compact PDF evidence cards."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import ssl
import threading
import time
import urllib.error
import urllib.request
from full_catalog import ROOT,connect,values
from common import WORKSPACE,file_hash,write_json,write_jsonl,operation,timestamp

PDF_VERSION='full-card-v1'
_font_lock=threading.Lock()


def display_values(item,key):
    rows=item.get(key,[])
    if not isinstance(rows,list):return []
    rows=[r for r in rows if isinstance(r,dict) and r.get('value') is not None]
    english=[r for r in rows if str(r.get('language_tag','')).startswith('en')]
    chosen=english or [r for r in rows if r.get('language_tag')=='zh_CN'] or rows[:1]
    return list(dict.fromkeys(str(r['value']) for r in chosen))


def evidence_fields(item):
    result=[]
    for key in ('brand','color','material','model_number','product_type','fabric_type','style','size'):
        for value in display_values(item,key):result.append((key,value))
    for key in ('width','height','length'):
        value=item.get('item_dimensions',{}).get(key)
        if isinstance(value,dict) and 'value' in value:result.append((key,str(value['value'])+' '+str(value.get('unit',''))))
    for value in item.get('item_weight',[]):
        if isinstance(value,dict) and 'value' in value:result.append(('weight',str(value['value'])+' '+str(value.get('unit',''))))
    return result


def asset_key(row):return hashlib.sha256(row['listing_key'].encode()).hexdigest()[:24]


def photo(row,allow_download=True):
    db=connect();image=db.execute('SELECT * FROM images WHERE image_id=?',(row['main_image_id'],)).fetchone();db.close()
    if image is None:return None,{'status':'no_source_image','image_id':row['main_image_id']}
    image=dict(image);directory=ROOT/'images';directory.mkdir(exist_ok=True)
    path=directory/(hashlib.sha256(image['image_id'].encode()).hexdigest()+Path(image['path']).suffix)
    meta=path.with_suffix(path.suffix+'.json')
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text('utf-8'))
        if file_hash(path)!=record['sha256']:raise ValueError('Image cache changed')
        return path,{**record,'cache_hit':True}
    if not allow_download:return None,{'status':'not_downloaded','image_id':row['main_image_id']}
    url='https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/original/'+image['path']
    started=time.perf_counter();errors=[]
    for attempt in range(2):
        try:
            context=ssl.create_default_context()
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}) if attempt else urllib.request.ProxyHandler(),urllib.request.HTTPSHandler(context=context))
            req=urllib.request.Request(url,headers={'User-Agent':'personal-ABO-full-RAG/1.0'})
            with opener.open(req,timeout=18) as response:
                data=response.read(25*1024*1024+1)
            if not 100<len(data)<=25*1024*1024:raise ValueError('Invalid source image size')
            partial=path.with_suffix(path.suffix+'.part');partial.write_bytes(data)
            from PIL import Image
            with Image.open(partial) as img:img.verify()
            partial.replace(path)
            record={'status':'downloaded','image_id':image['image_id'],'url':url,'file':str(path.relative_to(WORKSPACE)),
                    'bytes':len(data),'sha256':file_hash(path),'verified_at':timestamp(),'download_seconds':time.perf_counter()-started,
                    'note':'Local SHA and PIL validation; no official per-image checksum published.'}
            write_json(meta,record);return path,{**record,'cache_hit':False}
        except Exception as error:errors.append(type(error).__name__+': '+str(error))
    record={'status':'download_failed','image_id':image['image_id'],'url':url,'errors':errors,'seconds':time.perf_counter()-started}
    write_json(directory/(asset_key(row)+'.failed.json'),record)
    return None,record


def ensure_card(row,allow_download=True,*,photo_variant=False):
    directory=ROOT/('photo-cards' if photo_variant else 'cards')/asset_key(row)
    directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'document.json';pdf=directory/'product.pdf'
    if manifest.exists():
        doc=json.loads(manifest.read_text('utf-8'))
        if doc['raw_sha256']!=row['raw_sha256'] or doc['version']!=PDF_VERSION or file_hash(pdf)!=doc['sha256']:raise ValueError('Stale card cache')
        return doc
    image,asset=photo(row,allow_download)
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image as RLImage
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    with _font_lock:
        if 'ABOFull' not in pdfmetrics.getRegisteredFontNames():
            font = os.environ.get('ABO_FONT_PATH') or ('C:/Windows/Fonts/simhei.ttf' if os.name=='nt' else
                    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
            pdfmetrics.registerFont(TTFont('ABOFull',font))
    body=ParagraphStyle('body',fontName='ABOFull',fontSize=11,leading=16,spaceAfter=8,wordWrap='CJK')
    heading=ParagraphStyle('head',parent=body,fontSize=18,leading=25,textColor=HexColor('#15344c'),spaceAfter=14)
    small=ParagraphStyle('small',parent=body,fontSize=9,leading=13,textColor=HexColor('#555555'))
    item=json.loads(row['raw_json'])
    story=[Paragraph('ABO 商品资料 / PRODUCT EVIDENCE',heading),Paragraph(html.escape(row['title']),body),
      Paragraph(html.escape('SKU: '+row['sku']+' | '+row['domain']+' | '+row['product_type']),small),
      Paragraph('官方历史元数据的派生资料卡。不是厂家说明书，不代表实时价格、库存或现行售后政策。字段缺失保持缺失。',small),Spacer(1,10)]
    for label,value in evidence_fields(item):story.append(Paragraph(html.escape(label+': '+value),body))
    story+=[Spacer(1,8),Paragraph(html.escape('Source: '+row['source_member']+' / line '+str(row['source_line'])),small)]
    if image:
        from PIL import Image
        with Image.open(image) as img:
            copy=img.convert('RGB');copy.thumbnail((1536,1536));preview=directory/'photo-preview.jpg';copy.save(preview,quality=88)
            width,height=copy.size
        scale=min(500/width,560/height)
        story += [PageBreak(),Paragraph('原始商品图片 / SOURCE PHOTO',heading),Paragraph(html.escape(row['sku']+' | '+row['domain']),body),
                  RLImage(str(preview),width=width*scale,height=height*scale),Spacer(1,12),
                  Paragraph('图片来自ABO公开图像，预览仅缩放。不能从外观推断未报告的材质、承重或功能。',small)]
    else:
        story.append(Paragraph('原图当前不可用：'+html.escape(asset['status'])+'。本资料仅提供元数据证据，不冒称已读取商品照片。',small))
    bullets=display_values(item,'bullet_point')
    if bullets:
        story += [PageBreak(),Paragraph('原始产品要点 / SOURCE NOTES',heading)]
        for i,text in enumerate(bullets,1):story.append(Paragraph(html.escape(str(i)+'. '+text),body))
    def footer(canvas,doc):
        canvas.setFont('ABOFull',8);canvas.setFillColor(HexColor('#555555'))
        canvas.drawString(38,24,'Amazon.com / ABO / CC BY 4.0 | '+row['sku']+' | PDF page '+str(doc.page))
    partial=directory/'product.building.pdf'
    SimpleDocTemplate(str(partial),pagesize=(595,842),leftMargin=38,rightMargin=38,topMargin=35,bottomMargin=45,
      title=row['title'],author='ABO source; derived layout by AaronYue77',pageCompression=1).build(story,onFirstPage=footer,onLaterPages=footer)
    partial.replace(pdf)
    import pymupdf
    rows=[];doc_id='abofull-'+asset_key(row)+('-photo' if photo_variant else '')
    with pymupdf.open(pdf) as source:
        for index,page in enumerate(source,1):
            rows.append({'page_id':doc_id+f':p{index:04d}','doc_id':doc_id,'page':index,'title':row['title'],
              'listing_key':row['listing_key'],'sku':row['sku'],'domain':row['domain'],'product_type':row['product_type'],
              'pdf':str(pdf.relative_to(WORKSPACE)),'pdf_sha256':file_hash(pdf),'text':page.get_text(sort=True),
              'text_source':'derived_PDF_native_text','photo_page':'SOURCE PHOTO' in page.get_text(),'rank':index,'score':0.0})
    write_jsonl(directory/'pages.jsonl',rows)
    doc={'version':PDF_VERSION,'listing_key':row['listing_key'],'raw_sha256':row['raw_sha256'],'pdf':str(pdf.relative_to(WORKSPACE)),
        'sha256':file_hash(pdf),'pages':rows,'image_asset':asset,'image_file':str(image.relative_to(WORKSPACE)) if image else None,
        'source_member':row['source_member'],'source_line':row['source_line']}
    write_json(manifest,doc)
    return doc


def ensure_photo_card(row,card):
    """Keep frozen metadata-only cards intact while adding live photo evidence."""
    if any(p['photo_page'] for p in card['pages']) or card['image_asset']['status']=='no_source_image':
        return card
    image,asset=photo(row)
    return ensure_card(row,photo_variant=True) if image else {**card,'image_asset':asset}


def choose_pages(doc,question,budget=2):
    from core import tokenize
    tokens=set(tokenize(question));pages=doc['pages'];photo_query=bool(__import__('re').search(r'图片|照片|外观|扶手|图案|photo|image|appearance|armrest',question,__import__('re').I))
    if photo_query and any(p['photo_page'] for p in pages):
        ordering=sorted(pages,key=lambda p:(not p['photo_page'],p['page']))
    else:ordering=sorted(pages,key=lambda p:(-len(tokens&set(tokenize(p['text']))),p['page']))
    return [{**p,'version':doc['version'],'store_id':p['domain'],'rank':i,'score':float(len(tokens&set(tokenize(p['text']))))} for i,p in enumerate(ordering[:budget],1)]
