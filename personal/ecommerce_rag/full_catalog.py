"""All ABO listings: lossless source records, scalable FTS5 BM25, image map."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/'rag_common'))
from common import WORKSPACE,file_hash,write_json,operation,timestamp

ROOT=WORKSPACE/'data/ecommerce_rag/abo-full-v1'
SOURCE=Path(os.environ.get('ABO_SOURCE_DIR', WORKSPACE/'data/ecommerce_rag/abo-source'))
DB=ROOT/'catalog.sqlite'
EXPECTED=147702
SCHEMA='abo-full-v1'


def source_records():
    archive_path=SOURCE/'archives/abo-listings.tar'
    if not archive_path.is_file():
        expanded=SOURCE/'archives/abo-listings/listings/metadata'
        paths=[expanded/f'listings_{digit}.json' for digit in '0123456789abcdef']
        for path in paths:
            member=f'listings/metadata/{path.name}.gz'
            with path.open('rb') as stream:
                for number,line in enumerate(stream,1):
                    if line.strip():yield json.loads(line),member,number
        return
    with tarfile.open(archive_path,'r:') as archive:
        for member in sorted(archive.getmembers(),key=lambda m:m.name):
            if not member.name.endswith('.json.gz'):continue
            with archive.extractfile(member) as file:
                with gzip.GzipFile(fileobj=file) as stream:
                    for number,line in enumerate(stream,1):
                        if line.strip():yield json.loads(line),member.name,number


def values(item,key,languages=None):
    value=item.get(key,[])
    if not isinstance(value,list):return []
    rows=[r for r in value if isinstance(r,dict) and 'value' in r]
    if languages:
        preferred=[r for r in rows if r.get('language_tag') in languages]
        if preferred:rows=preferred
    return list(dict.fromkeys(str(r['value']) for r in rows if str(r['value']).strip()))


def field(item,key):
    return '; '.join(values(item,key,{'en_US','zh_CN'}))


def title(item):
    rows=item.get('item_name',[])
    for language in ('en_US','zh_CN'):
        found=[r['value'] for r in rows if r.get('language_tag')==language and r.get('value')]
        if found:return found[0]
    return next(iter(values(item,'item_name')),item['item_id'])


def flatten(value):
    if isinstance(value,dict):
        return ' '.join(str(k)+' '+flatten(v) for k,v in value.items() if k not in {'language_tag','standardized_values'})
    if isinstance(value,list):return ' '.join(flatten(v) for v in value)
    return str(value) if value is not None else ''


def terms(value):
    # FTS5 unicode61 does not split an unspaced Chinese sentence into words.
    # Keep Unicode words and add Han bigrams, preserving the original source separately.
    result=[]
    for token in re.findall(r'[\w]+',str(value).casefold(),re.UNICODE):
        result.append(token)
        for span in re.findall(r'[\u4e00-\u9fff]+',token):
            if len(span)>1:result.extend(span[i:i+2] for i in range(len(span)-1))
    return ' '.join(result)


def dense_text(item):
    parts=['name: '+field(item,'item_name'),'type: '+field(item,'product_type'),
           'brand: '+field(item,'brand'),'model: '+field(item,'model_number'),
           'color: '+field(item,'color'),'material: '+field(item,'material')]
    for key in ('item_dimensions','item_weight'):
        if key in item:parts.append(key+': '+flatten(item[key]))
    parts+=values(item,'bullet_point',{'en_US','zh_CN'})
    return '\n'.join(parts)


def connect(readonly=True):
    if readonly:
        if not DB.exists():raise FileNotFoundError('Build the full ABO catalog first')
        db=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True,timeout=30,check_same_thread=False)
    else:db=sqlite3.connect(DB,timeout=30)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA cache_size=-65536')
    return db


def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    if DB.exists():
        print('Complete full catalog exists:',DB,flush=True);return
    partial=ROOT/'catalog.building.sqlite'
    if partial.exists():raise FileExistsError('Incomplete build preserved; inspect it before retry')
    started=time.perf_counter();db=sqlite3.connect(partial)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-98304')
    db.executescript('''
      CREATE TABLE products(id INTEGER PRIMARY KEY, listing_key TEXT NOT NULL UNIQUE,
       sku TEXT NOT NULL, domain TEXT NOT NULL, product_type TEXT NOT NULL, title TEXT NOT NULL,
       brand TEXT NOT NULL, main_image_id TEXT, source_member TEXT NOT NULL, source_line INTEGER NOT NULL,
       raw_sha256 TEXT NOT NULL, raw_json TEXT NOT NULL, dense_text TEXT NOT NULL, family_key TEXT);
      CREATE VIRTUAL TABLE products_fts USING fts5(sku,name,attributes,body,
       tokenize='unicode61 remove_diacritics 2');
      CREATE TABLE images(image_id TEXT PRIMARY KEY,path TEXT NOT NULL,width INTEGER,height INTEGER);
      CREATE INDEX idx_sku ON products(sku); CREATE INDEX idx_type ON products(product_type);
      CREATE INDEX idx_domain ON products(domain);
    ''')
    kinds=Counter();domains=Counter();missing=Counter();seen={};count=0
    # Union related records through shared image/model, so evaluation can split families.
    parent=[];image_owner={};model_owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for item,member,line in source_records():
        sku=item['item_id'];domain=item.get('domain_name','unknown');key=sku+'@'+domain
        if key in seen:raise ValueError('Duplicate official listing identity: '+key)
        seen[key]=count+1;count+=1;kind=field(item,'product_type') or 'UNKNOWN'
        raw=json.dumps(item,ensure_ascii=False,separators=(',',':'),sort_keys=True)
        digest=hashlib.sha256(raw.encode()).hexdigest();name=title(item);brand=field(item,'brand')
        image=item.get('main_image_id');model=field(item,'model_number')
        parent.append(count-1)
        for owners,identity in [(image_owner,image),(model_owner,(brand.casefold(),model.casefold()) if model else None)]:
            if identity:
                if identity in owners:parent[find(count-1)]=find(owners[identity])
                else:owners[identity]=count-1
        db.execute('INSERT INTO products VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)',
            (count,key,sku,domain,kind,name,brand,image,member,line,digest,raw,dense_text(item)))
        attrs={k:v for k,v in item.items() if k not in {'item_name','bullet_point','item_keywords','other_image_id','main_image_id','spin_id','3dmodel_id'}}
        db.execute('INSERT INTO products_fts(rowid,sku,name,attributes,body) VALUES(?,?,?,?,?)',
            (count,sku,terms(' '.join(values(item,'item_name'))),terms(flatten(attrs)),
             terms(' '.join(values(item,'bullet_point')+values(item,'item_keywords')))))
        kinds[kind]+=1;domains[domain]+=1
        for attr in ('main_image_id','brand','color','material','item_dimensions','item_weight','bullet_point'):
            if not item.get(attr):missing[attr]+=1
        if count%1000==0:db.commit()
        if count%10000==0:print('ABO imported',count,'elapsed',round(time.perf_counter()-started,1),flush=True)
    assert count==EXPECTED,(count,EXPECTED)
    families={}
    for i in range(count):
        root=find(i)
        if root not in families:families[root]='abo-family-'+hashlib.sha256(str(root+1).encode()).hexdigest()[:20]
    db.executemany('UPDATE products SET family_key=? WHERE id=?',[(families[find(i)],i+1) for i in range(count)])
    images=0
    compressed=SOURCE/'images/metadata/images.csv.gz'
    image_metadata=compressed if compressed.is_file() else compressed.with_suffix('')
    opener=gzip.open if compressed.is_file() else open
    with opener(image_metadata,'rt',encoding='utf-8',newline='') as stream:
        batch=[]
        for r in csv.DictReader(stream):
            batch.append((r['image_id'],r['path'],int(r['width']),int(r['height'])));images+=1
            if len(batch)==5000:db.executemany('INSERT INTO images VALUES(?,?,?,?)',batch);batch=[]
        if batch:db.executemany('INSERT INTO images VALUES(?,?,?,?)',batch)
    db.execute('CREATE INDEX idx_family ON products(family_key)')
    db.execute("INSERT INTO products_fts(products_fts) VALUES('optimize')")
    db.commit()
    assert db.execute('SELECT count(*) FROM products').fetchone()[0]==EXPECTED
    assert db.execute('SELECT count(*) FROM products_fts').fetchone()[0]==EXPECTED
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    missing_images=db.execute('SELECT count(*) FROM products p LEFT JOIN images i ON p.main_image_id=i.image_id WHERE i.image_id IS NULL').fetchone()[0]
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.execute('PRAGMA journal_mode=DELETE');db.close()
    partial.replace(DB)
    source_archive=SOURCE/'archives/abo-listings.tar'
    manifest={'version':SCHEMA,'created_at':timestamp(),'records':count,'fts_records':count,'product_types':dict(kinds.most_common()),
       'domains':dict(domains),'missing_fields':dict(missing),'image_metadata_records':images,'listings_without_resolved_main_image':missing_images,
       'related_families':len(families),'database_sha256':file_hash(DB),
       'source_archive_sha256':file_hash(source_archive) if source_archive.is_file() else None,
       'source_image_metadata_sha256':file_hash(image_metadata),'source_layout':'original_archive' if source_archive.is_file() else 'expanded_files',
       'build_seconds':time.perf_counter()-started,
       'identity':'item_id + domain_name; every official row retained, including incomplete records',
       'coverage':'All metadata indexed. Official images fetched by resolved path on demand, not all downloaded or visually embedded.',
       'dense_policy':'One representative multilingual text per listing, max512 tokens; full original JSON/FTS remains available beyond truncation.',
       'source':'Amazon Berkeley Objects; custom application corpus/QA, not official benchmark or manufacturer PDF.'}
    write_json(ROOT/'manifest.json',manifest)
    for name in ('README.md','LICENSE-CC-BY-4.0.txt'):
        (ROOT/('SOURCE-'+name)).write_bytes((SOURCE/name).read_bytes())
    operation('全量ABO元数据建库','流式读取官方16个gzip成员；原JSON、来源行号和SHA原样保留；SQLite FTS5 BM25索引所有行；共享图/型号关联家族；导入全量图片路径元数据。',
       f'{count}条，{len(kinds)}类，{images}图像元数据；不筛掉字段缺失或非椅子商品。',[DB,ROOT/'manifest.json'],
       '原表和FTS行数均147702、SQLite integrity_check=ok；这是全元数据覆盖，原图按需获取。')
    print(json.dumps({k:manifest[k] for k in ('records','fts_records','image_metadata_records','related_families','build_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__':build()
