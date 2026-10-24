#!/usr/bin/env python3
"""Bounded official-API reads for eight existing review cases; never broad crawl."""
import datetime,hashlib,json,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review_revision_work/evidence';OUT.mkdir(parents=True,exist_ok=True)
BASE='https://en.wikipedia.org/w/api.php'
HEADERS={'User-Agent':'SafePlusResearchAudit/1.1 (https://github.com/Panpz202006/SafePlus; public evidence verification)'}
receipts=[]
def fetch(name,params):
    target=OUT/(name+'.json');url=BASE+'?'+urllib.parse.urlencode({'action':'query','format':'json','formatversion':2,'maxlag':5,**params})
    if target.exists():
        obj=json.loads(target.read_text());receipts.append({'name':name,'url':url,'status':'CACHED','sha256':hashlib.sha256(target.read_bytes()).hexdigest()});return obj
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for attempt in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=25) as r:body=r.read();status=r.status
            obj=json.loads(body)
            if 'error' in obj:raise ValueError(json.dumps(obj['error']))
            target.write_bytes(body);receipts.append({'name':name,'url':url,'status':'OK','http_status':status,'retrieved_utc':started,'attempts':attempt+1,'sha256':hashlib.sha256(body).hexdigest(),'has_continue':bool(obj.get('continue'))})
            print(name,'OK',len(body),flush=True);time.sleep(.6);return obj
        except Exception as e:
            if isinstance(e,urllib.error.HTTPError) and e.code==429:
                delay=float(e.headers.get('Retry-After','3'))
                if delay>30:break
            else:delay=1.5
            if attempt==0:time.sleep(delay)
            else:break
    receipts.append({'name':name,'url':url,'status':'FAILED','retrieved_utc':started,'error':str(e) if 'e' in locals() else 'unavailable'})
    print(name,'FAILED',flush=True);return None

tasks=[json.loads(x) for x in (ROOT/'HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.jsonl').read_text().splitlines()]
for t in tasks[:8]:
    user=t['username'];start=t['observation_start_utc'];end=t['observation_end_utc']
    fetch('history_'+t['entity_id'],{'list':'usercontribs|abuselog|logevents','ucuser':user,'ucdir':'newer','ucstart':start,'ucend':end,'uclimit':500,'ucprop':'ids|title|timestamp|comment|tags','afluser':user,'afldir':'newer','aflstart':start,'aflend':end,'afllimit':500,'aflprop':'ids|user|title|action|result|timestamp|hidden|revid|filter','letype':'block','letitle':'User:'+user,'ledir':'newer','lestart':start,'leend':end,'lelimit':500,'leprop':'ids|title|type|user|userid|timestamp|comment|details'})
fetch('clerk_revisiondelete_log',{'list':'logevents','letype':'delete','letitle':'Wikipedia:Requests for page protection/Increase','ledir':'newer','lestart':'2026-08-01T05:30:00Z','leend':'2026-08-02T05:30:00Z','lelimit':500,'leprop':'ids|title|type|user|userid|timestamp|comment|details'})
fetch('chimp_move_logs',{'list':'logevents','letype':'move','leuser':'Chimpanzeedeerkidney','ledir':'newer','lestart':'2026-09-02T00:00:00Z','leend':'2026-09-07T22:39:58Z','lelimit':500,'leprop':'ids|title|type|user|userid|timestamp|comment|details'})
fetch('visable_abuse_details',{'list':'abuselog','afllogid':44727043,'aflprop':'ids|user|title|action|result|timestamp|hidden|revid|filter|details'})
(OUT/'fetch_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
