from __future__ import annotations
import argparse,json,sys
from .api import create_app
from .ingest import ingest_manifest,reindex_all
from .search import search
from .store import Store

def main(argv=None):
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser(description="Sofia Library end-to-end system");p.add_argument("--db",default="data/sofia.db");p.add_argument("--config",default="config.yaml")
    sub=p.add_subparsers(dest="cmd",required=True);sub.add_parser("init");sub.add_parser("reindex")
    ing=sub.add_parser("ingest");ing.add_argument("manifest")
    find=sub.add_parser("search");find.add_argument("query")
    serve=sub.add_parser("serve");serve.add_argument("--host",default="127.0.0.1");serve.add_argument("--port",type=int,default=8000)
    args=p.parse_args(argv);store=Store(args.db)
    if args.cmd=="init":store.init();print(store.path.resolve())
    elif args.cmd=="ingest":print(json.dumps(ingest_manifest(store,args.manifest),ensure_ascii=False,indent=2))
    elif args.cmd=="search":store.init();print(json.dumps(search(store,args.query),ensure_ascii=False,indent=2))
    elif args.cmd=="reindex":store.init();print(json.dumps(reindex_all(store),ensure_ascii=False,indent=2))
    else:
        from .agent_api import create_agent_app
        import uvicorn;uvicorn.run(create_agent_app(),host=args.host,port=args.port)

if __name__=="__main__":main()
