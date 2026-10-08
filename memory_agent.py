#!/usr/bin/env python3
"""Workshop vector-memory RAG demo: chunk -> embed -> linear-list memory -> cosine top-k -> grounded answer.
Setup: pip install openai; set OPENROUTER_API_KEY; run python memory_agent.py
First run indexes kb.txt into memory.json. Use --reindex to rebuild. Never put API keys in code.
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit('Install dependency: pip install openai')

BASE_URL='https://openrouter.ai/api/v1'
EMBED_MODEL=os.getenv('OPENROUTER_EMBEDDING_MODEL','nvidia/nemotron-3-embed-1b:free')
CHAT_MODEL=os.getenv('OPENROUTER_CHAT_MODEL','nvidia/nemotron-3.5-lightning:free')
HERE=Path(__file__).resolve().parent
KB_PATH=HERE/'kb.txt'; MEMORY_PATH=HERE/'memory.json'
CHUNK_SIZE=500; CHUNK_OVERLAP=60; TOP_K=3

def make_client():
    key=os.getenv('OPENROUTER_API_KEY')
    if not key: raise RuntimeError('Set OPENROUTER_API_KEY in your environment first.')
    return OpenAI(base_url=BASE_URL,api_key=key)

def chunk_text(text,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
    """Simple overlapping character chunking so the workshop logic stays visible."""
    text=text.strip()
    if not text:return []
    if size<=0 or overlap<0 or overlap>=size:raise ValueError('Require size > 0 and 0 <= overlap < size')
    chunks=[]; start=0
    while start<len(text):
        end=min(start+size,len(text)); chunk=text[start:end].strip()
        if chunk:chunks.append(chunk)
        if end==len(text):break
        start=end-overlap
    return chunks

def embed_texts(client,texts):
    if not texts:return []
    result=client.embeddings.create(model=EMBED_MODEL,input=texts)
    return [row.embedding for row in sorted(result.data,key=lambda x:x.index)]

def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_or_build_memory(client,force=False):
    """Memory is intentionally a plain Python list, persisted as JSON (no vector DB)."""
    if not KB_PATH.exists():raise FileNotFoundError(f'Missing {KB_PATH}; put kb.txt beside this script.')
    digest=sha256(KB_PATH)
    if MEMORY_PATH.exists() and not force:
        try:
            saved=json.loads(MEMORY_PATH.read_text(encoding='utf-8'))
            if saved.get('kb_sha256')==digest and saved.get('embedding_model')==EMBED_MODEL and saved.get('memory'):
                print(f"Loaded {len(saved['memory'])} chunks from memory.json (index current).")
                return saved['memory']
        except (OSError,json.JSONDecodeError):print('Could not read cached index; rebuilding.')
    chunks=chunk_text(KB_PATH.read_text(encoding='utf-8'))
    print(f'Chunked kb.txt into {len(chunks)} chunks ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap).')
    print(f'Embedding chunks with {EMBED_MODEL} ...')
    t=time.perf_counter(); vectors=embed_texts(client,chunks)
    memory=[{'id':f'chunk-{i+1:03d}','text':text,'source':KB_PATH.name,'embedding':vec} for i,(text,vec) in enumerate(zip(chunks,vectors))]
    MEMORY_PATH.write_text(json.dumps({'kb_sha256':digest,'embedding_model':EMBED_MODEL,'memory':memory},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Indexed {len(memory)} chunks in {time.perf_counter()-t:.1f}s; saved memory.json.')
    return memory

def cosine_similarity(a,b):
    if len(a)!=len(b):raise ValueError('Embedding dimensions do not match.')
    dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def retrieve(client,memory,query,k=TOP_K):
    """Embed query, score every item in the list, sort by cosine similarity, return top-k."""
    qvec=embed_texts(client,[query])[0]
    ranked=[{**item,'score':cosine_similarity(qvec,item['embedding'])} for item in memory]
    return sorted(ranked,key=lambda item:item['score'],reverse=True)[:k]

def answer_from_context(client,query,hits):
    context='\n\n'.join(f"[{h['id']} | {h['source']} | similarity {h['score']:.3f}]\n{h['text']}" for h in hits)
    system=('You are the ShopSmart store assistant. Use ONLY the supplied CONTEXT. Do not invent product, policy, price, stock, or order facts. '
            "If unsupported, reply exactly in substance: I couldn't find that in the store knowledge base. Cite each factual claim with its chunk ID, e.g. [chunk-001]. Be concise.")
    result=client.chat.completions.create(model=CHAT_MODEL,temperature=0.2,messages=[
        {'role':'system','content':system},
        {'role':'user','content':f'CONTEXT:\n{context}\n\nQUESTION:\n{query}'}])
    return result.choices[0].message.content or '(Empty model response)'

def ask(client,memory,query,k):
    t=time.perf_counter(); hits=retrieve(client,memory,query,k); retrieval_s=time.perf_counter()-t
    print('\n--- Retrieved context (inspect retrieval before judging answer) ---')
    for h in hits:
        preview=' '.join(h['text'].split())
        print(f"{h['id']} | cosine={h['score']:.3f} | {h['source']}\n  {preview[:240]}{'...' if len(preview)>240 else ''}")
    t=time.perf_counter(); response=answer_from_context(client,query,hits); answer_s=time.perf_counter()-t
    print('\n--- Grounded answer ---\n'+response)
    print(f'\n[trace] retrieval={retrieval_s:.2f}s | generation={answer_s:.2f}s | top_k={k}')

def main():
    parser=argparse.ArgumentParser(description='ShopSmart workshop vector-memory agent')
    parser.add_argument('--reindex',action='store_true',help='Force re-embedding kb.txt')
    parser.add_argument('--top-k',type=int,default=TOP_K,help='Retrieved chunks (default 3)')
    args=parser.parse_args()
    if args.top_k<1:parser.error('--top-k must be >= 1')
    try:client=make_client(); memory=load_or_build_memory(client,args.reindex)
    except Exception as exc:raise SystemExit(f'Setup/indexing error: {exc}')
    print("\nShopSmart Memory Agent. Ask about store policies, products, or support.")
    print("Try an unsupported question to test grounding. Type 'exit' to quit.\n")
    while True:
        try:query=input('You: ').strip()
        except (EOFError,KeyboardInterrupt):print('\nBye!');break
        if query.lower() in {'exit','quit','q'}:print('Bye!');break
        if not query:continue
        try:ask(client,memory,query,args.top_k)
        except Exception as exc:print(f'Request failed: {exc}\nCheck model availability, API key, rate limits, and connection.',file=sys.stderr)
if __name__=='__main__':main()
