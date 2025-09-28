import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from groq import Groq
from dotenv import load_dotenv
from email.mime.text import MIMEText
import uuid
import os
import datetime
import smtplib
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult


load_dotenv()

inngest_client = inngest.Inngest(
    app_id='rag_app',
    logger=logging.getLogger('uvicorn'),
    is_production=False,
    serializer=inngest.PydanticSerializer()
) 

# uploaded documents are not directly sent to API endpoint. Instead, these requests are sent to Inngest server. This forwards the request (with PDF) to API.  

@inngest_client.create_function(
    fn_id='RAG: Ingest PDF',
    trigger=inngest.TriggerEvent(event='rag/ingest_pdf')
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data['pdf_path']
        source_id = ctx.event.data.get('source_id', pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:     # uploads and inserts chunks/source_ids into qdrantDb
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}: {i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    # run these steps chronological by adding 'await'. By removing 'await': steps can be run parallel
    chunks_and_src = await ctx.step.run('load-and-chunk', lambda: _load(ctx), output_type=RAGChunkAndSrc)      # wrap into an inngest 'step'           
    ingested = await ctx.step.run('embed-and-upsert', lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()    # converts model into py dict / json

###################################################################################################
@inngest_client.create_function(
    fn_id='RAG: Mail Alerting',
    trigger=inngest.TriggerEvent(event='rag/alert_new_doc')
)
async def rag_alert_new_doc(ctx: inngest.Context):
    source_id = ctx.event.data['source_id']
    # summarize document
    chunks = ctx.event.data["chunks"][:5]  
    context_block = "\n\n".join(chunks)

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "doc-summary",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You summarize new documents for alerting emails."},
                {"role": "user", "content": f"Summarize this document:\n{context_block}"}
            ]
        }
    )
    summary = res["choices"][0]["message"]["content"].strip()

    # send gmail notification
    sender = os.getenv("GMAIL_SENDER")
    recipient = os.getenv("GMAIL_RECEIVER")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    msg = MIMEText(f"New document ingested: {source_id}\n\nSummary:\n{summary}")
    msg["Subject"] = f"👮‍♂️🚨RAG-AGENT: new document alert: {source_id}"
    msg["From"] = sender
    msg["To"] = recipient

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, [recipient], msg.as_string())

    return {"alert_sent": True, "summary": summary}

###################################################################################################
@inngest_client.create_function(
    fn_id='RAG: Query PDF',
    trigger=inngest.TriggerEvent(event='rag/query_pdf_ai')
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found['contexts'], sources=found['sources'])
    
    question = ctx.event.data['question']
    top_k = int(ctx.event.data.get('top_k', 5))

    found = await ctx.step.run('embed-and-search', lambda: _search(question, top_k), output_type=RAGSearchResult) 

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question. \n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model='gpt-4o-mini'
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            'max_tokens': 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}




app = FastAPI()

# /api/inngest endpoint
inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_alert_new_doc, rag_query_pdf_ai])