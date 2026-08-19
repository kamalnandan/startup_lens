"""
Global Startup Intelligence Graph — Query API
FastAPI app wrapping Neo4j query layer.
Deploys on Azure Web App (Python).
"""

import os
import time
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j_query import query as neo4j_query
from query_logger import log_query

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Global Startup Intelligence Graph API",
    description="Neo4j-powered query API over 6000+ YC startup profiles",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    method: str
    cypher: Optional[str] = None
    result_count: int
    duration_seconds: float

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Global Startup Intelligence Graph API",
        "version": "2.0.0"
    }

@app.get("/status")
def index_status():
    """Return basic graph stats from Neo4j."""
    from neo4j import GraphDatabase
    from neo4j_query import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS type, count(n) AS count
                ORDER BY count DESC
            """)
            stats = {record["type"]: record["count"] for record in result}
        driver.close()
        return {"status": "ok", "graph_stats": stats}
    except Exception as e:
        return {"status": "ok", "graph_stats": {}, "note": str(e)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, raw_request: Request):
    """
    Main query endpoint.
    Automatically classifies query, generates Cypher,
    executes against Neo4j, and returns synthesized answer.
    """
    if not request.query or len(request.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Query too short")

    # Prefer X-Real-IP (forwarded by Streamlit) over x-forwarded-for (Azure infra)
    ip_address = raw_request.headers.get("x-real-ip") or raw_request.headers.get("x-forwarded-for", raw_request.client.host or "unknown")
    # Strip port number if present (e.g. "9.223.52.251:8386" → "9.223.52.251")
    ip_address = ip_address.split(",")[0].strip().split(":")[0]
    start_time = time.time()

    try:
        result = neo4j_query(request.query)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        duration = round(time.time() - start_time, 2)
        log_query(
            ip_address=ip_address,
            question=request.query,
            duration=duration,
            status="error",
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))

    duration = round(time.time() - start_time, 2)

    log_query(
        ip_address=ip_address,
        question=request.query,
        method=result.get("method", ""),
        cypher=result.get("cypher", ""),
        cypher_result=str(result.get("cypher_result", "")),
        answer=result.get("answer", ""),
        result_count=result.get("result_count", 0),
        duration=duration,
        status="success",
    )

    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        method=result["method"],
        cypher=result.get("cypher"),
        result_count=result.get("result_count", 0),
        duration_seconds=duration
    )