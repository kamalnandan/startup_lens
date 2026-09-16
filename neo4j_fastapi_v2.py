"""
StartupLens Assertion API — query service over the assertion graph.

A separate app from the production API rather than another route inside it.
The two read different graphs on different ports and disagree about what an
answer is, and keeping them in one process would mean one bad deploy, one
misread setting, or one restart could take production down with it. Separate
apps make the isolation structural: the fallback during a demo is a different
URL, not a flag.
"""

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from pydantic import BaseModel

import ask_v2
import graph_model_v2
from app_config import get_required_setting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = get_required_setting("STARTUPLENS_API_KEY")


def verify_api_key(request: Request):
    """Check X-API-Key header. Raises 401 if missing or invalid."""
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(
    title="StartupLens Assertion API",
    description="Answers grounded in extracted assertions, each carrying the "
                "sentence it came from.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One driver for the process. Opening a connection per request would spend
# more time on the handshake than on the query.
_settings = graph_model_v2.connection_settings()
_driver = None


def driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            _settings["uri"],
            auth=(_settings["username"], _settings["password"]))
    return _driver


class QueryRequest(BaseModel):
    query: str
    split: bool = True


class QueryPart(BaseModel):
    question: str
    status: str
    cypher: Optional[str] = None
    rows: list = []


class QueryResponse(BaseModel):
    query: str
    status: str
    answer: Optional[str] = None
    parts: list = []
    duration_seconds: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "StartupLens Assertion API",
            "version": "1.0.0"}


@app.get("/status")
def graph_status():
    """Report what the assertion graph holds, and which graph it is.

    The uri is returned deliberately. The failure this endpoint exists to
    catch is the service quietly reading the wrong graph, which looks exactly
    like a working service until an answer is wrong.
    """
    labels = ["V2Company", "V2Fact", "V2Person", "V2Organisation",
              "V2Place", "V2Category"]
    try:
        with driver().session(database=_settings["database"]) as session:
            counts = {
                label: session.run(
                    f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                for label in labels
            }
        return {"status": "ok", "uri": _settings["uri"], "counts": counts}
    except Exception as failure:      # noqa: BLE001 - reported, not raised
        # Returned rather than raised: an unreachable graph is the single most
        # likely deployment failure, and it is worth reading plainly.
        return {"status": "unreachable", "uri": _settings["uri"],
                "error": str(failure)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, raw_request: Request):
    """Answer a question from asserted facts, or report that it cannot."""
    verify_api_key(raw_request)

    if not request.query or len(request.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Query too short")

    started = time.time()
    try:
        result = ask_v2.ask(request.query, driver(), _settings["database"],
                            split=request.split)
    except Exception as failure:      # noqa: BLE001 - surfaced as a 500
        logger.error("Query failed: %s", failure)
        raise HTTPException(status_code=500, detail=str(failure))

    # Attempt logs are dropped from the response: they carry the Cypher of
    # every failed try, which is useful in a file and noise over the wire.
    parts = [{"question": part["question"], "status": part["status"],
              "cypher": part.get("cypher"), "rows": part.get("rows", [])}
             for part in result.get("parts", [])]

    return QueryResponse(
        query=request.query,
        status=result["status"],
        answer=result.get("answer"),
        parts=parts,
        duration_seconds=round(time.time() - started, 2),
    )
