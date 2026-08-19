"""
Query Logger — Azure Table Storage
Logs every query request with IP, location, question, method, cypher, results, and duration.
"""

import uuid
import logging
from datetime import datetime, timezone

import requests
from azure.data.tables import TableServiceClient
from app_config import get_required_setting

logger = logging.getLogger(__name__)

TABLE_NAME = "querylogs"

_table_client = None


def _get_table_client():
    """Lazily initialize and return the table client."""
    global _table_client
    if _table_client is None:
        connection_string = get_required_setting("AZURE_STORAGE_CONNECTION_STRING")
        service = TableServiceClient.from_connection_string(connection_string)
        service.create_table_if_not_exists(TABLE_NAME)
        _table_client = service.get_table_client(TABLE_NAME)
    return _table_client


def _lookup_location(ip_address: str) -> dict:
    """Resolve IP to location using ip-api.com (free, no key needed)."""
    try:
        resp = requests.get(
            f"http://ip-api.com/json/{ip_address}?fields=country,regionName,city,lat,lon",
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "country": data.get("country", ""),
                "region": data.get("regionName", ""),
                "city": data.get("city", ""),
                "latitude": data.get("lat", 0.0),
                "longitude": data.get("lon", 0.0),
            }
    except Exception as e:
        logger.debug(f"Geo-IP lookup failed for {ip_address}: {e}")
    return {"country": "", "region": "", "city": "", "latitude": 0.0, "longitude": 0.0}


def log_query(
    ip_address: str,
    question: str,
    method: str = "",
    cypher: str = "",
    cypher_result: str = "",
    answer: str = "",
    result_count: int = 0,
    duration: float = 0.0,
    status: str = "success",
    error: str = "",
):
    """Log a query to Azure Table Storage. Fails silently to not break the API."""
    try:
        now = datetime.now(timezone.utc)
        location = _lookup_location(ip_address)
        entity = {
            "PartitionKey": now.strftime("%Y-%m-%d"),
            "RowKey": str(uuid.uuid4()),
            "ip_address": ip_address,
            "country": location["country"],
            "region": location["region"],
            "city": location["city"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "question": question,
            "method": method,
            "cypher": cypher or "",
            "cypher_result": cypher_result or "",
            "answer": answer or "",
            "result_count": result_count,
            "duration": duration,
            "status": status,
            "error": error,
            "timestamp_utc": now.isoformat(),
        }
        _get_table_client().create_entity(entity)
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")
