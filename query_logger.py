"""
Query Logger — Azure Table Storage
Logs every query request with IP, question, method, cypher, results, and duration.
"""

import uuid
import logging
from datetime import datetime, timezone

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


def log_query(
    ip_address: str,
    question: str,
    method: str = "",
    cypher: str = "",
    result_count: int = 0,
    duration: float = 0.0,
    status: str = "success",
    error: str = "",
):
    """Log a query to Azure Table Storage. Fails silently to not break the API."""
    try:
        now = datetime.now(timezone.utc)
        entity = {
            "PartitionKey": now.strftime("%Y-%m-%d"),
            "RowKey": str(uuid.uuid4()),
            "ip_address": ip_address,
            "question": question,
            "method": method,
            "cypher": cypher or "",
            "result_count": result_count,
            "duration": duration,
            "status": status,
            "error": error,
            "timestamp_utc": now.isoformat(),
        }
        _get_table_client().create_entity(entity)
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")
