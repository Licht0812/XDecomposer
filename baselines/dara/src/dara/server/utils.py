import zoneinfo
from datetime import datetime

from jobflow import JobStore
from maggma.core import Store
from maggma.stores import MongoStore, MontyStore

from dara.server.setting import get_dara_server_settings


def convert_to_local_tz(dt):
    """Converts a UTC datetime object to the system's local timezone."""
    local_tz = datetime.now().astimezone().tzinfo  # gets system local tz
    return dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(local_tz)


def get_store(collection_name):
    """Get a MontyStore instance for job management."""
    setting = get_dara_server_settings()
    if setting.database_backend == "monty":
        store = MontyStore(
            database_name="dara",
            collection_name=collection_name,
            database_path=setting.montydb_path,
            storage="sqlite",
            key=["uuid", "index"],
        )
    elif setting.database_backend == "mongodb":
        store = MongoStore(
            database=setting.mongodb_database,
            collection_name=collection_name,
            host=setting.mongodb_host,
            port=setting.mongodb_port,
            username=setting.mongodb_username,
            password=setting.mongodb_password,
        )
    else:
        raise ValueError(f"Unsupported database backend: {setting.database_backend}")

    return store


def get_worker_store() -> MontyStore:
    """Get a MontyStore instance for job management."""
    return get_store("jobs")


def get_result_store() -> MontyStore:
    """Get a MontyStore instance for job management."""
    return get_store("results")


def get_job_store(docs_store: Store) -> JobStore:
    return JobStore(docs_store=docs_store)
