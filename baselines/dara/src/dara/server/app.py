import multiprocessing
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from dara.server.api_router import router
from dara.server.setting import get_dara_server_settings
from dara.server.worker import worker_process

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

_worker_process = None


@asynccontextmanager
async def launch_worker_process(app: FastAPI):
    """Context manager to launch the worker process."""
    global _worker_process  # noqa: PLW0603
    _worker_process = multiprocessing.Process(target=worker_process, daemon=True)
    _worker_process.start()
    try:
        yield
    finally:
        if _worker_process.is_alive():
            print("Terminating worker process...")
            _worker_process.terminate()


app = FastAPI(lifespan=launch_worker_process)

# Mount the frontend directory to the root path
frontend_dir = Path(__file__).parent / "ui" / "public"
app.include_router(router)

app.mount(
    "", StaticFiles(directory=frontend_dir.as_posix(), html=True), name="frontend"
)


def launch_app():
    """Main function to run the FastAPI application."""
    import uvicorn

    setting = get_dara_server_settings()
    uvicorn.run(
        app,
        host=setting.host,
        port=setting.port,
        workers=1,
        timeout_graceful_shutdown=1,
    )
