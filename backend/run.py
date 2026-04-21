from pathlib import Path
import os

import uvicorn

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(BASE_DIR)],
        app_dir=str(BASE_DIR),
    )
