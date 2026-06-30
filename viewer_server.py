#!/usr/bin/env python3
"""
Simple server to serve the viewer HTML and proxy simulator requests.
Solves CORS issues by acting as a proxy between browser and simulator.
"""

import httpx
import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Simulator Viewer")

BASE_URL = "https://ci.mines-stetienne.fr/simu"
GROUP_NUM = 10  # Default, can be overridden via CLI arg


@app.get("/")
async def root():
    """Serve the viewer HTML."""
    return FileResponse("viewer.html")


@app.get("/api/properties/{workstation}")
async def get_properties(workstation: str):
    """
    Proxy request to simulator's all/properties endpoint.

    Handles any workstation name (e.g., storageRack, fillingWorkshop, robotArm).
    """
    url = f"{BASE_URL}/{workstation}/all/properties"
    auth = (f"simu{GROUP_NUM}", f"simu{GROUP_NUM}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, auth=auth, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"DEBUG {workstation}: {data}")
            return data
    except Exception as e:
        print(f"ERROR {workstation}: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Simulator viewer server")
    parser.add_argument("--group", type=int, default=10, choices=range(1, 11),
                        help="Group number (1-10, default 10)")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port to run server on (default 8001)")
    args = parser.parse_args()

    GROUP_NUM = args.group
    print(f"Starting viewer for group {GROUP_NUM} on port {args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
