#!/usr/bin/env python3
import os
import argparse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--with-plan-caching", action="store_true", help="Enable plan caching (path from PLAN_CACHE_PATH env var)")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8008)
args = parser.parse_args()

# Pass the flag to the app via environment
os.environ["PLAN_CACHE_ENABLED"] = "true" if args.with_plan_caching else "false"

uvicorn.run("src.main:app", host=args.host, port=args.port)
