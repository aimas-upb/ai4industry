"""
BehaviorTree plan caching for goal schema and instance matching.

Stores successful BT plans indexed by (schema, instance) tuple.
Allows skipping discovery/planning for repeated goals.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ANSI color codes
BLUE = "\033[94m"
RESET = "\033[0m"


class PlanCache:
    """
    Stores and retrieves BehaviorTree plans by goal schema and instance.

    Caching strategy:
    - Key: (goal_schema, goal_instance) tuple
    - Value: BT JSON-IR dict
    - Storage: JSON file at configured path
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize the plan cache.

        Args:
            cache_path: Path to cache JSON file. Plans are always cached here if successful.
        """
        self.cache_path = cache_path
        self._cache = {}

        if cache_path:
            self._load_cache()
            logger.debug(f"{BLUE}[CACHE]{RESET} Initialized at {self.cache_path}")

    def _load_cache(self) -> None:
        """Load cache from disk if it exists."""
        if not self.cache_path.exists():
            logger.debug(f"{BLUE}[CACHE]{RESET} No existing cache file, starting fresh")
            self._cache = {}
            return

        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)
            self._cache = data
            logger.debug(f"{BLUE}[CACHE]{RESET} Loaded {len(self._cache)} cached plans")
        except Exception as e:
            logger.warning(f"{BLUE}[CACHE]{RESET} Failed to load cache: {e}, starting fresh")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.cache_path:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"{BLUE}[CACHE]{RESET} Saved cache to disk")
        except Exception as e:
            logger.warning(f"{BLUE}[CACHE]{RESET} Failed to save cache: {e}")

    def get(self, schema: str, instance: str) -> Optional[dict]:
        """
        Retrieve a cached plan for the given schema and instance.

        Args:
            schema: Goal predicate schema (e.g., "!carry(Param1, Param2, Param3)")
            instance: Goal predicate instance (e.g., '!carry("APAS", "DX10_output", "XY10_input")')

        Returns:
            BT JSON-IR dict if found, None otherwise
        """
        if not self.cache_path:
            return None

        key = f"{schema}|{instance}"
        if key in self._cache:
            logger.debug(f"{BLUE}[CACHE]{RESET} Hit: {schema} + {instance}")
            return self._cache[key]

        logger.debug(f"{BLUE}[CACHE]{RESET} Miss: {schema} + {instance}")
        return None

    def put(self, schema: str, instance: str, plan: dict) -> None:
        """
        Store a plan in the cache if it doesn't already exist.

        Args:
            schema: Goal predicate schema
            instance: Goal predicate instance
            plan: BT JSON-IR dict to cache
        """
        if not self.cache_path:
            return

        key = f"{schema}|{instance}"
        if key in self._cache:
            logger.debug(f"{BLUE}[CACHE]{RESET} Plan already cached for {schema} + {instance}, skipping")
            return

        self._cache[key] = plan
        self._save_cache()
        logger.debug(f"{BLUE}[CACHE]{RESET} Stored plan for {schema} + {instance}")

    def clear(self) -> None:
        """Clear all cached plans."""
        self._cache = {}
        self._save_cache()
        logger.debug(f"{BLUE}[CACHE]{RESET} Cache cleared")
