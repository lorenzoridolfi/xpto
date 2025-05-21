"""
LLM Cache System

This module implements a caching system for LLM responses to reduce API calls and improve performance.
The cache uses a combination of input hashing and similarity matching to identify similar requests.
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger("LLMCache")

@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    response: Any
    timestamp: datetime
    hash: str
    metadata: Dict[str, Any]

class LLMCache:
    """
    A caching system for LLM responses that includes:
    - Exact match caching using hashing
    - Similarity-based caching using TF-IDF and cosine similarity
    - Cache entry expiration
    - Metadata tracking
    - LLM parameter validation
    """

    PARAMS_FILE = ".llm_cache_params.json"

    def __init__(
        self,
        max_size: int = 1000,
        expiration_hours: int = 24,
        llm_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the LLM cache.

        Args:
            max_size (int): Maximum number of cache entries
            expiration_hours (int): Hours before cache entries expire
            llm_params (Optional[Dict[str, Any]]): Initial LLM parameters
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.expiration_hours = expiration_hours
        self.llm_params = llm_params
        # ... (load params if needed)

    def _compute_hash(self, messages: List[Dict[str, Any]]) -> str:
        sorted_messages = sorted([json.dumps(m, sort_keys=True) for m in messages])
        return hashlib.sha256("".join(sorted_messages).encode()).hexdigest()

    def _clean_expired_entries(self) -> None:
        now = datetime.utcnow()
        expired = [
            h for h, entry in self.cache.items()
            if now - entry.timestamp > timedelta(hours=self.expiration_hours)
        ]
        for h in expired:
            del self.cache[h]

    def get(self, messages: List[Dict[str, Any]]) -> Optional[Any]:
        self._clean_expired_entries()
        h = self._compute_hash(messages)
        if h in self.cache:
            entry = self.cache[h]
            entry.timestamp = datetime.utcnow()
            logger.info(f"Cache hit for hash: {h}")
            return entry.response
        logger.info("Cache miss")
        return None

    def put(self, messages: List[Dict[str, Any]], response: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._clean_expired_entries()
        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = min(self.cache.values(), key=lambda e: e.timestamp)
            del self.cache[oldest.hash]
        h = self._compute_hash(messages)
        entry = CacheEntry(
            response=response,
            timestamp=datetime.utcnow(),
            hash=h,
            metadata=metadata or {}
        )
        self.cache[h] = entry
        logger.info(f"Cached response for hash: {h}")

    def clear(self) -> None:
        self.cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics including size, hit rate, etc.
        """
        stats = {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expiration_hours": self.expiration_hours,
            "oldest_entry": min((e.timestamp for e in self.cache.values()), default=None),
            "newest_entry": max((e.timestamp for e in self.cache.values()), default=None)
        }
        
        if self.llm_params:
            stats["llm_params"] = self.llm_params
            stats["params_file"] = self.PARAMS_FILE
        
        return stats

    def clean_cache(
        self,
        mode: str = "expired",
        min_similarity: float = 0.95,
        max_age_hours: Optional[int] = None,
        min_hits: int = 0,
        max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Clean the cache based on various criteria.

        Args:
            mode (str): Cleaning mode, one of:
                - "expired": Remove expired entries
                - "similarity": Remove highly similar entries
                - "age": Remove entries older than max_age_hours
                - "hits": Remove entries with fewer hits than min_hits
                - "size": Reduce cache to max_size entries
                - "all": Apply all cleaning modes
            min_similarity (float): Minimum similarity threshold for removing similar entries
            max_age_hours (int): Maximum age in hours for entries
            min_hits (int): Minimum number of hits for entries to keep
            max_size (int): Maximum number of entries to keep

        Returns:
            Dict[str, Any]: Statistics about the cleaning operation
        """
        initial_size = len(self.cache)
        removed_count = 0
        removed_by_reason = {
            "expired": 0,
            "similarity": 0,
            "age": 0,
            "hits": 0,
            "size": 0
        }
        
        # Get current time for age-based cleaning
        current_time = datetime.utcnow()
        
        # Create a list of entries to remove
        entries_to_remove = set()
        
        if mode in ["expired", "all"]:
            # Remove expired entries
            for key, entry in self.cache.items():
                if current_time - entry.timestamp > timedelta(hours=self.expiration_hours):
                    entries_to_remove.add(key)
                    removed_by_reason["expired"] += 1
        
        if mode in ["age", "all"] and max_age_hours is not None:
            # Remove entries older than max_age_hours
            for key, entry in self.cache.items():
                if current_time - entry.timestamp > timedelta(hours=max_age_hours):
                    entries_to_remove.add(key)
                    removed_by_reason["age"] += 1
        
        if mode in ["hits", "all"]:
            # Remove entries with fewer hits than min_hits
            for key, entry in self.cache.items():
                hits = entry.metadata.get("hits", 0)
                if hits < min_hits:
                    entries_to_remove.add(key)
                    removed_by_reason["hits"] += 1
        
        if mode in ["size", "all"] or max_size is not None:
            # Reduce cache to max_size entries
            target_size = max_size if max_size is not None else self.max_size
            if len(self.cache) > target_size:
                # Sort entries by timestamp and remove oldest
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].timestamp
                )
                for key, _ in sorted_entries[:len(self.cache) - target_size]:
                    entries_to_remove.add(key)
                    removed_by_reason["size"] += 1
        
        # Remove the identified entries
        for key in entries_to_remove:
            if key in self.cache:
                del self.cache[key]
                removed_count += 1
        
        # Prepare statistics
        stats = {
            "initial_size": initial_size,
            "final_size": len(self.cache),
            "removed_count": removed_count,
            "removed_by_reason": removed_by_reason,
            "cleaning_mode": mode,
            "timestamp": current_time.isoformat()
        }
        
        logger.info(f"Cache cleaned: {removed_count} entries removed")
        logger.debug(f"Cleaning statistics: {json.dumps(stats, indent=2)}")
        
        return stats

    def _update_vectors(self) -> None:
        """Update the TF-IDF vectors after cache modifications."""
        if not self.cache:
            self.vectors = None
            self.vector_timestamps = []
            return
            
        texts = [entry.metadata.get("query_text", "") for entry in self.cache.values()]
        self.vectors = self.vectorizer.fit_transform(texts).toarray()
        self.vector_timestamps = [entry.timestamp for entry in self.cache.values()] 