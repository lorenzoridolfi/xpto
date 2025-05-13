"""
LLM Cache System

This module implements a caching system for LLM responses to reduce API calls and improve performance.
The cache uses a combination of input hashing and similarity matching to identify similar requests.
"""

import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

logger = logging.getLogger("LLMCache")

@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    response: Any
    timestamp: datetime
    hash: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0

@dataclass
class LLMParameters:
    """Represents LLM parameters for cache validation."""
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float

    def __hash__(self) -> int:
        """Compute hash of parameters for comparison."""
        return hash((
            self.model_name,
            self.temperature,
            self.max_tokens,
            self.top_p,
            self.frequency_penalty,
            self.presence_penalty
        ))

    def __eq__(self, other: Any) -> bool:
        """Compare parameters for equality."""
        if not isinstance(other, LLMParameters):
            return False
        return (
            self.model_name == other.model_name and
            self.temperature == other.temperature and
            self.max_tokens == other.max_tokens and
            self.top_p == other.top_p and
            self.frequency_penalty == other.frequency_penalty and
            self.presence_penalty == other.presence_penalty
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMParameters':
        """Create parameters from dictionary."""
        return cls(
            model_name=data["model_name"],
            temperature=float(data["temperature"]),
            max_tokens=int(data["max_tokens"]),
            top_p=float(data["top_p"]),
            frequency_penalty=float(data["frequency_penalty"]),
            presence_penalty=float(data["presence_penalty"])
        )

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
        similarity_threshold: float = 0.85,
        expiration_hours: int = 24,
        vectorizer: Optional[TfidfVectorizer] = None,
        llm_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LLM cache.

        Args:
            max_size (int): Maximum number of cache entries
            similarity_threshold (float): Threshold for considering responses similar
            expiration_hours (int): Hours before cache entries expire
            vectorizer (Optional[TfidfVectorizer]): Custom vectorizer for text similarity
            llm_params (Optional[Dict[str, Any]]): Initial LLM parameters
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.expiration_hours = expiration_hours
        self.vectorizer = vectorizer or TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._initialize_vectorizer()
        
        # Initialize LLM parameters
        self.llm_params = None
        if llm_params:
            self.update_llm_params(llm_params)
        else:
            self._load_saved_params()

    def _load_saved_params(self) -> None:
        """Load saved LLM parameters from file."""
        try:
            if os.path.exists(self.PARAMS_FILE):
                with open(self.PARAMS_FILE, 'r') as f:
                    data = json.load(f)
                    self.llm_params = LLMParameters.from_dict(data)
                    logger.info(f"Loaded saved LLM parameters from {self.PARAMS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load saved LLM parameters: {e}")

    def _save_params(self) -> None:
        """Save current LLM parameters to file."""
        if self.llm_params:
            try:
                with open(self.PARAMS_FILE, 'w') as f:
                    json.dump(self.llm_params.to_dict(), f, indent=2)
                logger.info(f"Saved LLM parameters to {self.PARAMS_FILE}")
            except Exception as e:
                logger.error(f"Failed to save LLM parameters: {e}")

    def update_llm_params(self, params: Dict[str, Any]) -> bool:
        """
        Update LLM parameters and clear cache if parameters changed.

        Args:
            params (Dict[str, Any]): New LLM parameters

        Returns:
            bool: True if cache was cleared due to parameter changes
        """
        new_params = LLMParameters(
            model_name=params.get("model", ""),
            temperature=float(params.get("temperature", 0.0)),
            max_tokens=int(params.get("max_tokens", 0)),
            top_p=float(params.get("top_p", 0.0)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            presence_penalty=float(params.get("presence_penalty", 0.0))
        )

        if self.llm_params is None:
            self.llm_params = new_params
            self._save_params()
            return False

        if self.llm_params != new_params:
            logger.info("LLM parameters changed, clearing cache")
            logger.debug(f"Old parameters: {self.llm_params}")
            logger.debug(f"New parameters: {new_params}")
            self.clear()
            self.llm_params = new_params
            self._save_params()
            return True

        return False

    def _initialize_vectorizer(self) -> None:
        """Initialize the TF-IDF vectorizer with some common patterns."""
        common_patterns = [
            "analyze the following",
            "summarize the content",
            "verify the information",
            "check the quality",
            "process the file",
            "generate a response",
            "validate the output"
        ]
        self.vectorizer.fit(common_patterns)

    def _compute_hash(self, messages: List[Dict[str, Any]]) -> str:
        """
        Compute a hash for the input messages.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries

        Returns:
            str: Hash of the messages
        """
        # Sort messages to ensure consistent hashing
        sorted_messages = sorted(
            [json.dumps(m, sort_keys=True) for m in messages]
        )
        return hashlib.sha256("".join(sorted_messages).encode()).hexdigest()

    def _compute_similarity(self, query: str, cache_entries: List[Tuple[str, CacheEntry]]) -> List[Tuple[str, float]]:
        """
        Compute similarity scores between query and cache entries.

        Args:
            query (str): Query text
            cache_entries (List[Tuple[str, CacheEntry]]): List of cache entries

        Returns:
            List[Tuple[str, float]]: List of (hash, similarity_score) tuples
        """
        if not cache_entries:
            return []

        # Transform query and cache entries
        query_vec = self.vectorizer.transform([query])
        cache_texts = [entry.metadata.get("query_text", "") for _, entry in cache_entries]
        cache_vecs = self.vectorizer.transform(cache_texts)

        # Compute similarities
        similarities = cosine_similarity(query_vec, cache_vecs)[0]
        return [(hash, score) for (hash, _), score in zip(cache_entries, similarities)]

    def _clean_expired_entries(self) -> None:
        """Remove expired cache entries."""
        now = datetime.utcnow()
        expired = [
            hash for hash, entry in self.cache.items()
            if now - entry.timestamp > timedelta(hours=self.expiration_hours)
        ]
        for hash in expired:
            del self.cache[hash]

    def _clean_oldest_entries(self) -> None:
        """Remove oldest entries if cache is full."""
        if len(self.cache) >= self.max_size:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
            entries_to_remove = sorted_entries[:len(sorted_entries) // 10]  # Remove 10% of oldest entries
            for hash, _ in entries_to_remove:
                del self.cache[hash]

    def get(
        self,
        messages: List[Dict[str, Any]],
        query_text: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get a response from the cache.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries
            query_text (Optional[str]): Text representation of the query for similarity matching

        Returns:
            Optional[Any]: Cached response if found, None otherwise
        """
        self._clean_expired_entries()
        
        # Try exact match first
        hash = self._compute_hash(messages)
        if hash in self.cache:
            entry = self.cache[hash]
            entry.timestamp = datetime.utcnow()  # Update timestamp
            logger.info(f"Cache hit (exact match) for hash: {hash}")
            return entry.response

        # Try similarity match if query_text is provided
        if query_text:
            cache_entries = list(self.cache.items())
            similarities = self._compute_similarity(query_text, cache_entries)
            
            # Find best match above threshold
            best_match = max(similarities, key=lambda x: x[1], default=(None, 0))
            if best_match[1] >= self.similarity_threshold:
                hash, score = best_match
                entry = self.cache[hash]
                entry.timestamp = datetime.utcnow()
                entry.similarity_score = score
                logger.info(f"Cache hit (similarity: {score:.2f}) for hash: {hash}")
                return entry.response

        logger.info("Cache miss")
        return None

    def put(
        self,
        messages: List[Dict[str, Any]],
        response: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a response in the cache.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries
            response (Any): Response to cache
            metadata (Optional[Dict[str, Any]]): Additional metadata to store
        """
        self._clean_expired_entries()
        self._clean_oldest_entries()

        hash = self._compute_hash(messages)
        query_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))

        entry = CacheEntry(
            response=response,
            timestamp=datetime.utcnow(),
            hash=hash,
            metadata={
                "query_text": query_text,
                **(metadata or {})
            }
        )

        self.cache[hash] = entry
        logger.info(f"Cached response for hash: {hash}")

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
            "similarity_threshold": self.similarity_threshold,
            "oldest_entry": min((e.timestamp for e in self.cache.values()), default=None),
            "newest_entry": max((e.timestamp for e in self.cache.values()), default=None)
        }
        
        if self.llm_params:
            stats["llm_params"] = self.llm_params.to_dict()
            stats["params_file"] = self.PARAMS_FILE
        
        return stats

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")

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
        
        if mode in ["similarity", "all"]:
            # Remove highly similar entries
            if self.vectors is not None and len(self.vectors) > 1:
                similarity_matrix = cosine_similarity(self.vectors)
                for i in range(len(similarity_matrix)):
                    for j in range(i + 1, len(similarity_matrix)):
                        if similarity_matrix[i][j] > min_similarity:
                            # Keep the newer entry
                            if self.vector_timestamps[i] < self.vector_timestamps[j]:
                                entries_to_remove.add(list(self.cache.keys())[i])
                            else:
                                entries_to_remove.add(list(self.cache.keys())[j])
                            removed_by_reason["similarity"] += 1
        
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
        
        # Update vectors if needed
        if removed_count > 0:
            self._update_vectors()
        
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