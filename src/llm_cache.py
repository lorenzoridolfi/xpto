"""
LLM Cache System

This module implements a sophisticated caching system for LLM responses to reduce API calls
and improve performance. The cache uses a combination of input hashing and similarity
matching to identify similar requests.

Features:
- Exact match caching using message hashing
- Similarity-based caching using TF-IDF and cosine similarity
- Cache entry expiration and cleanup
- Comprehensive metadata tracking
- LLM parameter validation
- Multi-language support
- Performance optimization
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
import time
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class CacheEntry:
    """
    Represents a single cache entry with metadata.
    
    Attributes:
        response (Any): The cached response
        timestamp (float): When the entry was created
        hits (int): Number of times this entry was accessed
        metadata (Dict[str, Any]): Additional metadata about the entry
        vector (Optional[np.ndarray]): TF-IDF vector for similarity matching
    """
    response: Any
    timestamp: float
    hits: int = 0
    metadata: Dict[str, Any] = None
    vector: Optional[np.ndarray] = None

@dataclass
class LLMParams:
    """
    Represents LLM parameters for cache validation.
    
    Attributes:
        model (str): Model name/identifier
        temperature (float): Temperature setting
        max_tokens (int): Maximum tokens to generate
        top_p (float): Top-p sampling parameter
        frequency_penalty (float): Frequency penalty
        presence_penalty (float): Presence penalty
    """
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float

    def compute_hash(self) -> str:
        """
        Compute hash of parameters for comparison.
        
        Returns:
            str: Hash of the parameters
        """
        param_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()

    def __eq__(self, other: 'LLMParams') -> bool:
        """
        Compare parameters for equality.
        
        Args:
            other (LLMParams): Other parameters to compare with
            
        Returns:
            bool: True if parameters are equal
        """
        return self.compute_hash() == other.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of parameters
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMParams':
        """
        Create parameters from dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing parameters
            
        Returns:
            LLMParams: New LLMParams instance
        """
        return cls(**data)

class LLMCache:
    """
    A sophisticated caching system for LLM responses.
    
    This class provides:
    - Exact match caching using message hashing
    - Similarity-based caching using TF-IDF and cosine similarity
    - Cache entry expiration and cleanup
    - Comprehensive metadata tracking
    - LLM parameter validation
    - Multi-language support
    - Performance optimization
    
    Attributes:
        max_size (int): Maximum number of cache entries
        similarity_threshold (float): Threshold for considering responses similar
        expiration_hours (int): Hours before cache entries expire
        vectorizer (TfidfVectorizer): Vectorizer for text similarity
        llm_params (Optional[LLMParams]): Current LLM parameters
        language (str): Language for tokenization
        cache (Dict[str, CacheEntry]): Cache storage
    """
    
    def __init__(self,
                 max_size: int = 1000,
                 similarity_threshold: float = 0.85,
                 expiration_hours: int = 24,
                 vectorizer: Optional[TfidfVectorizer] = None,
                 llm_params: Optional[Dict[str, Any]] = None,
                 language: str = "portuguese"):
        """
        Initialize the LLM cache.

        Args:
            max_size (int): Maximum number of cache entries
            similarity_threshold (float): Threshold for considering responses similar
            expiration_hours (int): Hours before cache entries expire
            vectorizer (Optional[TfidfVectorizer]): Custom vectorizer for text similarity
            llm_params (Optional[Dict[str, Any]]): Initial LLM parameters
            language (str): Language for tokenization and stopwords
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.expiration_hours = expiration_hours
        self.language = language
        self.cache: Dict[str, CacheEntry] = {}
        
        # Initialize vectorizer
        self.vectorizer = vectorizer or self._initialize_vectorizer()
        
        # Initialize LLM parameters
        self.llm_params = LLMParams.from_dict(llm_params) if llm_params else None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK's tokenizer.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        return word_tokenize(text.lower(), language=self.language)

    def _initialize_vectorizer(self) -> TfidfVectorizer:
        """
        Initialize the TF-IDF vectorizer with common patterns.
        
        Returns:
            TfidfVectorizer: Initialized vectorizer
        """
        return TfidfVectorizer(
            tokenizer=self._tokenize_text,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

    def _compute_hash(self, messages: List[Dict[str, Any]]) -> str:
        """
        Compute a hash for the input messages.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries

        Returns:
            str: Hash of the messages
        """
        # Sort messages to ensure consistent hashing
        sorted_messages = sorted(messages, key=lambda x: json.dumps(x, sort_keys=True))
        message_str = json.dumps(sorted_messages, sort_keys=True)
        return hashlib.sha256(message_str.encode()).hexdigest()

    def _compute_similarity(self,
                          query: str,
                          cache_entries: List[Tuple[str, CacheEntry]]) -> List[Tuple[str, float]]:
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
            
        # Get vectors for all entries
        vectors = [entry.vector for _, entry in cache_entries if entry.vector is not None]
        if not vectors:
            return []
            
        # Compute query vector
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, vectors)[0]
        
        # Return (hash, similarity) pairs
        return [(hash, score) for (hash, _), score in zip(cache_entries, similarities)]

    def _remove_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.expiration_hours * 3600
        ]
        for key in expired:
            del self.cache[key]

    def _remove_oldest(self) -> None:
        """Remove oldest entries if cache is full."""
        if len(self.cache) >= self.max_size:
            # Sort by timestamp and remove oldest
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
            for key, _ in sorted_entries[:len(sorted_entries) - self.max_size + 1]:
                del self.cache[key]

    def get(self,
            messages: List[Dict[str, Any]],
            query_text: Optional[str] = None) -> Optional[Any]:
        """
        Get a response from the cache.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries
            query_text (Optional[str]): Text representation of the query for similarity matching

        Returns:
            Optional[Any]: Cached response if found, None otherwise
        """
        # Remove expired entries
        self._remove_expired()
        
        # Try exact match first
        message_hash = self._compute_hash(messages)
        if message_hash in self.cache:
            entry = self.cache[message_hash]
            entry.hits += 1
            return entry.response
            
        # Try similarity match if query text provided
        if query_text:
            cache_entries = list(self.cache.items())
            similarities = self._compute_similarity(query_text, cache_entries)
            
            # Find best match above threshold
            best_match = max(similarities, key=lambda x: x[1], default=(None, 0))
            if best_match[1] >= self.similarity_threshold:
                hash_key, _ = best_match
                entry = self.cache[hash_key]
                entry.hits += 1
                return entry.response
                
        return None

    def set(self,
            messages: List[Dict[str, Any]],
            response: Any,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a response in the cache.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries
            response (Any): Response to cache
            metadata (Optional[Dict[str, Any]]): Additional metadata to store
        """
        # Remove expired entries
        self._remove_expired()
        
        # Compute hash and vector
        message_hash = self._compute_hash(messages)
        message_text = " ".join(msg.get("content", "") for msg in messages)
        vector = self.vectorizer.transform([message_text]).toarray()[0]
        
        # Create cache entry
        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            metadata=metadata or {},
            vector=vector
        )
        
        # Store in cache
        self.cache[message_hash] = entry
        
        # Remove oldest if needed
        self._remove_oldest()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics including:
                - size: Current number of entries
                - hits: Total number of cache hits
                - misses: Total number of cache misses
                - hit_rate: Cache hit rate
                - avg_hits: Average hits per entry
                - oldest_entry: Timestamp of oldest entry
                - newest_entry: Timestamp of newest entry
        """
        if not self.cache:
            return {
                "size": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "avg_hits": 0.0,
                "oldest_entry": None,
                "newest_entry": None
            }
            
        total_hits = sum(entry.hits for entry in self.cache.values())
        total_entries = len(self.cache)
        
        return {
            "size": total_entries,
            "hits": total_hits,
            "misses": total_entries - total_hits,
            "hit_rate": total_hits / total_entries if total_entries > 0 else 0.0,
            "avg_hits": total_hits / total_entries if total_entries > 0 else 0.0,
            "oldest_entry": min(entry.timestamp for entry in self.cache.values()),
            "newest_entry": max(entry.timestamp for entry in self.cache.values())
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.vectorizer = self._initialize_vectorizer()

    def clean(self,
              mode: str = "all",
              min_similarity: float = 0.95,
              max_age_hours: int = 24,
              min_hits: int = 1,
              max_size: int = 1000) -> Dict[str, Any]:
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
            Dict[str, Any]: Statistics about the cleaning operation including:
                - removed_entries: Number of entries removed
                - remaining_entries: Number of entries remaining
                - removed_by_mode: Dictionary of entries removed by each mode
        """
        initial_size = len(self.cache)
        removed_by_mode = {}
        
        if mode in ["expired", "all"]:
            current_time = time.time()
            expired = [
                key for key, entry in self.cache.items()
                if current_time - entry.timestamp > self.expiration_hours * 3600
            ]
            for key in expired:
                del self.cache[key]
            removed_by_mode["expired"] = len(expired)
            
        if mode in ["similarity", "all"]:
            # Group similar entries
            entries = list(self.cache.items())
            to_remove = set()
            
            for i, (key1, entry1) in enumerate(entries):
                if key1 in to_remove:
                    continue
                    
                for key2, entry2 in entries[i+1:]:
                    if key2 in to_remove:
                        continue
                        
                    if entry1.vector is not None and entry2.vector is not None:
                        similarity = cosine_similarity(
                            entry1.vector.reshape(1, -1),
                            entry2.vector.reshape(1, -1)
                        )[0][0]
                        
                        if similarity >= min_similarity:
                            # Keep the one with more hits
                            if entry1.hits >= entry2.hits:
                                to_remove.add(key2)
                            else:
                                to_remove.add(key1)
                                break
                                
            for key in to_remove:
                del self.cache[key]
            removed_by_mode["similarity"] = len(to_remove)
            
        if mode in ["age", "all"]:
            current_time = time.time()
            old = [
                key for key, entry in self.cache.items()
                if current_time - entry.timestamp > max_age_hours * 3600
            ]
            for key in old:
                del self.cache[key]
            removed_by_mode["age"] = len(old)
            
        if mode in ["hits", "all"]:
            low_hits = [
                key for key, entry in self.cache.items()
                if entry.hits < min_hits
            ]
            for key in low_hits:
                del self.cache[key]
            removed_by_mode["hits"] = len(low_hits)
            
        if mode in ["size", "all"]:
            if len(self.cache) > max_size:
                # Sort by hits and timestamp
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: (-x[1].hits, x[1].timestamp)
                )
                to_remove = sorted_entries[max_size:]
                for key, _ in to_remove:
                    del self.cache[key]
                removed_by_mode["size"] = len(to_remove)
                
        # Update vectors after cleaning
        self._update_vectors()
        
        return {
            "removed_entries": initial_size - len(self.cache),
            "remaining_entries": len(self.cache),
            "removed_by_mode": removed_by_mode
        }

    def _update_vectors(self) -> None:
        """Update the TF-IDF vectors after cache modifications."""
        if not self.cache:
            return
            
        # Get all texts
        texts = [
            " ".join(msg.get("content", "") for msg in entry.metadata.get("messages", []))
            for entry in self.cache.values()
        ]
        
        # Update vectorizer
        self.vectorizer.fit(texts)
        
        # Update vectors
        for entry in self.cache.values():
            text = " ".join(msg.get("content", "") for msg in entry.metadata.get("messages", []))
            entry.vector = self.vectorizer.transform([text]).toarray()[0] 