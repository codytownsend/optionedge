"""
Base repository abstract class and common repository functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Generic type for repository entities
T = TypeVar('T')


class RepositoryError(Exception):
    """Base exception for repository operations."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class DataNotFoundError(RepositoryError):
    """Raised when requested data is not found."""
    pass


class DataValidationError(RepositoryError):
    """Raised when data validation fails."""
    pass


class CacheExpiredError(RepositoryError):
    """Raised when cached data has expired."""
    pass


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base class for all repositories.
    
    Provides common functionality for data access, caching, and validation.
    """
    
    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Initialize base repository.
        
        Args:
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """Get all entities matching filters."""
        pass
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        pass
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = []
        
        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={value}")
        
        return ":".join(key_parts)
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = cache_entry['timestamp']
        expiry_time = cache_time + timedelta(seconds=self.cache_ttl_seconds)
        
        return datetime.utcnow() < expiry_time
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if valid."""
        if cache_key not in self._cache:
            return None
        
        cache_entry = self._cache[cache_key]
        
        if not self._is_cache_valid(cache_entry):
            # Remove expired entry
            del self._cache[cache_key]
            return None
        
        self.logger.debug(f"Cache hit for key: {cache_key}")
        return cache_entry['data']
    
    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': datetime.utcnow()
        }
        self.logger.debug(f"Cache set for key: {cache_key}")
    
    def _clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cache entries, optionally matching a pattern."""
        if pattern is None:
            self._cache.clear()
            self.logger.debug("Entire cache cleared")
        else:
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self._cache[key]
            self.logger.debug(f"Cache cleared for pattern: {pattern}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        valid_entries = sum(1 for entry in self._cache.values() 
                          if self._is_cache_valid(entry))
        expired_entries = total_entries - valid_entries
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'hit_rate': self._calculate_hit_rate(),
            'cache_ttl_seconds': self.cache_ttl_seconds
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified implementation)."""
        # This would need to track hits/misses in a real implementation
        # For now, return 0.0 as placeholder
        return 0.0
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries and return count removed."""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if not self._is_cache_valid(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def validate_entity(self, entity: T) -> bool:
        """
        Validate entity before save/update operations.
        
        Override in subclasses for specific validation logic.
        """
        return entity is not None
    
    def _handle_error(self, operation: str, error: Exception) -> None:
        """Handle and log repository errors."""
        error_msg = f"Repository operation '{operation}' failed: {error}"
        self.logger.error(error_msg)
        
        # Re-raise as RepositoryError
        if isinstance(error, RepositoryError):
            raise error
        else:
            raise RepositoryError(error_msg, error)
    
    def exists(self, entity_id: str) -> bool:
        """Check if entity exists by ID."""
        try:
            entity = self.get_by_id(entity_id)
            return entity is not None
        except DataNotFoundError:
            return False
        except Exception as e:
            self._handle_error("exists", e)
            return False
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters."""
        try:
            entities = self.get_all(filters)
            return len(entities)
        except Exception as e:
            self._handle_error("count", e)
            return 0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get repository health status."""
        try:
            cache_stats = self.get_cache_stats()
            
            return {
                'repository_class': self.__class__.__name__,
                'status': 'healthy',
                'cache_stats': cache_stats,
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'repository_class': self.__class__.__name__,
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }


class InMemoryRepository(BaseRepository[T]):
    """
    In-memory implementation of repository for testing and development.
    """
    
    def __init__(self, cache_ttl_seconds: int = 300):
        super().__init__(cache_ttl_seconds)
        self._storage: Dict[str, T] = {}
        self._next_id = 1
    
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID from memory storage."""
        try:
            entity = self._storage.get(entity_id)
            if entity is None:
                raise DataNotFoundError(f"Entity with ID {entity_id} not found")
            return entity
        except Exception as e:
            if isinstance(e, DataNotFoundError):
                raise
            self._handle_error("get_by_id", e)
            return None
    
    def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """Get all entities from memory storage."""
        try:
            entities = list(self._storage.values())
            
            # Apply filters if provided
            if filters:
                entities = self._apply_filters(entities, filters)
            
            return entities
        except Exception as e:
            self._handle_error("get_all", e)
            return []
    
    def save(self, entity: T) -> T:
        """Save entity to memory storage."""
        try:
            if not self.validate_entity(entity):
                raise DataValidationError("Entity validation failed")
            
            # Generate ID if needed (simplified approach)
            entity_id = getattr(entity, 'id', None) or str(self._next_id)
            if entity_id == str(self._next_id):
                self._next_id += 1
            
            self._storage[entity_id] = entity
            self.logger.debug(f"Entity saved with ID: {entity_id}")
            
            return entity
        except Exception as e:
            self._handle_error("save", e)
            raise
    
    def delete(self, entity_id: str) -> bool:
        """Delete entity from memory storage."""
        try:
            if entity_id in self._storage:
                del self._storage[entity_id]
                self.logger.debug(f"Entity deleted with ID: {entity_id}")
                return True
            else:
                raise DataNotFoundError(f"Entity with ID {entity_id} not found")
        except Exception as e:
            self._handle_error("delete", e)
            return False
    
    def _apply_filters(self, entities: List[T], filters: Dict[str, Any]) -> List[T]:
        """Apply filters to entity list (simplified implementation)."""
        filtered_entities = []
        
        for entity in entities:
            match = True
            
            for filter_key, filter_value in filters.items():
                entity_value = getattr(entity, filter_key, None)
                
                if entity_value != filter_value:
                    match = False
                    break
            
            if match:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def clear_all(self) -> None:
        """Clear all entities from memory storage."""
        self._storage.clear()
        self._clear_cache()
        self.logger.debug("All entities cleared from memory storage")