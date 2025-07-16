"""
Base repository class for the Options Trading Engine.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime, timedelta
import logging

from ...infrastructure.error_handling import RepositoryError, ValidationError
from ...infrastructure.cache import CacheManager

T = TypeVar('T')

logger = logging.getLogger(__name__)


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository class providing common functionality
    for all data repositories.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logger
    
    @abstractmethod
    def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def create(self, entity: T) -> T:
        """Create a new entity."""
        pass
    
    @abstractmethod
    def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete an entity by ID."""
        pass
    
    @abstractmethod
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """Find all entities matching filters."""
        pass
    
    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments."""
        key_parts = [str(arg) for arg in args]
        return f"{prefix}:{':'.join(key_parts)}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache."""
        try:
            return self.cache_manager.get(cache_key)
        except Exception as e:
            self.logger.warning(f"Cache get failed for key {cache_key}: {e}")
            return None
    
    def _set_to_cache(self, cache_key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set data to cache."""
        try:
            self.cache_manager.set(cache_key, data, ttl=ttl)
        except Exception as e:
            self.logger.warning(f"Cache set failed for key {cache_key}: {e}")
    
    def _invalidate_cache(self, cache_key: str) -> None:
        """Invalidate cache entry."""
        try:
            self.cache_manager.delete(cache_key)
        except Exception as e:
            self.logger.warning(f"Cache invalidation failed for key {cache_key}: {e}")
    
    def _validate_entity(self, entity: T) -> None:
        """Validate entity before persistence."""
        if entity is None:
            raise ValidationError("Entity cannot be None")
    
    def _handle_repository_error(self, operation: str, error: Exception) -> None:
        """Handle repository errors consistently."""
        error_msg = f"Repository operation '{operation}' failed: {str(error)}"
        self.logger.error(error_msg)
        raise RepositoryError(error_msg) from error
    
    def _get_timestamp(self) -> datetime:
        """Get current timestamp."""
        return datetime.now()
    
    def _is_data_fresh(self, timestamp: datetime, max_age_minutes: int = 60) -> bool:
        """Check if data is still fresh based on timestamp."""
        if timestamp is None:
            return False
        
        age = datetime.now() - timestamp
        return age <= timedelta(minutes=max_age_minutes)
    
    def _apply_filters(self, data: List[T], filters: Dict[str, Any]) -> List[T]:
        """Apply filters to data list."""
        if not filters:
            return data
        
        filtered_data = []
        for item in data:
            match = True
            for key, value in filters.items():
                if hasattr(item, key):
                    item_value = getattr(item, key)
                    if item_value != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                filtered_data.append(item)
        
        return filtered_data
    
    def _sort_data(self, data: List[T], sort_by: str, descending: bool = False) -> List[T]:
        """Sort data by specified field."""
        if not sort_by:
            return data
        
        try:
            return sorted(
                data,
                key=lambda x: getattr(x, sort_by, None) or 0,
                reverse=descending
            )
        except Exception as e:
            self.logger.warning(f"Sort failed for field {sort_by}: {e}")
            return data
    
    def _paginate_data(self, data: List[T], page: int = 1, page_size: int = 100) -> List[T]:
        """Paginate data."""
        if page < 1:
            page = 1
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return data[start_idx:end_idx]
    
    def clear_cache(self) -> None:
        """Clear all cache entries for this repository."""
        try:
            self.cache_manager.clear()
        except Exception as e:
            self.logger.warning(f"Cache clear failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            return self.cache_manager.get_stats()
        except Exception as e:
            self.logger.warning(f"Cache stats retrieval failed: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on repository."""
        return {
            'status': 'healthy',
            'timestamp': self._get_timestamp().isoformat(),
            'cache_stats': self.get_cache_stats()
        }