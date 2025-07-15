"""Metrics collection and monitoring utilities."""

import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import threading
import statistics


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class MetricType:
    """Metric type constants."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: str
    description: str
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Counter metric implementation."""
    
    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value = 0.0
        self._lock = threading.Lock()
    
    def increment(self, amount: float = 1.0):
        """Increment counter by amount."""
        with self._lock:
            self._value += amount
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0.0
    
    def get_value(self) -> float:
        """Get current counter value."""
        with self._lock:
            return self._value


class Gauge:
    """Gauge metric implementation."""
    
    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set_value(self, value: float):
        """Set gauge value."""
        with self._lock:
            self._value = value
    
    def increment(self, amount: float = 1.0):
        """Increment gauge by amount."""
        with self._lock:
            self._value += amount
    
    def decrement(self, amount: float = 1.0):
        """Decrement gauge by amount."""
        with self._lock:
            self._value -= amount
    
    def get_value(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value


class Histogram:
    """Histogram metric implementation."""
    
    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._values = deque(maxlen=10000)  # Keep last 10k values
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        """Observe a value."""
        with self._lock:
            self._values.append(value)
    
    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            if not self._values:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "median": 0.0,
                    "p95": 0.0,
                    "p99": 0.0
                }
            
            values = list(self._values)
            count = len(values)
            sum_val = sum(values)
            
            return {
                "count": count,
                "sum": sum_val,
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99)
            }
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((len(sorted_values) - 1) * p)
        return sorted_values[index]


class Timer:
    """Timer metric implementation."""
    
    def __init__(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self._histogram = Histogram(name, description, tags)
    
    def time(self, func: Optional[Callable] = None):
        """Time a function or use as context manager."""
        if func:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._histogram.observe(duration)
            return wrapper
        else:
            return self._TimerContext(self._histogram)
    
    def observe(self, duration: float):
        """Observe a duration."""
        self._histogram.observe(duration)
    
    def get_stats(self) -> Dict[str, float]:
        """Get timer statistics."""
        return self._histogram.get_stats()
    
    class _TimerContext:
        """Context manager for timing."""
        
        def __init__(self, histogram: Histogram):
            self._histogram = histogram
            self._start_time = None
        
        def __enter__(self):
            self._start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._start_time:
                duration = time.time() - self._start_time
                self._histogram.observe(duration)


class MetricsRegistry:
    """Registry for all metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._definitions: Dict[str, MetricDefinition] = {}
        self._lock = threading.Lock()
    
    def register_counter(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Counter:
        """Register a new counter metric."""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already exists")
            
            counter = Counter(name, description, tags)
            self._metrics[name] = counter
            self._definitions[name] = MetricDefinition(name, MetricType.COUNTER, description, tags=tags or {})
            return counter
    
    def register_gauge(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Gauge:
        """Register a new gauge metric."""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already exists")
            
            gauge = Gauge(name, description, tags)
            self._metrics[name] = gauge
            self._definitions[name] = MetricDefinition(name, MetricType.GAUGE, description, tags=tags or {})
            return gauge
    
    def register_histogram(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Histogram:
        """Register a new histogram metric."""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already exists")
            
            histogram = Histogram(name, description, tags)
            self._metrics[name] = histogram
            self._definitions[name] = MetricDefinition(name, MetricType.HISTOGRAM, description, tags=tags or {})
            return histogram
    
    def register_timer(self, name: str, description: str = "", tags: Optional[Dict[str, str]] = None) -> Timer:
        """Register a new timer metric."""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already exists")
            
            timer = Timer(name, description, tags)
            self._metrics[name] = timer
            self._definitions[name] = MetricDefinition(name, MetricType.TIMER, description, tags=tags or {})
            return timer
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get metric by name."""
        with self._lock:
            return self._metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        with self._lock:
            return list(self._metrics.keys())
    
    def get_definitions(self) -> Dict[str, MetricDefinition]:
        """Get all metric definitions."""
        with self._lock:
            return dict(self._definitions)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all metric values."""
        with self._lock:
            metrics_data = {}
            
            for name, metric in self._metrics.items():
                definition = self._definitions[name]
                
                if definition.type == MetricType.COUNTER:
                    metrics_data[name] = {
                        "type": definition.type,
                        "value": metric.get_value(),
                        "description": definition.description,
                        "tags": definition.tags
                    }
                elif definition.type == MetricType.GAUGE:
                    metrics_data[name] = {
                        "type": definition.type,
                        "value": metric.get_value(),
                        "description": definition.description,
                        "tags": definition.tags
                    }
                elif definition.type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                    metrics_data[name] = {
                        "type": definition.type,
                        "stats": metric.get_stats(),
                        "description": definition.description,
                        "tags": definition.tags
                    }
            
            return metrics_data


class ApplicationMetrics:
    """Application-specific metrics."""
    
    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup application-specific metrics."""
        # API call metrics
        self.api_calls_total = self.registry.register_counter(
            "api_calls_total",
            "Total number of API calls",
            {"service": "options_engine"}
        )
        
        self.api_call_duration = self.registry.register_timer(
            "api_call_duration_seconds",
            "API call duration in seconds"
        )
        
        self.api_errors_total = self.registry.register_counter(
            "api_errors_total",
            "Total number of API errors"
        )
        
        # Strategy generation metrics
        self.strategies_generated = self.registry.register_counter(
            "strategies_generated_total",
            "Total number of strategies generated"
        )
        
        self.strategy_generation_duration = self.registry.register_timer(
            "strategy_generation_duration_seconds",
            "Strategy generation duration in seconds"
        )
        
        # Trade selection metrics
        self.trades_selected = self.registry.register_counter(
            "trades_selected_total",
            "Total number of trades selected"
        )
        
        self.trade_selection_duration = self.registry.register_timer(
            "trade_selection_duration_seconds",
            "Trade selection duration in seconds"
        )
        
        # Market data metrics
        self.market_data_fetches = self.registry.register_counter(
            "market_data_fetches_total",
            "Total number of market data fetches"
        )
        
        self.market_data_age = self.registry.register_gauge(
            "market_data_age_seconds",
            "Age of market data in seconds"
        )
        
        # Cache metrics
        self.cache_hits = self.registry.register_counter(
            "cache_hits_total",
            "Total number of cache hits"
        )
        
        self.cache_misses = self.registry.register_counter(
            "cache_misses_total",
            "Total number of cache misses"
        )
        
        # Portfolio metrics
        self.portfolio_delta = self.registry.register_gauge(
            "portfolio_delta",
            "Current portfolio delta"
        )
        
        self.portfolio_vega = self.registry.register_gauge(
            "portfolio_vega",
            "Current portfolio vega"
        )
        
        self.portfolio_value = self.registry.register_gauge(
            "portfolio_value_dollars",
            "Current portfolio value in dollars"
        )
    
    def record_api_call(self, endpoint: str, duration: float, success: bool = True):
        """Record an API call."""
        self.api_calls_total.increment()
        self.api_call_duration.observe(duration)
        
        if not success:
            self.api_errors_total.increment()
    
    def record_strategy_generation(self, count: int, duration: float):
        """Record strategy generation."""
        self.strategies_generated.increment(count)
        self.strategy_generation_duration.observe(duration)
    
    def record_trade_selection(self, count: int, duration: float):
        """Record trade selection."""
        self.trades_selected.increment(count)
        self.trade_selection_duration.observe(duration)
    
    def record_market_data_fetch(self, age_seconds: float):
        """Record market data fetch."""
        self.market_data_fetches.increment()
        self.market_data_age.set_value(age_seconds)
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits.increment()
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses.increment()
    
    def update_portfolio_metrics(self, delta: float, vega: float, value: float):
        """Update portfolio metrics."""
        self.portfolio_delta.set_value(delta)
        self.portfolio_vega.set_value(vega)
        self.portfolio_value.set_value(value)


# Global metrics instances
metrics_registry = MetricsRegistry()
app_metrics = ApplicationMetrics(metrics_registry)


def timed(metric_name: str, description: str = ""):
    """Decorator to time function execution."""
    def decorator(func):
        timer = metrics_registry.register_timer(metric_name, description)
        return timer.time(func)
    return decorator


def count_calls(metric_name: str, description: str = ""):
    """Decorator to count function calls."""
    def decorator(func):
        counter = metrics_registry.register_counter(metric_name, description)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            counter.increment()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator