"""
Continuous improvement framework with feedback collection for the options trading engine.
Collects performance data, user feedback, and system metrics to drive improvements.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
import numpy as np
from pathlib import Path

from ..error_handling import handle_errors, FeedbackError
from ..monitoring.monitoring_system import get_monitoring_system


class FeedbackType(Enum):
    """Types of feedback."""
    PERFORMANCE = "performance"
    USER_EXPERIENCE = "user_experience"
    STRATEGY_RESULT = "strategy_result"
    SYSTEM_ERROR = "system_error"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    OPTIMIZATION = "optimization"


class FeedbackPriority(Enum):
    """Feedback priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeedbackEntry:
    """Feedback entry structure."""
    id: str
    feedback_type: FeedbackType
    priority: FeedbackPriority
    title: str
    description: str
    context: Dict[str, Any]
    timestamp: datetime
    source: str = "system"
    user_id: Optional[str] = None
    processed: bool = False
    processed_at: Optional[datetime] = None
    resolution: Optional[str] = None
    impact_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    metric_name: str
    baseline_value: float
    target_value: float
    current_value: float
    trend: str  # "improving", "stable", "declining"
    last_updated: datetime
    sample_count: int


@dataclass
class ImprovementRecommendation:
    """Improvement recommendation."""
    id: str
    category: str
    title: str
    description: str
    impact_estimate: float
    effort_estimate: float
    priority_score: float
    implementation_steps: List[str]
    expected_benefits: List[str]
    metrics_to_track: List[str]
    created_at: datetime
    status: str = "pending"  # pending, approved, implemented, rejected


class ContinuousImprovementSystem:
    """
    Continuous improvement system with comprehensive feedback collection.
    
    Features:
    - Performance baseline tracking and comparison
    - Automated feedback collection from system metrics
    - User feedback integration
    - Improvement recommendation generation
    - A/B testing framework
    - Performance regression detection
    - Automated optimization suggestions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.config = config or {}
        
        # Initialize database
        self.db_path = self.config.get('database_path', 'data/feedback.db')
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Core components
        self.monitoring_system = get_monitoring_system()
        
        # Feedback collection
        self.feedback_queue = deque()
        self.feedback_handlers = []
        
        # Performance baselines
        self.performance_baselines = {}
        self.baseline_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Improvement tracking
        self.improvement_recommendations = {}
        self.implemented_improvements = {}
        
        # A/B testing
        self.ab_tests = {}
        self.ab_test_results = {}
        
        # Analysis settings
        self.analysis_interval = self.config.get('analysis_interval', 3600)  # 1 hour
        self.baseline_window = self.config.get('baseline_window', 86400)  # 24 hours
        self.improvement_threshold = self.config.get('improvement_threshold', 0.1)  # 10%
        
        # Background processing
        self.is_running = False
        self.analysis_thread = None
        
    def _initialize_database(self):
        """Initialize SQLite database for feedback storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    feedback_type TEXT,
                    priority TEXT,
                    title TEXT,
                    description TEXT,
                    context TEXT,
                    timestamp TEXT,
                    source TEXT,
                    user_id TEXT,
                    processed BOOLEAN,
                    processed_at TEXT,
                    resolution TEXT,
                    impact_score REAL,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    metric_name TEXT PRIMARY KEY,
                    baseline_value REAL,
                    target_value REAL,
                    current_value REAL,
                    trend TEXT,
                    last_updated TEXT,
                    sample_count INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS improvement_recommendations (
                    id TEXT PRIMARY KEY,
                    category TEXT,
                    title TEXT,
                    description TEXT,
                    impact_estimate REAL,
                    effort_estimate REAL,
                    priority_score REAL,
                    implementation_steps TEXT,
                    expected_benefits TEXT,
                    metrics_to_track TEXT,
                    created_at TEXT,
                    status TEXT
                )
            ''')
            
            conn.commit()
    
    @handle_errors(operation_name="start_improvement_system")
    def start(self):
        """Start the continuous improvement system."""
        if self.is_running:
            self.logger.warning("Continuous improvement system is already running")
            return
        
        self.is_running = True
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True
        )
        self.analysis_thread.start()
        
        self.logger.info("Continuous improvement system started")
    
    def stop(self):
        """Stop the continuous improvement system."""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=10)
        
        self.logger.info("Continuous improvement system stopped")
    
    def _analysis_loop(self):
        """Main analysis loop for continuous improvement."""
        while self.is_running:
            try:
                # Update performance baselines
                self._update_performance_baselines()
                
                # Analyze performance trends
                self._analyze_performance_trends()
                
                # Generate improvement recommendations
                self._generate_improvement_recommendations()
                
                # Process feedback queue
                self._process_feedback_queue()
                
                # Check for regressions
                self._check_for_regressions()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in improvement analysis loop: {str(e)}")
                time.sleep(self.analysis_interval)
    
    @handle_errors(operation_name="submit_feedback")
    def submit_feedback(self, feedback: FeedbackEntry):
        """Submit feedback to the system."""
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO feedback 
                (id, feedback_type, priority, title, description, context, timestamp, 
                 source, user_id, processed, processed_at, resolution, impact_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.id,
                feedback.feedback_type.value,
                feedback.priority.value,
                feedback.title,
                feedback.description,
                json.dumps(feedback.context),
                feedback.timestamp.isoformat(),
                feedback.source,
                feedback.user_id,
                feedback.processed,
                feedback.processed_at.isoformat() if feedback.processed_at else None,
                feedback.resolution,
                feedback.impact_score,
                json.dumps(feedback.metadata)
            ))
            conn.commit()
        
        # Add to processing queue
        self.feedback_queue.append(feedback)
        
        # Process high-priority feedback immediately
        if feedback.priority in [FeedbackPriority.HIGH, FeedbackPriority.CRITICAL]:
            self._process_feedback_entry(feedback)
        
        self.logger.info(f"Feedback submitted: {feedback.id} - {feedback.title}")
    
    def _process_feedback_queue(self):
        """Process pending feedback entries."""
        while self.feedback_queue:
            feedback = self.feedback_queue.popleft()
            if not feedback.processed:
                self._process_feedback_entry(feedback)
    
    def _process_feedback_entry(self, feedback: FeedbackEntry):
        """Process a single feedback entry."""
        try:
            # Call registered feedback handlers
            for handler in self.feedback_handlers:
                try:
                    handler(feedback)
                except Exception as e:
                    self.logger.error(f"Feedback handler failed: {str(e)}")
            
            # Update processing status
            feedback.processed = True
            feedback.processed_at = datetime.now()
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE feedback 
                    SET processed = ?, processed_at = ?, impact_score = ?
                    WHERE id = ?
                ''', (
                    True,
                    feedback.processed_at.isoformat(),
                    feedback.impact_score,
                    feedback.id
                ))
                conn.commit()
            
            self.logger.info(f"Feedback processed: {feedback.id}")
            
        except Exception as e:
            self.logger.error(f"Error processing feedback {feedback.id}: {str(e)}")
    
    @handle_errors(operation_name="update_performance_baselines")
    def _update_performance_baselines(self):
        """Update performance baselines from monitoring data."""
        # Get current metrics from monitoring system
        metrics_summary = self.monitoring_system.get_metrics_summary(self.baseline_window)
        
        for metric_name, metric_data in metrics_summary.get('metrics', {}).items():
            current_value = metric_data.get('avg', 0)
            
            if metric_name in self.performance_baselines:
                baseline = self.performance_baselines[metric_name]
                
                # Update baseline
                baseline.current_value = current_value
                baseline.last_updated = datetime.now()
                baseline.sample_count += 1
                
                # Calculate trend
                baseline.trend = self._calculate_trend(baseline)
                
                # Update database
                self._save_baseline(baseline)
                
            else:
                # Create new baseline
                baseline = PerformanceBaseline(
                    metric_name=metric_name,
                    baseline_value=current_value,
                    target_value=current_value * 1.1,  # 10% improvement target
                    current_value=current_value,
                    trend="stable",
                    last_updated=datetime.now(),
                    sample_count=1
                )
                
                self.performance_baselines[metric_name] = baseline
                self._save_baseline(baseline)
            
            # Track history
            self.baseline_history[metric_name].append({
                'value': current_value,
                'timestamp': datetime.now().isoformat()
            })
    
    def _calculate_trend(self, baseline: PerformanceBaseline) -> str:
        """Calculate performance trend."""
        if baseline.metric_name not in self.baseline_history:
            return "stable"
        
        history = list(self.baseline_history[baseline.metric_name])
        if len(history) < 2:
            return "stable"
        
        # Calculate trend over recent history
        recent_values = [h['value'] for h in history[-10:]]
        if len(recent_values) >= 2:
            slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            if slope > 0.01:  # 1% improvement threshold
                return "improving"
            elif slope < -0.01:  # 1% decline threshold
                return "declining"
        
        return "stable"
    
    def _save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO performance_baselines 
                (metric_name, baseline_value, target_value, current_value, trend, last_updated, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                baseline.metric_name,
                baseline.baseline_value,
                baseline.target_value,
                baseline.current_value,
                baseline.trend,
                baseline.last_updated.isoformat(),
                baseline.sample_count
            ))
            conn.commit()
    
    @handle_errors(operation_name="analyze_performance_trends")
    def _analyze_performance_trends(self):
        """Analyze performance trends and generate insights."""
        for metric_name, baseline in self.performance_baselines.items():
            # Check for significant changes
            if baseline.sample_count > 10:
                change_percentage = ((baseline.current_value - baseline.baseline_value) / baseline.baseline_value) * 100
                
                if abs(change_percentage) > 10:  # 10% change threshold
                    feedback_type = FeedbackType.PERFORMANCE
                    priority = FeedbackPriority.HIGH if abs(change_percentage) > 20 else FeedbackPriority.MEDIUM
                    
                    feedback = FeedbackEntry(
                        id=f"trend_{metric_name}_{int(time.time())}",
                        feedback_type=feedback_type,
                        priority=priority,
                        title=f"Performance trend detected: {metric_name}",
                        description=f"Metric {metric_name} has changed by {change_percentage:.1f}% from baseline",
                        context={
                            'metric_name': metric_name,
                            'baseline_value': baseline.baseline_value,
                            'current_value': baseline.current_value,
                            'change_percentage': change_percentage,
                            'trend': baseline.trend,
                            'sample_count': baseline.sample_count
                        },
                        timestamp=datetime.now(),
                        source="trend_analysis"
                    )
                    
                    self.submit_feedback(feedback)
    
    @handle_errors(operation_name="generate_improvement_recommendations")
    def _generate_improvement_recommendations(self):
        """Generate improvement recommendations based on analysis."""
        recommendations = []
        
        # Analyze performance baselines for improvement opportunities
        for metric_name, baseline in self.performance_baselines.items():
            if baseline.trend == "declining":
                recommendation = self._generate_performance_recommendation(metric_name, baseline)
                if recommendation:
                    recommendations.append(recommendation)
        
        # Analyze feedback patterns
        feedback_recommendations = self._analyze_feedback_patterns()
        recommendations.extend(feedback_recommendations)
        
        # Save recommendations
        for rec in recommendations:
            self._save_recommendation(rec)
            self.improvement_recommendations[rec.id] = rec
    
    def _generate_performance_recommendation(self, metric_name: str, baseline: PerformanceBaseline) -> Optional[ImprovementRecommendation]:
        """Generate performance improvement recommendation."""
        # Define improvement strategies based on metric type
        strategies = {
            'cpu_usage': {
                'title': 'Optimize CPU Usage',
                'description': 'Implement CPU optimization strategies',
                'steps': ['Profile CPU-intensive operations', 'Optimize algorithms', 'Use parallel processing'],
                'benefits': ['Reduced CPU usage', 'Improved response times', 'Better scalability']
            },
            'memory_usage': {
                'title': 'Optimize Memory Usage',
                'description': 'Implement memory optimization strategies',
                'steps': ['Analyze memory leaks', 'Optimize data structures', 'Implement caching'],
                'benefits': ['Reduced memory usage', 'Better stability', 'Improved performance']
            },
            'api_response_time': {
                'title': 'Optimize API Response Times',
                'description': 'Improve API performance',
                'steps': ['Implement caching', 'Optimize queries', 'Use connection pooling'],
                'benefits': ['Faster response times', 'Better user experience', 'Reduced load']
            }
        }
        
        # Find matching strategy
        strategy = None
        for key, strat in strategies.items():
            if key in metric_name.lower():
                strategy = strat
                break
        
        if not strategy:
            return None
        
        # Calculate impact and effort estimates
        impact_estimate = abs((baseline.current_value - baseline.target_value) / baseline.target_value)
        effort_estimate = min(impact_estimate * 2, 1.0)  # Effort roughly 2x impact, capped at 1.0
        priority_score = impact_estimate / effort_estimate if effort_estimate > 0 else 0
        
        return ImprovementRecommendation(
            id=f"perf_{metric_name}_{int(time.time())}",
            category="performance",
            title=strategy['title'],
            description=strategy['description'],
            impact_estimate=impact_estimate,
            effort_estimate=effort_estimate,
            priority_score=priority_score,
            implementation_steps=strategy['steps'],
            expected_benefits=strategy['benefits'],
            metrics_to_track=[metric_name],
            created_at=datetime.now()
        )
    
    def _analyze_feedback_patterns(self) -> List[ImprovementRecommendation]:
        """Analyze feedback patterns to generate recommendations."""
        recommendations = []
        
        # Get recent feedback
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT feedback_type, COUNT(*) as count
                FROM feedback
                WHERE timestamp > ?
                GROUP BY feedback_type
                ORDER BY count DESC
            ''', ((datetime.now() - timedelta(days=7)).isoformat(),))
            
            feedback_counts = cursor.fetchall()
        
        # Generate recommendations based on feedback patterns
        for feedback_type, count in feedback_counts:
            if count >= 5:  # Threshold for pattern detection
                recommendation = self._generate_feedback_recommendation(feedback_type, count)
                if recommendation:
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_feedback_recommendation(self, feedback_type: str, count: int) -> Optional[ImprovementRecommendation]:
        """Generate recommendation based on feedback pattern."""
        recommendations_map = {
            'system_error': {
                'title': 'Improve Error Handling',
                'description': f'Address recurring system errors ({count} reports)',
                'steps': ['Analyze error logs', 'Implement better error handling', 'Add monitoring'],
                'benefits': ['Reduced errors', 'Better reliability', 'Improved user experience']
            },
            'performance': {
                'title': 'Address Performance Issues',
                'description': f'Optimize performance based on feedback ({count} reports)',
                'steps': ['Profile performance', 'Optimize bottlenecks', 'Implement caching'],
                'benefits': ['Better performance', 'Improved user satisfaction', 'Reduced load']
            },
            'user_experience': {
                'title': 'Improve User Experience',
                'description': f'Address user experience issues ({count} reports)',
                'steps': ['Analyze user feedback', 'Improve interface', 'Add user testing'],
                'benefits': ['Better usability', 'Higher satisfaction', 'Reduced support burden']
            }
        }
        
        if feedback_type not in recommendations_map:
            return None
        
        strategy = recommendations_map[feedback_type]
        
        return ImprovementRecommendation(
            id=f"feedback_{feedback_type}_{int(time.time())}",
            category="user_feedback",
            title=strategy['title'],
            description=strategy['description'],
            impact_estimate=min(count / 10, 1.0),  # Scale impact based on feedback count
            effort_estimate=0.5,  # Medium effort
            priority_score=min(count / 10, 1.0) / 0.5,
            implementation_steps=strategy['steps'],
            expected_benefits=strategy['benefits'],
            metrics_to_track=[f"{feedback_type}_feedback_count"],
            created_at=datetime.now()
        )
    
    def _save_recommendation(self, recommendation: ImprovementRecommendation):
        """Save recommendation to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO improvement_recommendations 
                (id, category, title, description, impact_estimate, effort_estimate, 
                 priority_score, implementation_steps, expected_benefits, metrics_to_track, 
                 created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                recommendation.id,
                recommendation.category,
                recommendation.title,
                recommendation.description,
                recommendation.impact_estimate,
                recommendation.effort_estimate,
                recommendation.priority_score,
                json.dumps(recommendation.implementation_steps),
                json.dumps(recommendation.expected_benefits),
                json.dumps(recommendation.metrics_to_track),
                recommendation.created_at.isoformat(),
                recommendation.status
            ))
            conn.commit()
    
    @handle_errors(operation_name="check_for_regressions")
    def _check_for_regressions(self):
        """Check for performance regressions."""
        for metric_name, baseline in self.performance_baselines.items():
            if baseline.trend == "declining":
                # Check if this is a significant regression
                regression_threshold = 0.2  # 20% degradation
                current_degradation = (baseline.baseline_value - baseline.current_value) / baseline.baseline_value
                
                if current_degradation > regression_threshold:
                    feedback = FeedbackEntry(
                        id=f"regression_{metric_name}_{int(time.time())}",
                        feedback_type=FeedbackType.PERFORMANCE,
                        priority=FeedbackPriority.CRITICAL,
                        title=f"Performance regression detected: {metric_name}",
                        description=f"Metric {metric_name} has degraded by {current_degradation*100:.1f}%",
                        context={
                            'metric_name': metric_name,
                            'baseline_value': baseline.baseline_value,
                            'current_value': baseline.current_value,
                            'degradation_percentage': current_degradation * 100,
                            'regression_threshold': regression_threshold * 100
                        },
                        timestamp=datetime.now(),
                        source="regression_detection"
                    )
                    
                    self.submit_feedback(feedback)
    
    def _cleanup_old_data(self):
        """Clean up old data from database."""
        cutoff_date = (datetime.now() - timedelta(days=90)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean up old processed feedback
            conn.execute('''
                DELETE FROM feedback 
                WHERE processed = 1 AND processed_at < ?
            ''', (cutoff_date,))
            
            # Clean up old recommendations
            conn.execute('''
                DELETE FROM improvement_recommendations 
                WHERE status IN ('implemented', 'rejected') AND created_at < ?
            ''', (cutoff_date,))
            
            conn.commit()
    
    def get_improvement_dashboard(self) -> Dict[str, Any]:
        """Get data for improvement dashboard."""
        # Get recent recommendations
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM improvement_recommendations
                WHERE status = 'pending'
                ORDER BY priority_score DESC
                LIMIT 10
            ''')
            
            recommendations = []
            for row in cursor.fetchall():
                recommendations.append({
                    'id': row[0],
                    'category': row[1],
                    'title': row[2],
                    'description': row[3],
                    'impact_estimate': row[4],
                    'effort_estimate': row[5],
                    'priority_score': row[6],
                    'created_at': row[10]
                })
        
        # Get performance trends
        performance_summary = {}
        for metric_name, baseline in self.performance_baselines.items():
            performance_summary[metric_name] = {
                'baseline_value': baseline.baseline_value,
                'current_value': baseline.current_value,
                'trend': baseline.trend,
                'target_value': baseline.target_value,
                'sample_count': baseline.sample_count
            }
        
        return {
            'recommendations': recommendations,
            'performance_summary': performance_summary,
            'system_info': {
                'analysis_interval': self.analysis_interval,
                'baseline_window': self.baseline_window,
                'improvement_threshold': self.improvement_threshold
            }
        }
    
    def add_feedback_handler(self, handler: Callable[[FeedbackEntry], None]):
        """Add a feedback handler."""
        self.feedback_handlers.append(handler)
    
    def approve_recommendation(self, recommendation_id: str):
        """Approve an improvement recommendation."""
        if recommendation_id in self.improvement_recommendations:
            recommendation = self.improvement_recommendations[recommendation_id]
            recommendation.status = "approved"
            self._save_recommendation(recommendation)
            
            self.logger.info(f"Improvement recommendation approved: {recommendation_id}")
    
    def implement_recommendation(self, recommendation_id: str, implementation_notes: str = ""):
        """Mark recommendation as implemented."""
        if recommendation_id in self.improvement_recommendations:
            recommendation = self.improvement_recommendations[recommendation_id]
            recommendation.status = "implemented"
            recommendation.metadata['implementation_notes'] = implementation_notes
            self._save_recommendation(recommendation)
            
            self.logger.info(f"Improvement recommendation implemented: {recommendation_id}")


# Global improvement system instance
_global_improvement_system = None

def get_improvement_system() -> ContinuousImprovementSystem:
    """Get global improvement system instance."""
    global _global_improvement_system
    if _global_improvement_system is None:
        _global_improvement_system = ContinuousImprovementSystem()
    return _global_improvement_system