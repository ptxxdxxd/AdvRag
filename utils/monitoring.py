"""
Performance monitoring and analytics for RAG system
Tracks metrics, generates insights, and provides optimization recommendations
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: datetime
    active_sessions: int
    total_queries: int
    avg_response_time: float
    avg_cost_per_query: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    active_model: str


@dataclass
class QueryAnalytics:
    """Analytics for individual queries"""
    query: str
    timestamp: datetime
    response_time: float
    cost: float
    retrieval_score: float
    provider: str
    model_used: str
    tokens_used: int
    error_occurred: bool
    user_feedback: Optional[int] = None  # 1-5 rating


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, storage_path: str = "./data/performance"):
        """Initialize performance monitor"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.query_analytics: List[QueryAnalytics] = []
        self.system_snapshots: List[PerformanceSnapshot] = []
        
        # Real-time metrics
        self.current_metrics = {
            "total_queries": 0,
            "active_sessions": 0,
            "error_count": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "last_update": datetime.now()
        }
        
        # Performance thresholds
        self.thresholds = {
            "response_time": 5.0,  # seconds
            "cost_per_query": 0.01,  # dollars
            "error_rate": 0.05,  # 5%
            "retrieval_score": 0.7  # minimum acceptable score
        }
        
        logger.info(f"Performance monitor initialized with storage at {storage_path}")
    
    def log_query(
        self,
        query: str,
        response_time: float,
        cost: float,
        retrieval_score: float,
        provider: str = "unknown",
        model_used: str = "unknown",
        tokens_used: int = 0,
        error_occurred: bool = False
    ):
        """Log a query for analytics"""
        
        analytics = QueryAnalytics(
            query=query,
            timestamp=datetime.now(),
            response_time=response_time,
            cost=cost,
            retrieval_score=retrieval_score,
            provider=provider,
            model_used=model_used,
            tokens_used=tokens_used,
            error_occurred=error_occurred
        )
        
        self.query_analytics.append(analytics)
        self._update_current_metrics(analytics)
        
        # Save to storage periodically
        if len(self.query_analytics) % 100 == 0:
            self._save_analytics()
    
    def _update_current_metrics(self, analytics: QueryAnalytics):
        """Update current performance metrics"""
        
        self.current_metrics["total_queries"] += 1
        self.current_metrics["total_cost"] += analytics.cost
        self.current_metrics["last_update"] = datetime.now()
        
        if analytics.error_occurred:
            self.current_metrics["error_count"] += 1
        
        # Calculate averages
        total_queries = self.current_metrics["total_queries"]
        self.current_metrics["avg_response_time"] = (
            (self.current_metrics["avg_response_time"] * (total_queries - 1) + 
             analytics.response_time) / total_queries
        )
        self.current_metrics["avg_cost_per_query"] = (
            self.current_metrics["total_cost"] / total_queries
        )
    
    def add_system_snapshot(self):
        """Add a system performance snapshot"""
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            active_sessions=self.current_metrics["active_sessions"],
            total_queries=self.current_metrics["total_queries"],
            avg_response_time=self.current_metrics["avg_response_time"],
            avg_cost_per_query=self.current_metrics["avg_cost_per_query"],
            error_rate=self._calculate_error_rate(),
            cpu_usage=self._get_cpu_usage(),
            memory_usage=self._get_memory_usage(),
            active_model="current_active_model"
        )
        
        self.system_snapshots.append(snapshot)
        
        # Keep only recent snapshots
        if len(self.system_snapshots) > 1000:
            self.system_snapshots = self.system_snapshots[-1000:]
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if self.current_metrics["total_queries"] == 0:
            return 0.0
        return self.current_metrics["error_count"] / self.current_metrics["total_queries"]
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (placeholder implementation)"""
        # In a real system, this would use psutil or similar
        return np.random.uniform(10, 80)  # Mock data
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (placeholder implementation)"""
        # In a real system, this would use psutil or similar
        return np.random.uniform(100, 800)  # Mock data in MB
    
    def get_recent_analytics(self, hours: int = 24) -> List[QueryAnalytics]:
        """Get analytics from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [qa for qa in self.query_analytics if qa.timestamp > cutoff_time]
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        recent_analytics = self.get_recent_analytics(hours)
        
        if not recent_analytics:
            return {"error": "No data available for the specified time period"}
        
        df = pd.DataFrame([asdict(qa) for qa in recent_analytics])
        
        report = {
            "time_period": f"Last {hours} hours",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_queries": len(recent_analytics),
                "unique_queries": df['query'].nunique(),
                "avg_response_time": df['response_time'].mean(),
                "median_response_time": df['response_time'].median(),
                "avg_cost_per_query": df['cost'].mean(),
                "total_cost": df['cost'].sum(),
                "error_rate": df['error_occurred'].mean(),
                "avg_retrieval_score": df['retrieval_score'].mean()
            },
            "performance_metrics": {
                "response_time_p50": float(df['response_time'].quantile(0.5)),
                "response_time_p95": float(df['response_time'].quantile(0.95)),
                "response_time_p99": float(df['response_time'].quantile(0.99)),
                "cost_per_query_p50": float(df['cost'].quantile(0.5)),
                "cost_per_query_p95": float(df['cost'].quantile(0.95))
            },
            "provider_breakdown": self._analyze_provider_performance(df),
            "model_breakdown": self._analyze_model_performance(df),
            "error_analysis": self._analyze_errors(df),
            "quality_trends": self._analyze_quality_trends(df),
            "recommendations": self._generate_recommendations(df)
        }
        
        return report
    
    def _analyze_provider_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by provider"""
        provider_stats = df.groupby('provider').agg({
            'response_time': ['mean', 'count'],
            'cost': ['mean', 'sum'],
            'retrieval_score': 'mean',
            'error_occurred': 'mean'
        }).round(4)
        
        return {
            provider: {
                'query_count': int(stats[('response_time', 'count')]),
                'avg_response_time': float(stats[('response_time', 'mean')]),
                'avg_cost': float(stats[('cost', 'mean')]),
                'total_cost': float(stats[('cost', 'sum')]),
                'avg_retrieval_score': float(stats[('retrieval_score', 'mean')]),
                'error_rate': float(stats[('error_occurred', 'mean')])
            }
            for provider, stats in provider_stats.iterrows()
        }
    
    def _analyze_model_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by model"""
        model_stats = df.groupby('model_used').agg({
            'response_time': ['mean', 'count'],
            'cost': 'mean',
            'retrieval_score': 'mean'
        }).round(4)
        
        return {
            model: {
                'query_count': int(stats[('response_time', 'count')]),
                'avg_response_time': float(stats[('response_time', 'mean')]),
                'avg_cost': float(stats[('cost', 'mean')]),
                'avg_retrieval_score': float(stats[('retrieval_score', 'mean')])
            }
            for model, stats in model_stats.iterrows()
        }
    
    def _analyze_errors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_queries = df[df['error_occurred']]
        
        if len(error_queries) == 0:
            return {"error_rate": 0.0, "common_issues": []}
        
        return {
            "error_rate": float(error_queries.shape[0] / df.shape[0]),
            "error_count": int(error_queries.shape[0]),
            "error_queries": error_queries['query'].tolist()[:10]  # First 10 error queries
        }
    
    def _analyze_quality_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if len(df) < 10:
            return {"trend": "insufficient_data"}
        
        # Calculate trends
        df_sorted = df.sort_values('timestamp')
        
        # Retrieval score trend
        retrieval_trend = np.polyfit(range(len(df_sorted)), df_sorted['retrieval_score'], 1)[0]
        
        # Response time trend  
        time_trend = np.polyfit(range(len(df_sorted)), df_sorted['response_time'], 1)[0]
        
        return {
            "retrieval_score_trend": "improving" if retrieval_trend > 0.001 else "declining" if retrieval_trend < -0.001 else "stable",
            "retrieval_trend_value": float(retrieval_trend),
            "response_time_trend": "improving" if time_trend < -0.01 else "declining" if time_trend > 0.01 else "stable",
            "response_time_trend_value": float(time_trend)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Response time recommendations
        if df['response_time'].mean() > self.thresholds["response_time"]:
            recommendations.append(
                f"Consider optimizing for response time. Current average: {df['response_time'].mean():.2f}s"
            )
        
        # Cost recommendations
        if df['cost'].mean() > self.thresholds["cost_per_query"]:
            recommendations.append(
                f"High cost per query detected. Current average: ${df['cost'].mean():.4f}"
            )
        
        # Error rate recommendations
        error_rate = df['error_occurred'].mean()
        if error_rate > self.thresholds["error_rate"]:
            recommendations.append(
                f"High error rate: {error_rate:.1%}. Investigate failure patterns."
            )
        
        # Quality recommendations
        if df['retrieval_score'].mean() < self.thresholds["retrieval_score"]:
            recommendations.append(
                f"Low retrieval quality. Consider fine-tuning or improving retrieval strategies."
            )
        
        # Provider recommendations
        provider_performance = df.groupby('provider')['response_time'].mean()
        if len(provider_performance) > 1:
            best_provider = provider_performance.idxmin()
            recommendations.append(
                f"Provider '{best_provider}' has the best response time. Consider using it more frequently."
            )
        
        return recommendations
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        return {
            **self.current_metrics,
            "error_rate": self._calculate_error_rate(),
            "threshold_breaches": self._check_threshold_breaches(),
            "system_health": self._assess_system_health()
        }
    
    def _check_threshold_breaches(self) -> Dict[str, bool]:
        """Check if any thresholds are being breached"""
        current_metrics = self.current_metrics
        
        return {
            "response_time_high": current_metrics["avg_response_time"] > self.thresholds["response_time"],
            "cost_high": current_metrics["avg_cost_per_query"] > self.thresholds["cost_per_query"],
            "error_rate_high": self._calculate_error_rate() > self.thresholds["error_rate"],
            "retrieval_score_low": self._get_avg_retrieval_score() < self.thresholds["retrieval_score"]
        }
    
    def _get_avg_retrieval_score(self) -> float:
        """Get average retrieval score from recent queries"""
        recent_analytics = self.get_recent_analytics(hours=1)
        if recent_analytics:
            return np.mean([qa.retrieval_score for qa in recent_analytics])
        return 0.0
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        breaches = self._check_threshold_breaches()
        
        if sum(breaches.values()) == 0:
            status = "healthy"
        elif sum(breaches.values()) <= 2:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "breach_count": sum(breaches.values()),
            "issues": [key for key, value in breaches.items() if value]
        }
    
    def _save_analytics(self):
        """Save analytics to disk"""
        try:
            analytics_file = self.storage_path / "query_analytics.json"
            analytics_data = [asdict(qa) for qa in self.query_analytics]
            
            # Convert datetime objects to strings for JSON serialization
            for item in analytics_data:
                item['timestamp'] = item['timestamp'].isoformat()
            
            with open(analytics_file, 'w') as f:
                json.dump(analytics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save analytics: {str(e)}")
    
    def load_analytics(self):
        """Load analytics from disk"""
        try:
            analytics_file = self.storage_path / "query_analytics.json"
            
            if analytics_file.exists():
                with open(analytics_file, 'r') as f:
                    analytics_data = json.load(f)
                
                # Convert back to QueryAnalytics objects
                self.query_analytics = []
                for item in analytics_data:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    self.query_analytics.append(QueryAnalytics(**item))
                
                logger.info(f"Loaded {len(self.query_analytics)} analytics records")
                
        except Exception as e:
            logger.error(f"Failed to load analytics: {str(e)}")
    
    def export_performance_data(self, hours: int = 24, format: str = "json") -> str:
        """Export performance data for external analysis"""
        
        report = self.generate_performance_report(hours)
        
        if format.lower() == "json":
            return json.dumps(report, indent=2)
        elif format.lower() == "csv":
            # Export as CSV for easier analysis
            recent_analytics = self.get_recent_analytics(hours)
            if recent_analytics:
                df = pd.DataFrame([asdict(qa) for qa in recent_analytics])
                return df.to_csv(index=False)
        
        return json.dumps(report, indent=2)


class OptimizationEngine:
    """Engine for suggesting system optimizations based on analytics"""
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize with performance monitor"""
        self.monitor = monitor
    
    def analyze_optimization_opportunities(self) -> Dict[str, Any]:
        """Analyze system for optimization opportunities"""
        
        recent_analytics = self.monitor.get_recent_analytics(hours=24)
        
        if not recent_analytics:
            return {"status": "no_data"}
        
        df = pd.DataFrame([asdict(qa) for qa in recent_analytics])
        
        opportunities = {
            "cost_optimization": self._analyze_cost_optimization(df),
            "performance_optimization": self._analyze_performance_optimization(df),
            "quality_optimization": self._analyze_quality_optimization(df),
            "provider_optimization": self._analyze_provider_optimization(df)
        }
        
        return opportunities
    
    def _analyze_cost_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cost optimization opportunities"""
        
        # Identify expensive queries
        expensive_queries = df[df['cost'] > df['cost'].quantile(0.9)]
        
        # Identify most costly providers
        provider_costs = df.groupby('provider')['cost'].mean().sort_values(ascending=False)
        
        return {
            "potential_savings": float(expensive_queries['cost'].sum() * 0.3),  # 30% potential savings
            "expensive_queries_count": len(expensive_queries),
            "most_expensive_provider": provider_costs.index[0],
            "cost_reduction_strategies": [
                "Implement response caching for similar queries",
                "Switch to more cost-effective models for non-critical queries",
                "Optimize prompt length to reduce token costs"
            ]
        }
    
    def _analyze_performance_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance optimization opportunities"""
        
        slow_queries = df[df['response_time'] > df['response_time'].quantile(0.9)]
        
        return {
            "slow_queries_count": len(slow_queries),
            "avg_slow_query_time": float(slow_queries['response_time'].mean()),
            "performance_strategies": [
                "Implement parallel processing for independent operations",
                "Cache frequently accessed documents",
                "Optimize embedding retrieval with approximate nearest neighbors"
            ]
        }
    
    def _analyze_quality_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality optimization opportunities"""
        
        low_quality_queries = df[df['retrieval_score'] < 0.7]
        
        return {
            "low_quality_count": len(low_quality_queries),
            "quality_strategies": [
                "Fine-tune embeddings on domain-specific data",
                "Implement query expansion techniques",
                "Improve document chunking strategies"
            ]
        }
    
    def _analyze_provider_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze provider optimization opportunities"""
        
        provider_stats = df.groupby('provider').agg({
            'response_time': 'mean',
            'cost': 'mean',
            'retrieval_score': 'mean'
        })
        
        # Find best provider by different metrics
        fastest_provider = provider_stats['response_time'].idxmin()
        cheapest_provider = provider_stats['cost'].idxmin()
        best_quality_provider = provider_stats['retrieval_score'].idxmax()
        
        return {
            "fastest_provider": fastest_provider,
            "cheapest_provider": cheapest_provider,
            "best_quality_provider": best_quality_provider,
            "recommendations": [
                f"Use {fastest_provider} for time-critical queries",
                f"Use {cheapest_provider} for routine queries",
                f"Use {best_quality_provider} for high-quality requirements"
            ]
        }