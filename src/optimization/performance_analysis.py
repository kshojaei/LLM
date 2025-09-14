"""
Performance Analysis Module for RAG System
This module provides comprehensive performance analysis and optimization tools.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import time
import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import psutil
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance analysis."""
    enable_profiling: bool = True
    enable_memory_monitoring: bool = True
    enable_latency_tracking: bool = True
    enable_cost_tracking: bool = True
    sample_rate: float = 1.0  # Fraction of requests to profile
    history_size: int = 1000

class PerformanceProfiler:
    """
    Comprehensive performance profiler for RAG systems.
    
    This class tracks various performance metrics including latency,
    memory usage, cost, and throughput.
    """
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.metrics_history = deque(maxlen=self.config.history_size)
        self.current_metrics = {}
        self.start_time = None
        self.memory_monitor = None
        
        if self.config.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor()
    
    @contextmanager
    def profile_request(self, request_id: str, request_type: str = "query"):
        """Context manager for profiling individual requests."""
        if not self.config.enable_profiling:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Record metrics
            metrics = {
                'request_id': request_id,
                'request_type': request_type,
                'timestamp': start_time,
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'peak_memory': self._get_peak_memory(),
                'cpu_usage': self._get_cpu_usage()
            }
            
            self._record_metrics(metrics)
    
    def profile_component(self, component_name: str):
        """Decorator for profiling individual components."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.config.enable_profiling:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Record component metrics
                    metrics = {
                        'component': component_name,
                        'timestamp': start_time,
                        'duration': end_time - start_time,
                        'memory_delta': end_memory - start_memory,
                        'cpu_usage': self._get_cpu_usage()
                    }
                    
                    self._record_metrics(metrics)
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.config.enable_memory_monitoring:
            return 0.0
        
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.config.enable_memory_monitoring:
            return 0.0
        
        process = psutil.Process()
        return process.memory_info().peak_wss / 1024 / 1024  # Convert to MB
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent()
    
    def _record_metrics(self, metrics: Dict[str, Any]):
        """Record metrics to history."""
        self.metrics_history.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(list(self.metrics_history))
        
        summary = {
            'total_requests': len(df),
            'avg_duration': df['duration'].mean() if 'duration' in df.columns else 0,
            'p95_duration': df['duration'].quantile(0.95) if 'duration' in df.columns else 0,
            'p99_duration': df['duration'].quantile(0.99) if 'duration' in df.columns else 0,
            'avg_memory_delta': df['memory_delta'].mean() if 'memory_delta' in df.columns else 0,
            'peak_memory': df['peak_memory'].max() if 'peak_memory' in df.columns else 0,
            'avg_cpu_usage': df['cpu_usage'].mean() if 'cpu_usage' in df.columns else 0
        }
        
        return summary
    
    def get_component_breakdown(self) -> Dict[str, Any]:
        """Get performance breakdown by component."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(list(self.metrics_history))
        
        if 'component' not in df.columns:
            return {}
        
        component_stats = {}
        for component in df['component'].unique():
            component_data = df[df['component'] == component]
            component_stats[component] = {
                'count': len(component_data),
                'avg_duration': component_data['duration'].mean(),
                'total_duration': component_data['duration'].sum(),
                'avg_memory_delta': component_data['memory_delta'].mean()
            }
        
        return component_stats

class MemoryMonitor:
    """Memory usage monitoring."""
    
    def __init__(self):
        self.memory_history = deque(maxlen=1000)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Memory monitoring loop."""
        while self.monitoring:
            memory_usage = self._get_memory_usage()
            self.memory_history.append({
                'timestamp': time.time(),
                'memory_mb': memory_usage
            })
            time.sleep(interval)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory_history:
            return {}
        
        memory_values = [entry['memory_mb'] for entry in self.memory_history]
        
        return {
            'current_memory': memory_values[-1] if memory_values else 0,
            'avg_memory': np.mean(memory_values),
            'peak_memory': max(memory_values),
            'min_memory': min(memory_values),
            'memory_trend': self._calculate_trend(memory_values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate memory trend."""
        if len(values) < 2:
            return 'stable'
        
        # Simple trend calculation
        recent_avg = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        older_avg = np.mean(values[:10]) if len(values) >= 10 else np.mean(values)
        
        if recent_avg > older_avg * 1.1:
            return 'increasing'
        elif recent_avg < older_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'

class CostAnalyzer:
    """Cost analysis for RAG systems."""
    
    def __init__(self):
        self.cost_history = []
        self.cost_models = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},  # per 1K tokens
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'llama-3-8b': {'input': 0.0, 'output': 0.0},  # Free if self-hosted
        }
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        if model not in self.cost_models:
            return 0.0
        
        cost_per_1k = self.cost_models[model]
        input_cost = (input_tokens / 1000) * cost_per_1k['input']
        output_cost = (output_tokens / 1000) * cost_per_1k['output']
        
        return input_cost + output_cost
    
    def record_request_cost(self, model: str, input_tokens: int, output_tokens: int, 
                          request_id: str = None):
        """Record cost for a request."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        self.cost_history.append({
            'timestamp': time.time(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'request_id': request_id
        })
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics."""
        if not self.cost_history:
            return {}
        
        df = pd.DataFrame(self.cost_history)
        
        summary = {
            'total_cost': df['cost'].sum(),
            'avg_cost_per_request': df['cost'].mean(),
            'total_requests': len(df),
            'total_input_tokens': df['input_tokens'].sum(),
            'total_output_tokens': df['output_tokens'].sum(),
            'cost_by_model': df.groupby('model')['cost'].sum().to_dict()
        }
        
        return summary

class LatencyAnalyzer:
    """Latency analysis and optimization."""
    
    def __init__(self):
        self.latency_history = []
        self.component_latencies = defaultdict(list)
    
    def record_latency(self, component: str, duration: float, metadata: Dict[str, Any] = None):
        """Record latency for a component."""
        self.latency_history.append({
            'timestamp': time.time(),
            'component': component,
            'duration': duration,
            'metadata': metadata or {}
        })
        
        self.component_latencies[component].append(duration)
    
    def analyze_latency_bottlenecks(self) -> Dict[str, Any]:
        """Analyze latency bottlenecks."""
        if not self.latency_history:
            return {}
        
        df = pd.DataFrame(self.latency_history)
        
        # Component analysis
        component_stats = {}
        for component in df['component'].unique():
            component_data = df[df['component'] == component]
            component_stats[component] = {
                'avg_duration': component_data['duration'].mean(),
                'p95_duration': component_data['duration'].quantile(0.95),
                'p99_duration': component_data['duration'].quantile(0.99),
                'total_duration': component_data['duration'].sum(),
                'percentage_of_total': component_data['duration'].sum() / df['duration'].sum() * 100
            }
        
        # Identify bottlenecks
        bottlenecks = []
        for component, stats in component_stats.items():
            if stats['percentage_of_total'] > 20:  # More than 20% of total time
                bottlenecks.append({
                    'component': component,
                    'percentage': stats['percentage_of_total'],
                    'avg_duration': stats['avg_duration']
                })
        
        return {
            'component_stats': component_stats,
            'bottlenecks': sorted(bottlenecks, key=lambda x: x['percentage'], reverse=True),
            'total_avg_duration': df['duration'].mean()
        }
    
    def get_latency_trends(self) -> Dict[str, Any]:
        """Get latency trends over time."""
        if not self.latency_history:
            return {}
        
        df = pd.DataFrame(self.latency_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        
        # Hourly trends
        hourly_stats = df.groupby('hour')['duration'].agg(['mean', 'count']).to_dict()
        
        return {
            'hourly_avg_duration': hourly_stats['mean'],
            'hourly_request_count': hourly_stats['count']
        }

class PerformanceOptimizer:
    """Performance optimization recommendations."""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Get performance summary
        summary = self.profiler.get_performance_summary()
        component_breakdown = self.profiler.get_component_breakdown()
        
        # Latency recommendations
        if summary.get('p95_duration', 0) > 5.0:  # 5 seconds
            recommendations.append({
                'type': 'latency',
                'priority': 'high',
                'title': 'High latency detected',
                'description': f"P95 latency is {summary['p95_duration']:.2f}s, consider optimization",
                'suggestions': [
                    'Implement caching for frequent queries',
                    'Use smaller embedding models for faster retrieval',
                    'Optimize database queries',
                    'Consider async processing'
                ]
            })
        
        # Memory recommendations
        if summary.get('peak_memory', 0) > 1000:  # 1GB
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'title': 'High memory usage',
                'description': f"Peak memory usage is {summary['peak_memory']:.2f}MB",
                'suggestions': [
                    'Implement model quantization',
                    'Use gradient checkpointing',
                    'Clear unused variables',
                    'Consider model sharding'
                ]
            })
        
        # Component-specific recommendations
        for component, stats in component_breakdown.items():
            if stats['avg_duration'] > 2.0:  # 2 seconds
                recommendations.append({
                    'type': 'component',
                    'priority': 'medium',
                    'title': f'Slow component: {component}',
                    'description': f"Component {component} takes {stats['avg_duration']:.2f}s on average",
                    'suggestions': [
                        f'Profile {component} for specific bottlenecks',
                        'Consider parallel processing',
                        'Optimize data structures',
                        'Use more efficient algorithms'
                    ]
                })
        
        return recommendations
    
    def create_performance_report(self, output_path: Path):
        """Create a comprehensive performance report."""
        summary = self.profiler.get_performance_summary()
        component_breakdown = self.profiler.get_component_breakdown()
        recommendations = self.generate_recommendations()
        
        report = {
            'timestamp': time.time(),
            'summary': summary,
            'component_breakdown': component_breakdown,
            'recommendations': recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create profiler
    config = PerformanceConfig(enable_profiling=True, enable_memory_monitoring=True)
    profiler = PerformanceProfiler(config)
    
    # Example of profiling a request
    with profiler.profile_request("req_001", "query"):
        time.sleep(0.1)  # Simulate work
    
    # Example of profiling a component
    @profiler.profile_component("embedding_generation")
    def generate_embeddings(texts):
        time.sleep(0.05)  # Simulate embedding generation
        return [f"embedding_{i}" for i in range(len(texts))]
    
    # Test component
    embeddings = generate_embeddings(["text1", "text2", "text3"])
    
    # Get performance summary
    summary = profiler.get_performance_summary()
    print(f"Performance summary: {summary}")
    
    # Get component breakdown
    breakdown = profiler.get_component_breakdown()
    print(f"Component breakdown: {breakdown}")
    
    # Generate recommendations
    optimizer = PerformanceOptimizer(profiler)
    recommendations = optimizer.generate_recommendations()
    print(f"Recommendations: {recommendations}")
    
    # Test cost analyzer
    cost_analyzer = CostAnalyzer()
    cost_analyzer.record_request_cost("gpt-3.5-turbo", 100, 50, "req_001")
    cost_summary = cost_analyzer.get_cost_summary()
    print(f"Cost summary: {cost_summary}")
    
    # Test latency analyzer
    latency_analyzer = LatencyAnalyzer()
    latency_analyzer.record_latency("retrieval", 0.5)
    latency_analyzer.record_latency("generation", 2.0)
    bottlenecks = latency_analyzer.analyze_latency_bottlenecks()
    print(f"Latency bottlenecks: {bottlenecks}")
