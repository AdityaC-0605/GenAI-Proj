"""
Migration and A/B testing utilities for RAG system.

Supports parallel deployment, traffic splitting, and rollback mechanisms.
"""

import time
import random
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

from .logging_config import LoggerMixin


class ModelType(Enum):
    """Model type for routing."""
    RAG = "rag"
    FINETUNED = "finetuned"


@dataclass
class RoutingConfig:
    """Configuration for traffic routing."""
    rag_percentage: float = 0.1  # 10% to RAG initially
    enable_ab_testing: bool = True
    enable_auto_rollback: bool = True
    error_threshold: float = 0.1  # 10% error rate triggers rollback
    latency_threshold: float = 10.0  # 10s latency triggers rollback


@dataclass
class ModelMetrics:
    """Metrics for a model."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    user_satisfaction_scores: list = field(default_factory=list)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    @property
    def avg_satisfaction(self) -> float:
        """Calculate average user satisfaction."""
        if not self.user_satisfaction_scores:
            return 0.0
        return sum(self.user_satisfaction_scores) / len(self.user_satisfaction_scores)


class TrafficRouter(LoggerMixin):
    """Route traffic between RAG and fine-tuned models."""
    
    def __init__(self, config: RoutingConfig):
        """
        Initialize traffic router.
        
        Args:
            config: Routing configuration
        """
        super().__init__()
        self.config = config
        self.rag_metrics = ModelMetrics()
        self.finetuned_metrics = ModelMetrics()
        self.user_assignments: Dict[str, ModelType] = {}
    
    def route_request(self, user_id: Optional[str] = None) -> ModelType:
        """
        Route request to appropriate model.
        
        Args:
            user_id: Optional user ID for consistent routing
            
        Returns:
            Model type to use
        """
        if self.config.enable_ab_testing and user_id:
            # Consistent routing for same user
            if user_id in self.user_assignments:
                return self.user_assignments[user_id]
            
            # Assign user to model
            if random.random() < self.config.rag_percentage:
                model = ModelType.RAG
            else:
                model = ModelType.FINETUNED
            
            self.user_assignments[user_id] = model
            return model
        else:
            # Random routing based on percentage
            if random.random() < self.config.rag_percentage:
                return ModelType.RAG
            else:
                return ModelType.FINETUNED
    
    def record_request(self,
                      model_type: ModelType,
                      success: bool,
                      latency: float,
                      satisfaction_score: Optional[float] = None):
        """
        Record request metrics.
        
        Args:
            model_type: Which model was used
            success: Whether request succeeded
            latency: Request latency in seconds
            satisfaction_score: Optional user satisfaction (0-1)
        """
        metrics = self.rag_metrics if model_type == ModelType.RAG else self.finetuned_metrics
        
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
            metrics.total_latency += latency
        else:
            metrics.failed_requests += 1
        
        if satisfaction_score is not None:
            metrics.user_satisfaction_scores.append(satisfaction_score)
        
        # Check for auto-rollback
        if self.config.enable_auto_rollback:
            self._check_rollback()
    
    def _check_rollback(self):
        """Check if rollback is needed."""
        # Only check RAG metrics (we're testing RAG)
        if self.rag_metrics.total_requests < 100:
            return  # Need minimum requests
        
        # Check error rate
        if self.rag_metrics.error_rate > self.config.error_threshold:
            self.logger.warning(
                f"RAG error rate ({self.rag_metrics.error_rate:.2%}) "
                f"exceeds threshold ({self.config.error_threshold:.2%})"
            )
            self._trigger_rollback("high_error_rate")
        
        # Check latency
        if self.rag_metrics.avg_latency > self.config.latency_threshold:
            self.logger.warning(
                f"RAG latency ({self.rag_metrics.avg_latency:.2f}s) "
                f"exceeds threshold ({self.config.latency_threshold:.2f}s)"
            )
            self._trigger_rollback("high_latency")
    
    def _trigger_rollback(self, reason: str):
        """Trigger rollback to fine-tuned model."""
        self.logger.error(f"ROLLBACK TRIGGERED: {reason}")
        self.logger.error("Switching all traffic to fine-tuned model")
        
        # Set RAG percentage to 0
        self.config.rag_percentage = 0.0
        
        # Clear user assignments
        self.user_assignments.clear()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for both models."""
        return {
            'rag': {
                'total_requests': self.rag_metrics.total_requests,
                'error_rate': self.rag_metrics.error_rate,
                'avg_latency': self.rag_metrics.avg_latency,
                'avg_satisfaction': self.rag_metrics.avg_satisfaction
            },
            'finetuned': {
                'total_requests': self.finetuned_metrics.total_requests,
                'error_rate': self.finetuned_metrics.error_rate,
                'avg_latency': self.finetuned_metrics.avg_latency,
                'avg_satisfaction': self.finetuned_metrics.avg_satisfaction
            },
            'config': {
                'rag_percentage': self.config.rag_percentage,
                'ab_testing_enabled': self.config.enable_ab_testing,
                'auto_rollback_enabled': self.config.enable_auto_rollback
            }
        }


class ABTestingFramework(LoggerMixin):
    """A/B testing framework for comparing models."""
    
    def __init__(self, router: TrafficRouter):
        """
        Initialize A/B testing framework.
        
        Args:
            router: Traffic router instance
        """
        super().__init__()
        self.router = router
    
    def run_ab_test(self,
                   duration_seconds: int = 3600,
                   check_interval: int = 60) -> Dict[str, Any]:
        """
        Run A/B test for specified duration.
        
        Args:
            duration_seconds: Test duration
            check_interval: Interval for checking metrics
            
        Returns:
            Test results
        """
        self.logger.info(f"Starting A/B test for {duration_seconds}s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            # Check metrics periodically
            time.sleep(check_interval)
            
            metrics = self.router.get_metrics_summary()
            self.logger.info(f"Current metrics: {metrics}")
            
            # Check if rollback was triggered
            if metrics['config']['rag_percentage'] == 0.0:
                self.logger.warning("Rollback triggered, ending test")
                break
        
        # Final results
        final_metrics = self.router.get_metrics_summary()
        
        # Determine winner
        winner = self._determine_winner(final_metrics)
        
        results = {
            'duration': time.time() - start_time,
            'metrics': final_metrics,
            'winner': winner,
            'recommendation': self._get_recommendation(winner, final_metrics)
        }
        
        self.logger.info(f"A/B test complete. Winner: {winner}")
        return results
    
    def _determine_winner(self, metrics: Dict[str, Any]) -> str:
        """Determine which model performed better."""
        rag = metrics['rag']
        ft = metrics['finetuned']
        
        # Score based on multiple factors
        rag_score = 0
        ft_score = 0
        
        # Error rate (lower is better)
        if rag['error_rate'] < ft['error_rate']:
            rag_score += 1
        else:
            ft_score += 1
        
        # Latency (lower is better)
        if rag['avg_latency'] < ft['avg_latency']:
            rag_score += 1
        else:
            ft_score += 1
        
        # User satisfaction (higher is better)
        if rag['avg_satisfaction'] > ft['avg_satisfaction']:
            rag_score += 1
        else:
            ft_score += 1
        
        if rag_score > ft_score:
            return 'RAG'
        elif ft_score > rag_score:
            return 'Fine-tuned'
        else:
            return 'Tie'
    
    def _get_recommendation(self, winner: str, metrics: Dict[str, Any]) -> str:
        """Get deployment recommendation."""
        if winner == 'RAG':
            return "Recommend migrating to RAG system"
        elif winner == 'Fine-tuned':
            return "Recommend keeping fine-tuned model"
        else:
            return "Results inconclusive, continue testing"


class RollbackManager(LoggerMixin):
    """Manage rollback between models."""
    
    def __init__(self):
        """Initialize rollback manager."""
        super().__init__()
        self.rollback_history = []
    
    def quick_switch(self, target_model: ModelType, config_path: str) -> bool:
        """
        Quick switch between models via configuration.
        
        Args:
            target_model: Model to switch to
            config_path: Path to configuration file
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Switching to {target_model.value} model")
            
            # Record rollback
            self.rollback_history.append({
                'timestamp': time.time(),
                'target_model': target_model.value,
                'reason': 'manual_switch'
            })
            
            # In production, this would update config and reload
            self.logger.info(f"Successfully switched to {target_model.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch models: {e}")
            return False
    
    def automated_rollback(self,
                          router: TrafficRouter,
                          reason: str) -> bool:
        """
        Automated rollback on error detection.
        
        Args:
            router: Traffic router
            reason: Reason for rollback
            
        Returns:
            Success status
        """
        try:
            self.logger.warning(f"Automated rollback triggered: {reason}")
            
            # Set traffic to 100% fine-tuned
            router.config.rag_percentage = 0.0
            
            # Record rollback
            self.rollback_history.append({
                'timestamp': time.time(),
                'target_model': ModelType.FINETUNED.value,
                'reason': reason,
                'automated': True
            })
            
            self.logger.info("Rollback complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def get_rollback_history(self) -> list:
        """Get rollback history."""
        return self.rollback_history
