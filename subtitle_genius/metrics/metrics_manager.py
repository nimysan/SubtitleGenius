"""
Metrics Manager for SubtitleGenius
使用prometheus-client管理和导出系统指标
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, push_to_gateway

logger = logging.getLogger("subtitle_genius.metrics.manager")

class MetricsManager:
    """
    Prometheus指标管理器
    
    负责初始化、收集和导出系统指标到Prometheus Pushgateway
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, pushgateway_url: Optional[str] = None) -> 'MetricsManager':
        """
        获取MetricsManager单例实例
        
        Args:
            pushgateway_url: Prometheus Pushgateway的URL
            
        Returns:
            MetricsManager实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(pushgateway_url)
        return cls._instance
    
    def __init__(self, pushgateway_url: Optional[str] = None):
        """
        初始化指标管理器
        
        Args:
            pushgateway_url: Prometheus Pushgateway的URL，如果为None则使用环境变量
        """
        self.registry = CollectorRegistry()
        
        # 从环境变量或参数获取Pushgateway URL
        self.pushgateway_url = pushgateway_url or os.environ.get(
            'PROMETHEUS_PUSHGATEWAY_URL', 'localhost:9529'
        )
        
        # 应用标识，用于在Pushgateway中区分不同应用
        self.app_name = os.environ.get('APP_NAME', 'subtitle_genius')
        
        # 存储所有注册的指标
        self.metrics: Dict[str, Any] = {}
        
        # 自动推送线程
        self.push_interval = int(os.environ.get('METRICS_PUSH_INTERVAL_SECONDS', '15'))
        self.auto_push_enabled = os.environ.get('METRICS_AUTO_PUSH', 'true').lower() == 'true'
        self._push_thread = None
        
        # 系统指标
        self._init_system_metrics()
        
        logger.info(f"MetricsManager initialized with pushgateway: {self.pushgateway_url}")
        
        # 如果启用了自动推送，启动推送线程
        if self.auto_push_enabled:
            self.start_auto_push()
    
    def _init_system_metrics(self):
        """初始化系统级指标"""
        # 系统健康指标
        self.metrics['system_up'] = Gauge(
            'subtitle_genius_system_up', 
            'System uptime in seconds',
            registry=self.registry
        )
        
        # 系统启动时间
        self.system_start_time = time.time()
        
        # 更新系统启动时间
        self.metrics['system_up'].set_function(lambda: time.time() - self.system_start_time)
    
    def register_counter(self, name: str, description: str, labelnames: list = None) -> Counter:
        """
        注册一个Counter指标
        
        Args:
            name: 指标名称
            description: 指标描述
            labelnames: 标签名称列表
            
        Returns:
            创建的Counter实例
        """
        if name in self.metrics:
            return self.metrics[name]
        
        counter = Counter(
            name, 
            description, 
            labelnames or [], 
            registry=self.registry
        )
        self.metrics[name] = counter
        return counter
    
    def register_gauge(self, name: str, description: str, labelnames: list = None) -> Gauge:
        """
        注册一个Gauge指标
        
        Args:
            name: 指标名称
            description: 指标描述
            labelnames: 标签名称列表
            
        Returns:
            创建的Gauge实例
        """
        if name in self.metrics:
            return self.metrics[name]
        
        gauge = Gauge(
            name, 
            description, 
            labelnames or [], 
            registry=self.registry
        )
        self.metrics[name] = gauge
        return gauge
    
    def register_histogram(
        self, 
        name: str, 
        description: str, 
        labelnames: list = None,
        buckets: list = None
    ) -> Histogram:
        """
        注册一个Histogram指标
        
        Args:
            name: 指标名称
            description: 指标描述
            labelnames: 标签名称列表
            buckets: 直方图桶配置
            
        Returns:
            创建的Histogram实例
        """
        if name in self.metrics:
            return self.metrics[name]
        
        histogram = Histogram(
            name, 
            description, 
            labelnames or [], 
            buckets=buckets,
            registry=self.registry
        )
        self.metrics[name] = histogram
        return histogram
    
    def register_summary(
        self, 
        name: str, 
        description: str, 
        labelnames: list = None
    ) -> Summary:
        """
        注册一个Summary指标
        
        Args:
            name: 指标名称
            description: 指标描述
            labelnames: 标签名称列表
            
        Returns:
            创建的Summary实例
        """
        if name in self.metrics:
            return self.metrics[name]
        
        summary = Summary(
            name, 
            description, 
            labelnames or [], 
            registry=self.registry
        )
        self.metrics[name] = summary
        return summary
    
    def push_metrics(self, job: str = None):
        """
        将指标推送到Pushgateway
        
        Args:
            job: 作业名称，默认使用app_name
        """
        try:
            job_name = job or self.app_name
            logger.debug(f"Pushing metrics to {self.pushgateway_url} with job={job_name}")
            push_to_gateway(
                self.pushgateway_url, 
                job=job_name, 
                registry=self.registry
            )
            logger.debug("Metrics pushed successfully")
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
    
    def _auto_push_worker(self):
        """自动推送指标的工作线程"""
        logger.info(f"Starting metrics auto-push thread with interval {self.push_interval}s")
        while self.auto_push_enabled:
            try:
                self.push_metrics()
            except Exception as e:
                logger.error(f"Error in metrics auto-push: {e}")
            
            # 等待下一次推送
            time.sleep(self.push_interval)
    
    def start_auto_push(self):
        """启动自动推送线程"""
        if self._push_thread is None or not self._push_thread.is_alive():
            self.auto_push_enabled = True
            self._push_thread = threading.Thread(
                target=self._auto_push_worker,
                daemon=True,
                name="metrics-auto-push"
            )
            self._push_thread.start()
            logger.info("Metrics auto-push thread started")
    
    def stop_auto_push(self):
        """停止自动推送线程"""
        self.auto_push_enabled = False
        if self._push_thread and self._push_thread.is_alive():
            self._push_thread.join(timeout=2.0)
            logger.info("Metrics auto-push thread stopped")


# 便捷函数，获取全局指标管理器实例
def get_metrics_manager(pushgateway_url: Optional[str] = None) -> MetricsManager:
    """
    获取全局MetricsManager实例
    
    Args:
        pushgateway_url: Prometheus Pushgateway的URL
        
    Returns:
        MetricsManager实例
    """
    return MetricsManager.get_instance(pushgateway_url)
