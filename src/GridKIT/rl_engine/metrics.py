"""

Metrics Builder: Post-training metrics construction from RL training logs.

This module provides functions and a service class to build EpisodeMetrics,
SimResult, and IterationMetrics objects from JSONL log files created during
RLlib training.

Usage:
    Using MetricsBuilder service class (Builder-Pattern)
    from GridKIT.rl_engine.metrics_builder import MetricsBuilder
    
    builder = MetricsBuilder(log_dir=Path("/tmp/gridkit_rl_logs")) #TODO change path
    
    ep = builder.build_episode_metrics("w0100_0001")
    sr = builder.build_sim_result("w0100_0001")
    iterations = builder.load_iteration_metrics()

    Or get everything at once
    all_metrics = builder.build_all()
"""



    def __init__(self, log_dir: Path):
        """
        Initialize the builder.
        
        Args:
            log_dir: Directory containing episode log files
        """
        self.log_dir = log_dir
        self._collector: Optional[EpisodeMetricsCollector] = None
    
    def _get_collector(self) -> EpisodeMetricsCollector:
        """Lazy-load the collector."""
        if self._collector is None:
            self._collector = EpisodeMetricsCollector(self.log_dir)
            self._collector.collect_all()
        return self._collector
    
    def build_episode_metrics(self, episode_id: str) -> EpisodeMetrics:
        """
        Build EpisodeMetrics for a specific episode.
        
        Args:
            episode_id: Episode identifier (e.g., "w0100_0001")
            
        Returns:
            EpisodeMetrics object
        """
        log_file = self.log_dir / f"episode_{episode_id}.jsonl"
        return build_episode_metrics_from_file(log_file)
    
    def build_sim_result(self, episode_id: str) -> SimResult:
        """
        Build SimResult for a specific episode.
        
        Args:
            episode_id: Episode identifier (e.g., "w0100_0001")
            
        Returns:
            SimResult object
        """
        log_file = self.log_dir / f"episode_{episode_id}.jsonl"
        return build_sim_result_from_file(log_file)
    
    def load_iteration_metrics(self) -> List[IterationMetrics]:
        """
        Load IterationMetrics from saved JSON file.
        
        Returns:
            List of IterationMetrics objects
        """
        output_file = self.log_dir / "iteration_metrics.json"
        
        if not output_file.exists():
            return []
        
        with open(output_file, 'r') as f:
            metrics_dicts = json.load(f)
        
        return [IterationMetrics(**m) for m in metrics_dicts]
    
    def build_all(self) -> AllMetrics:
        """
        Build all metrics at once.
        
        Returns:
            AllMetrics container with episode_metrics, sim_results, iteration_metrics
        """
        collector = self._get_collector()
        
        # Get all episode metrics
        episode_metrics = collector.get_all_metrics_skip_first()
        
        # Get all sim results (from all workers)
        sim_results: List[SimResult] = []
        for worker_id in collector.metrics_by_worker.keys():
            sim_results.extend(collector.get_sim_results_for_worker(worker_id))
        
        # Load iteration metrics
        iteration_metrics = self.load_iteration_metrics()
        
        return AllMetrics(
            episode_metrics=episode_metrics,
            sim_results=sim_results,
            iteration_metrics=iteration_metrics,
        )