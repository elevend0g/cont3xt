"""
Performance targets and benchmarking utilities for integration tests
"""

# Performance targets in milliseconds
PERFORMANCE_TARGETS = {
    "context_assembly": 100,      # ms - Building context window from memory
    "pressure_relief": 500,       # ms - Complete pressure relief cycle
    "memory_search": 200,         # ms - Semantic memory search
    "entity_extraction": 50,      # ms per chunk - NLP entity extraction
    "graph_query": 100,           # ms - Neo4j graph queries
    "vector_search": 150,         # ms - Qdrant vector similarity search
    "sqlite_operation": 50,       # ms - SQLite CRUD operations
    "chunk_creation": 25,         # ms - Creating and tokenizing chunks
}

# Memory usage targets
MEMORY_TARGETS = {
    "max_context_size": 12000,     # tokens - Maximum context window size
    "relief_threshold": 9600,      # tokens - 80% threshold for pressure relief
    "post_relief_size": 7200,      # tokens - 60% target after relief
    "chunks_per_relief": 2,        # number - Minimum chunks removed per relief
    "max_chunk_size": 3200,        # tokens - Maximum individual chunk size
    "min_chunk_size": 200,         # tokens - Minimum chunk size
}

# Data integrity targets
ACCURACY_TARGETS = {
    "entity_extraction_precision": 0.85,  # Precision for entity extraction
    "memory_retrieval_recall": 0.80,      # Recall for relevant memory retrieval
    "character_consistency": 0.95,        # Character trait consistency score
    "plot_continuity": 0.90,              # Plot thread continuity score
    "semantic_similarity": 0.75,          # Minimum similarity for relevant chunks
}

# Test data limits
TEST_LIMITS = {
    "max_test_interactions": 500,         # Maximum interactions for performance tests
    "max_test_duration": 300,             # seconds - Maximum test duration
    "memory_growth_limit": 1.5,           # multiplier - Max memory growth factor
    "response_time_variance": 0.2,        # Maximum acceptable variance in response times
}


class PerformanceBenchmark:
    """Utility class for performance benchmarking during tests"""
    
    def __init__(self):
        self.measurements = {}
        self.baselines = {}
    
    def record_measurement(self, operation: str, duration_ms: float) -> None:
        """Record a performance measurement"""
        if operation not in self.measurements:
            self.measurements[operation] = []
        self.measurements[operation].append(duration_ms)
    
    def get_average(self, operation: str) -> float:
        """Get average duration for an operation"""
        if operation not in self.measurements:
            return 0.0
        return sum(self.measurements[operation]) / len(self.measurements[operation])
    
    def get_percentile(self, operation: str, percentile: float) -> float:
        """Get percentile duration for an operation"""
        if operation not in self.measurements:
            return 0.0
        measurements = sorted(self.measurements[operation])
        index = int(len(measurements) * percentile / 100)
        return measurements[min(index, len(measurements) - 1)]
    
    def check_target(self, operation: str, target_ms: float) -> bool:
        """Check if operation meets performance target"""
        avg = self.get_average(operation)
        return avg <= target_ms
    
    def get_performance_report(self) -> dict:
        """Generate comprehensive performance report"""
        report = {
            "operations": {},
            "targets_met": 0,
            "targets_total": 0,
            "overall_pass": True
        }
        
        for operation, target in PERFORMANCE_TARGETS.items():
            if operation in self.measurements:
                avg = self.get_average(operation)
                p95 = self.get_percentile(operation, 95)
                meets_target = avg <= target
                
                report["operations"][operation] = {
                    "average_ms": round(avg, 2),
                    "p95_ms": round(p95, 2),
                    "target_ms": target,
                    "meets_target": meets_target,
                    "measurement_count": len(self.measurements[operation])
                }
                
                if meets_target:
                    report["targets_met"] += 1
                else:
                    report["overall_pass"] = False
                    
                report["targets_total"] += 1
        
        return report


class MemoryValidator:
    """Utility class for validating memory usage and data integrity"""
    
    def __init__(self):
        self.token_counts = []
        self.chunk_counts = []
        self.entity_counts = []
    
    def record_context_state(self, total_tokens: int, chunk_count: int, entity_count: int) -> None:
        """Record current context state"""
        self.token_counts.append(total_tokens)
        self.chunk_counts.append(chunk_count)
        self.entity_counts.append(entity_count)
    
    def check_memory_targets(self) -> dict:
        """Check if memory usage meets targets"""
        if not self.token_counts:
            return {"status": "no_data", "checks": {}}
        
        max_tokens = max(self.token_counts)
        current_tokens = self.token_counts[-1]
        
        checks = {
            "max_context_size": max_tokens <= MEMORY_TARGETS["max_context_size"],
            "current_size_valid": current_tokens <= MEMORY_TARGETS["max_context_size"],
            "growth_controlled": len(self.token_counts) < 2 or 
                               (current_tokens / self.token_counts[0]) <= TEST_LIMITS["memory_growth_limit"]
        }
        
        return {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "max_tokens": max_tokens,
            "current_tokens": current_tokens,
            "target_max": MEMORY_TARGETS["max_context_size"]
        }
    
    def validate_pressure_relief(self, before_tokens: int, after_tokens: int) -> dict:
        """Validate a pressure relief operation"""
        relief_ratio = after_tokens / before_tokens if before_tokens > 0 else 1.0
        target_ratio = MEMORY_TARGETS["post_relief_size"] / MEMORY_TARGETS["max_context_size"]
        
        return {
            "tokens_before": before_tokens,
            "tokens_after": after_tokens,
            "relief_ratio": round(relief_ratio, 3),
            "target_ratio": round(target_ratio, 3),
            "meets_target": relief_ratio <= target_ratio * 1.1,  # 10% tolerance
            "tokens_freed": before_tokens - after_tokens
        }