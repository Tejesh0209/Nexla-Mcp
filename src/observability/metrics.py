import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Metrics:
    query_count:   int   = 0
    total_tokens:  int   = 0
    route_counts:  dict  = field(default_factory=lambda: defaultdict(int))
    latencies_ms:  list  = field(default_factory=list)

    def record(self, route: str, tokens: int, latency_ms: float) -> None:
        self.query_count  += 1
        self.total_tokens += tokens
        self.route_counts[route] += 1
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        lats = sorted(self.latencies_ms)
        n    = len(lats)
        return {
            "queries":            self.query_count,
            "avg_tokens":         round(self.total_tokens / max(self.query_count, 1), 1),
            "route_distribution": dict(self.route_counts),
            "p50_latency_ms":     round(lats[n // 2], 1)      if lats else 0,
            "p99_latency_ms":     round(lats[int(n * 0.99)], 1) if lats else 0,
        }


# single shared instance — imported by mcp_server.py
metrics = Metrics()