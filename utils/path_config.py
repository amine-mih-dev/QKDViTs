from dataclasses import dataclass, asdict

@dataclass
class ProjectPaths:
    results_dir: str = "results"
    qres_dir: str = "qres"
    aucs_dir: str = "aucs"
    final_results_dir: str = "final_results"

    def as_dict(self):
        return asdict(self)
