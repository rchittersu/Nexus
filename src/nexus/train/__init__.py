"""
Training: config-driven Flux.2 Klein on precomputed SSTK MDS data.

  from nexus.train import main
    main()  # Uses configs/klein4b/run1.yaml, MDS from precompute.py
"""

from .main import main

__all__ = ["main"]
