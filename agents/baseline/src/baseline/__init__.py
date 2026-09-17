"""
Baseline ISD Agent: Single prompt ADDIE generator

A simple baseline agent for comparison that generates complete
instructional design outputs with a single LLM call.
Uses the shared provider-neutral LLM factory.
"""

__version__ = "0.1.0"

from baseline.generator import BaselineGenerator

__all__ = ["BaselineGenerator", "__version__"]
