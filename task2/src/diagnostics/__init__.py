"""Diagnostics package for Task 2.
"""
from task2.src.diagnostics.calibration import run_connector_magnitude_calibration
from task2.src.diagnostics.smoke_tests import (
    test_autograd_gradient_flow,
    test_t4_memory_and_checkpointing,
)

__all__ = [
    "run_connector_magnitude_calibration",
    "test_autograd_gradient_flow",
    "test_t4_memory_and_checkpointing",
]
