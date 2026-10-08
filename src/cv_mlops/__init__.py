"""Shared library for the CV MLOps demo: data prep, modeling, serving wrapper, evaluation.

Every notebook in src/notebooks imports from here so the exact same preprocessing and
model-building code runs locally (tests), during training (serverless GPU), at batch
inference (Spark UDF) and behind the real-time serving endpoint.
"""
