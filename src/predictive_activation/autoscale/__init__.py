"""Predictive autoscaling: forecast request load and scale before pods are needed.

Offline pieces (no cluster required): workload traces, a queueing simulator with
pod startup delay, forecasters, scaling policies and an evaluation harness.
"""
