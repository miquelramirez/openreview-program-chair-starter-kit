# Paper Classification System

# 2025 data

import numpy as np

type_of_contribution = [
    "Theory",
    "Algorithms",
    "Models and Representations",
    "Position Paper",
    "Tools"]

topic = [
    "Abstract Models",
    "Machine Learning",
    "Robotics and Control Theory",
    "Human-Aware",
    "Applications",
    "Knowledge Engineering"]

mandatory_subject_tags = [
    "PS: Activity and plan recognition",
    "PS: Applications",
    "PS: Learning for planning and scheduling",
    "PS: Mixed discrete/continuous planning",
    "PS: Model-based reasoning",
    "PS: Optimization of spatio-temporal systems",
    "PS: Plan execution and monitoring",
    "PS: Planning under uncertainty",
    "PS: Planning with large language models",
    "PS: Planning with Markov decision process models (MDPs, POMDPs)",
    "PS: Re-planning and plan repair",
    "PS: Routing",
    "PS: Scheduling",
    "PS: Scheduling under uncertainty",
    "PS: Temporal planning",
    "PS: Distributed and multi-agent planning",
    "PS: Planning with Hierarchical Task Networks (HTN)",
    "PS: Classical (fully-observable, deterministic) planning",
    "PS: Fully observable non-deterministic planning",
    "PS: Partially observable planning",
    "PS: Planning with incomplete models",
    "PS: Real-time planning",
    "PS: Theoretical foundations of planning",
    "PS: Multi-agent path-finding",
    "PS: Generalized planning",
    "PS: Search in planning and scheduling",
    "PS: SAT, SMT and CP",
    "PS: Local search and evolutionary programming",
    "PS: Sub-modular and gradient-free optimization",
    "PS: Mathematical programming",
    "PS: Infinite-horizon optimal control problems",
    "PS: Model checking for trust, safety and robustness",
    "ROB: Motion and path planning"]

additional_subject_tags = [
    "HAI: Human-Aware planning and behavior prediction",
    "HAI: Planning and decision support for human-machine teams",
    "KRR: Reasoning about actions",
    "KRR: Reasoning about knowledge and belief",
    "ML: Reinforcement learning",
    "ML: Representation learning",
    "UAI: Sequential decision making",
    "UAI: Uncertainty representations"]

def get_type_id(value: str) -> int:
    return type_of_contribution.index(value)

def get_topic_id(value: str) -> int:
    return topic.index(value)

def get_mandatory_subject_tag_id(value: str) -> int:
    return mandatory_subject_tags.index(value)

def get_additional_subject_tag_id(value: str) -> int:
    return len(mandatory_subject_tags) + additional_subject_tags.index(value)

def map_to_hypercube(v: list, dim: int) -> np.array:
    """
    Maps vector v of integers into the dim-dimensional hypercube
    :param v:
    :param dim:
    :return:
    """
    w = np.zeros(dim, dtype=np.int64)
    for v_i in v:
        w[v_i] = 1
    return w

def map_paper_type_vec(v: list) -> np.array:
    """

    :param v:
    :return:
    """
    return map_to_hypercube(v, len(type_of_contribution))

def map_paper_subject_vec(v: list) -> np.array:
    """

    :param v:
    :return:
    """
    return map_to_hypercube(v, len(mandatory_subject_tags) + len(additional_subject_tags))