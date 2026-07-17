def learning_rate_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    if not 0 <= step <= total_steps:
        raise ValueError("step must be between zero and total_steps")
    if warmup_steps and step <= warmup_steps:
        return max_learning_rate * step / warmup_steps
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    return max_learning_rate - (max_learning_rate - min_learning_rate) * progress
