
# learning rate schedule: stable then decay
def get_lr(step, num_step, cooldown_frac=0.4):
    x = step / num_step  # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac
        # return w * 1.0 + (1 - w) * 0.1
