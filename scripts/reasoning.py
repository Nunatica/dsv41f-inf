"""DeepSeek V4.1 forced-decode Off output contract."""
DSV41F_REASONING_BUDGET_MESSAGE = "... Reasoning budget is exceeded. It's time to answer now."


def verify_off_reasoning(value, *, allow_empty=False, allow_partial=False):
    if value == DSV41F_REASONING_BUDGET_MESSAGE:
        return
    if allow_empty and not value:
        return  # Reference servers may close the thinking block in prefill.
    if allow_partial and value and DSV41F_REASONING_BUDGET_MESSAGE.startswith(value):
        return  # max_tokens can stop during the forced prefix.
    raise RuntimeError(f'Expected forced Off message, got {value!r}')
