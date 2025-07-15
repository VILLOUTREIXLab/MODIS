def adjust_time(seconds: int) -> str:
    """
    Converts a given number of seconds into a more appropriate time unit

    Args:
        seconds (int): The number of seconds to be converted.

    Return:
        (str): A string representing the time duration in a more convenient unit.
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

def accuracy(logits, target):
    pred = logits.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).sum().item()
    return correct / logits.size(0)