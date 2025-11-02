def accuracy(logits,targets):
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()
