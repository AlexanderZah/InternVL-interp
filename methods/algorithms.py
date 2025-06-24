

def internal_confidence(tokenizer, softmax_probs, class_):
    class_token_indices = tokenizer.encode(class_)[1:]
    return softmax_probs[class_token_indices].max()


def internal_confidence_heatmap(tokenizer, softmax_probs, class_):
    class_token_indices = tokenizer.encode(class_)[1:]
    return softmax_probs[class_token_indices].max(axis=0).T


def internal_confidence_segmentation(tokenizer, softmax_probs, class_, num_patches=24):
    class_token_indices = tokenizer.encode(class_)[1:]
    return (
        softmax_probs[class_token_indices]
        .max(axis=0)
        .max(axis=0)
        .reshape(num_patches, num_patches)
        .astype(float)
    )
