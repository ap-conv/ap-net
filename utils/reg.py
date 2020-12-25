import torch


def regularize(*features):
    if len(features) == 2:
        return regularize_2(*features)
    elif len(features) == 3:
        feature_1, feature_2, feature_3 = features
        return regularize_2(feature_1, feature_2) + regularize_2(feature_1, feature_3) + regularize_2(feature_2, feature_3)
    else:
        raise ValueError

def regularize_2(feature_1, feature_2):
    N, C_1, H, W = feature_1.shape
    S_1 = H * W
    N, C_2, H, W = feature_2.shape
    S_2 = H * W
    feature_1 = feature_1.view(N, C_1, S_1)
    feature_2 = feature_2.view(N, C_2, S_2).transpose(1, 2)
    result = torch.matmul(feature_1, feature_2)
    return torch.mean(result)
