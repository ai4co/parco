import torch

from rl4co.data.transforms import (
    TensorDict,
    batchify,
    min_max_normalize,
    symmetric_augmentation,
)


def graph_dilation(X, c, min_s=0.5, max_s=1.0):
    s = (torch.rand([X.shape[0]], device=X.device)) * (max_s - min_s) + min_s

    # Expand dimensions of s and c for broadcasting with X
    s = s[..., None, None]
    c = c.expand_as(X)

    Y = s * (X - c) + c
    return Y, s, c


def augment_graph(X, min_s=0.5, max_s=1.0, **kw):
    batch_size, num_nodes, _ = X.shape
    c = torch.rand(batch_size, 1, 2, device=X.device)
    out, s, c = graph_dilation(X, c, min_s=min_s, max_s=max_s)
    return out, s, c


class DilationAugmentation(object):
    def __init__(
        self,
        env_name: str = None,
        num_augment: int = 8,
        use_symmetric_augment: bool = True,
        min_s: float = 0.5,
        max_s: float = 1.0,
        normalize: bool = False,
        first_aug_identity: bool = True,
        **unused_kwargs,
    ):
        self.feats = ["locs"]
        self.num_augment = num_augment
        self.use_symmetric_augment = use_symmetric_augment
        self.normalize = normalize
        self.augmentation = augment_graph
        if use_symmetric_augment:
            self.aug_sym = symmetric_augmentation
        self.min_s = min_s
        self.max_s = max_s
        self.first_aug_identity = first_aug_identity

    def __call__(self, td: TensorDict) -> TensorDict:
        td_aug = batchify(td, self.num_augment)

        for feat in self.feats:
            init_aug_feat = td_aug[feat][:, 0].clone()

            # Dilation augmentation
            aug_feat, s, c = self.augmentation(
                td_aug[feat], min_s=self.min_s, max_s=self.max_s
            )

            # Symmetric augmentation
            if self.use_symmetric_augment:
                aug_feat = self.aug_sym(aug_feat, self.num_augment)

            # Set feat
            td_aug[feat] = aug_feat
            if self.normalize:
                td_aug[feat] = min_max_normalize(td_aug[feat])

            if self.first_aug_identity:
                # first augmentation is identity
                aug_feat[:, 0] = init_aug_feat

        return td_aug, s, c
