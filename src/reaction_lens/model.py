"""Description-conditioned scorers with no reaction-ID-specific parameters.

Tensor notation: B = publications, R = reaction options, S = sentence spans,
D = encoder width, P = learned projection width. Each output logit depends on
one reaction and the supplied publication features, never on competing options.
"""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CatalogScorer(nn.Module):
    """Normalized dual projections with a learned positive scale and bias.

    Input shapes: articles [B, D], descriptions [R, D]. Output: logits [B, R].
    Projections are initialized identically and subsequently learned separately.
    """

    def __init__(self, input_dim: int = 768, projection_dim: int = 256):
        super().__init__()
        self.article = nn.Linear(input_dim, projection_dim, bias=False)
        self.reaction = nn.Linear(input_dim, projection_dim, bias=False)
        self.reaction.load_state_dict(self.article.state_dict())
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def reaction_vectors(self, descriptions: Tensor) -> Tensor:
        """Project catalog features once for reuse across publications: [R, P]."""
        return F.normalize(self.reaction(descriptions), dim=-1)

    def forward(self, articles: Tensor, descriptions: Tensor) -> Tensor:
        return self.score_prepared(articles, self.reaction_vectors(descriptions))

    def score_prepared(self, articles: Tensor, reaction_vectors: Tensor) -> Tensor:
        queries = F.normalize(self.article(articles), dim=-1)
        return self.log_scale.clamp(max=math.log(100)).exp() * (queries @ reaction_vectors.T) + self.bias


class LocalCatalogScorer(CatalogScorer):
    """Global similarity plus the strongest sentence match above a learned null.

    z = exp(clamped scale) * (global + softplus(weight) * local_excess) + bias.
    The article projection is shared by publication and sentence features. Only
    two scalar parameters are added to the global scorer. Null means no local
    contribution.

    Sentences: [B, S, D]; boolean validity mask: [B, S]. Option blocks bound
    intermediate memory without filtering reactions or capping selections.
    """

    def __init__(self, input_dim: int = 768, projection_dim: int = 256, option_block: int = 1024):
        super().__init__(input_dim, projection_dim)
        self.local_weight = nn.Parameter(torch.tensor(-2.0))
        self.null_similarity = nn.Parameter(torch.tensor(math.atanh(0.5)))
        if option_block < 1:
            raise ValueError('Option block must be positive')
        self.option_block = option_block

    def forward(self, articles: Tensor, descriptions: Tensor, sentences: Tensor, mask: Tensor) -> Tensor:
        return self.score_prepared(articles, self.reaction_vectors(descriptions), sentences, mask)

    def score_prepared(self, articles: Tensor, reaction_vectors: Tensor, sentences: Tensor, mask: Tensor) -> Tensor:
        if sentences.ndim != 3 or mask.shape != sentences.shape[:2] or sentences.shape[0] != len(articles):
            raise ValueError('Invalid sentence batch/mask')
        if sentences.shape[1] == 0 or mask.dtype != torch.bool:
            raise ValueError('Need padded sentence axis and boolean mask')
        global_queries = F.normalize(self.article(articles), dim=-1)
        local_queries = F.normalize(self.article(sentences), dim=-1)
        scale = self.log_scale.clamp(max=math.log(100.0)).exp()
        null = self.null_similarity.tanh()
        weight = F.softplus(self.local_weight)
        outputs = []
        for start in range(0, len(reaction_vectors), self.option_block):
            options = reaction_vectors[start:start + self.option_block]
            similarity = local_queries @ options.T
            best = similarity.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
            local = torch.maximum(best, null) - null
            outputs.append(scale * (global_queries @ options.T + weight * local) + self.bias)
        return torch.cat(outputs, dim=1)

    def aggregation_parameters(self) -> dict[str, float]:
        return {
            'local_weight': float(F.softplus(self.local_weight).detach()),
            'null_similarity': float(self.null_similarity.tanh().detach()),
        }
