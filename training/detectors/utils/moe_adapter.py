import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEAdapter(nn.Module):
    """Token-wise mixture of bottleneck adapters with a residual output.

    Established components: a learned softmax gate, top-k expert dispatch, and
    down/nonlinear/up adapter experts. DFD-HR inserts one module after attention
    and another after the MLP in every CLIP block.

    ``x=[B,L,D]`` is flattened to ``[B*L,D]`` for routing and restored before
    return. With ``num_experts=top_k=4`` every token uses all four experts with
    learned weights; setting ``top_k < num_experts`` makes dispatch sparse.
    ``load_balancing_weight`` is recorded for compatibility but is not consumed
    by the current training loss.
    """

    def __init__(
        self,
        D_features,
        num_experts=4,
        k=1,
        top_k=None,
        mlp_ratio=0.25,
        expert_mlp_ratio=0.25,
        act_layer=nn.GELU,
        skip_connect=True,
        noise=True,
        load_balancing_weight=0.0,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.num_experts = num_experts
        self.k = top_k if top_k is not None else k
        self.D_features = D_features
        self.expert_hidden = int(D_features * expert_mlp_ratio)
        self.load_balancing_weight = load_balancing_weight
        if self.k < 1 or self.k > self.num_experts:
            raise ValueError(
                f'top_k must be in [1, num_experts], got {self.k} '
                f'for {self.num_experts} experts'
            )

        # Gate each token independently; this is not one decision per image.
        self.gate = nn.Linear(D_features, num_experts)
        self.gate_noise = noise

        # Each established adapter bottleneck maps D -> D*ratio -> D.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(D_features, self.expert_hidden),
                    act_layer(),
                    nn.Linear(self.expert_hidden, D_features),
                )
                for _ in range(num_experts)
            ]
        )

        # Optional shared projection only when the two configured ratios differ.
        shared_hidden = int(D_features * mlp_ratio)
        self.D_fc1 = (
            nn.Linear(D_features, shared_hidden)
            if mlp_ratio != expert_mlp_ratio
            else nn.Identity()
        )
        self.D_fc2 = (
            nn.Linear(shared_hidden, D_features)
            if mlp_ratio != expert_mlp_ratio
            else nn.Identity()
        )
        self.act = act_layer()

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, self.D_features)

        # [B*L,D] -> [B*L,E].
        logits = self.gate(x_flat)
        if self.gate_noise and self.training:
            # DFD-HR Eq. 13: stochastic gate noise encourages diverse weights.
            noise = torch.randn_like(logits) * F.softplus(logits)
            logits = logits + noise

        raw_gates = F.softmax(logits, dim=-1)
        top_k_vals, top_k_indices = torch.topk(raw_gates, self.k, dim=-1)

        # Renormalize after TopK so selected expert weights sum to one.
        gates = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)

        expert_outputs = torch.zeros_like(x_flat)

        # Dispatch once per expert, then accumulate its weighted contribution
        # for every token that selected it.
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not torch.any(expert_mask):
                continue

            expert_input = x_flat[expert_mask]
            expert_result = self.experts[i](expert_input)

            for j in range(self.k):
                weight_mask = top_k_indices[expert_mask, j] == i
                if not torch.any(weight_mask):
                    continue

                gate_weight = gates[expert_mask][weight_mask, j].to(expert_result.dtype)
                weighted_result = expert_result[weight_mask] * gate_weight.unsqueeze(1)
                expert_outputs[expert_mask] = expert_outputs[expert_mask].scatter_add(
                    0,
                    torch.nonzero(weight_mask, as_tuple=False).expand(-1, self.D_features),
                    weighted_result,
                )

        if not isinstance(self.D_fc1, nn.Identity):
            expert_outputs = self.act(self.D_fc1(expert_outputs))
            expert_outputs = self.D_fc2(expert_outputs)

        expert_outputs = expert_outputs.view(original_shape)

        # Paper Eq. 14 defines MOA as input plus the routed adapter mixture.
        if self.skip_connect:
            return x + expert_outputs
        return expert_outputs

    def extra_repr(self):
        return (
            f"experts={self.num_experts}, k={self.k}, "
            f"input_dim={self.D_features}, "
            f"expert_hidden={self.expert_hidden}, "
            f"noise={self.gate_noise}, "
            f"load_balancing_weight={self.load_balancing_weight}"
        )


if __name__ == '__main__':
    x = torch.rand(8, 257, 1024)
    moe_adapter = MoEAdapter(D_features=1024)
    x_output = moe_adapter(x)
    print(x_output.size())
