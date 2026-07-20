import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st

class MoEAdapter(nn.Module):
    def __init__(self, 
                 D_features, 
                 num_experts=4, 
                 k=1,
                 top_k=None,
                 mlp_ratio=0.25,
                 expert_mlp_ratio=0.25,
                 act_layer=nn.GELU, 
                 skip_connect=True,
                 noise=True,
                 load_balancing_weight=0.0):
        """
        MoE-enhanced Adapter with Mixture of Experts
        Args:
            num_experts: number of experts
            k: number of experts to activate per sample
            expert_mlp_ratio: hidden-dim shrink ratio inside each expert MLP
            noise: whether to add noise in the gating network
        """
        super().__init__()
        self.skip_connect = skip_connect
        self.num_experts = num_experts
        self.k = top_k if top_k is not None else k
        self.D_features = D_features
        self.expert_hidden = int(D_features * expert_mlp_ratio)
        self.load_balancing_weight = load_balancing_weight
        if self.k < 1 or self.k > self.num_experts:
            raise ValueError(f'top_k must be in [1, num_experts], got {self.k} for {self.num_experts} experts')
        
        # Gating network
        self.gate = nn.Linear(D_features, num_experts)
        self.gate_noise = noise
        
        # Expert pool (each expert is an independent Adapter)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D_features, self.expert_hidden),
                act_layer(),
                nn.Linear(self.expert_hidden, D_features)
            ) for _ in range(num_experts)
        ])
        
        # Shared down/up projection layers (optional)
        shared_hidden = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, shared_hidden) if mlp_ratio != expert_mlp_ratio else nn.Identity()
        self.D_fc2 = nn.Linear(shared_hidden, D_features) if mlp_ratio != expert_mlp_ratio else nn.Identity()
        self.act = act_layer()

    def forward(self, x):
        # x shape: (batch_size, seq_len, D_features)
        original_shape = x.shape
        x_flat = x.view(-1, self.D_features)  # flatten to (N, D)
        
        # Gating network forward
        logits = self.gate(x_flat)
        if self.gate_noise and self.training:
            # Add trainable noise (improves expert diversity)
            noise = torch.randn_like(logits) * F.softplus(logits)
            logits = logits + noise

        # Select top-k experts
        raw_gates = F.softmax(logits, dim=-1)
        top_k_vals, top_k_indices = torch.topk(raw_gates, self.k, dim=-1)
        
        # Normalize gate values
        gates = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)
        
        # Initialize output buffer
        expert_outputs = torch.zeros_like(x_flat)
        
        # Routing logic (supports multiple experts per token)
        for i in range(self.num_experts):
            # Build mask of tokens routed to the current expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not torch.any(expert_mask):
                continue
                
            # Gather tokens assigned to this expert
            expert_input = x_flat[expert_mask]
            
            # Expert forward
            expert_result = self.experts[i](expert_input)
            
            # Weight and accumulate results
            for j in range(self.k):
                # Locate the gate weight slot for this expert within each token
                weight_mask = top_k_indices[expert_mask, j] == i
                if not torch.any(weight_mask):
                    continue
                    
                # Apply gate weight
                gate_weight = gates[expert_mask][weight_mask, j].to(expert_result.dtype)
                weighted_result = expert_result[weight_mask] * gate_weight.unsqueeze(1)
                expert_outputs[expert_mask] = expert_outputs[expert_mask].scatter_add(
                    0, 
                    torch.nonzero(weight_mask, as_tuple=False).expand(-1, self.D_features),
                    weighted_result
                )

        # Pass through shared MLP layers (if enabled)
        if not isinstance(self.D_fc1, nn.Identity):
            expert_outputs = self.act(self.D_fc1(expert_outputs))
            expert_outputs = self.D_fc2(expert_outputs)
        
        # Restore original shape
        expert_outputs = expert_outputs.view(original_shape)
        
        # Residual connection
        if self.skip_connect:
            return x + expert_outputs
        else:
            return expert_outputs

    def extra_repr(self):
        return (f"experts={self.num_experts}, k={self.k}, "
                f"input_dim={self.D_features}, "
                f"expert_hidden={self.expert_hidden}, "
                f"noise={self.gate_noise}, "
                f"load_balancing_weight={self.load_balancing_weight}")
        
if __name__ == '__main__':
    x = torch.rand(8, 257, 1024)
    moe_adapter = MoEAdapter(D_features=1024)
    x_output = moe_adapter(x)
    print(x_output.size())
