import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        # load the base and set params to not require gradients
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        # set the default params
        self.r = r
        self.scale = alpha / r
        self.merged_weights = False

        # Initialize A and B matrices
        self.A = nn.Parameter(
            torch.empty((r, self.base.in_features)), device=base.weight.device
        )
        self.B = nn.Parameter(
            torch.zeros((self.base.out_features, r)), device=base.weight.device
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        if self.merged_weights:
            return self.base(x)

        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merge_weights(self):
        if self.merged_weights:
            return
        # Merge the weights of A and B into the base linear layer
        self.base.weight.data += (self.B @ self.A) * self.scale
        self.merged_weights = True

    def unmerge_weights(self):
        if not self.merged_weights:
            return
        # Unmerge the weights of A and B from the base linear layer
        self.base.weight.data -= (self.B @ self.A) * self.scale
        self.merged_weights = False


def add_lora(model, target_names=("q_proj", "v_proj"), r=4, alpha=1.0):
    for name, module in model.named_modules():
        for (
            child_name,
            child,
        ) in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                # Replace the linear layer with LoRALinear
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
    return model


# B is zero and A is gaussian
# alpha and r are constant
if __name__ == "__main__":
    # Example usage of LoRALinear
    lora_linear = LoRALinear()
    print("LoRALinear instance created:", lora_linear)

    # load the smalles smollm model
    model_name = "HuggingFaceTB/SmolLM-135M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # check the gradients for the loRA layers
