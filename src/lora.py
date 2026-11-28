import logging
import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.utils import count_params

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=1.0, lora_dtype: str | None = None):
        super().__init__()
        if not lora_dtype:
            lora_torch_dtype = base.weight.dtype
        else:
            try:
                lora_torch_dtype = getattr(torch, lora_dtype)
                logging.info(f"Using LoRA dtype: {lora_dtype}")
            except AttributeError:
                lora_torch_dtype = base.weight.dtype
                logging.warning(f"Invalid LoRA dtype: {lora_dtype}. Using default dtype.")

        
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
            torch.empty(
                (r, self.base.in_features),
                dtype=lora_torch_dtype,
                device=base.weight.device,
            ),
        )
        self.B = nn.Parameter(
            torch.zeros(
                (self.base.out_features, r),
                dtype=lora_torch_dtype,
                device=base.weight.device,
            ),
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        if self.merged_weights:
            return self.base(x)

        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge_weights(self):
        if self.merged_weights:
            return
        # Merge the weights of A and B into the base linear layer
        self.base.weight.data += (self.B @ self.A) * self.scale
        self.merged_weights = True

    @torch.no_grad()
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


def train_step(model, optimizer, inputs, labels):
    # the optimizer is created once outside, so Adam's moment estimates persist across steps
    optimizer.zero_grad()
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_loss(model, inputs):
    model.eval()
    return model(**inputs, labels=inputs["input_ids"]).loss.item()


def apply_lora(base_model, r=4, alpha=1.0):
    """Create a LoRA model from a base model.

    params:
        base_model: The base model to add LoRA to
        r: The rank of the LoRA matrices
        alpha: The scaling factor for the LoRA matrices
    """
    for p in base_model.parameters():
        p.requires_grad = False

    lora_model = add_lora(base_model, r=r, alpha=alpha)
    return lora_model

if __name__ == "__main__":
    # load the smalles smollm model
    model_name = "HuggingFaceTB/SmolLM-135M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    print("Fresh model param count:")
    count_params(model)

    # only finetune the LoRA layers
    for p in model.parameters():
        p.requires_grad = False
    print("Model param count after setting requires_grad=False:")
    count_params(model)


    # check the gradients for the loRA layers
    model = add_lora(model, target_names=("q_proj", "v_proj"), r=4, alpha=1.0)
    print("Model param count after adding LoRA layers:")
    count_params(model)

    # train on a gibberish sample:
    inputs = tokenizer(
        "zq florp vex kandu mib trassel ow quenzi", return_tensors="pt"
    ).to(model.device)
    labels = inputs["input_ids"]

    # unrelated normal sentence
    heldout = tokenizer(
        "The capital of France is Paris.", return_tensors="pt"
    ).to(model.device)

    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    lora_before   = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    heldout_before = eval_loss(model, heldout)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    model.train()
    losses = [train_step(model, optimizer, inputs, labels) for _ in range(50)]
    print("gibberish loss every 10 steps:", [round(x, 3) for x in losses[::10]], "final:", round(losses[-1], 3))
    print(f"held-out loss before/after: {heldout_before:.3f} / {eval_loss(model, heldout):.3f}")
    
    params = dict(model.named_parameters())
    print("frozen weights changed:", any(not torch.equal(params[n], w) for n, w in frozen_before.items()))  # want False
    print("LoRA weights changed:  ", any(not torch.equal(params[n], w) for n, w in lora_before.items()))    # want True