from importlib.util import find_spec
from typing import Dict, Any, Union, Tuple, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore

import torch.nn as nn

class _ForwardRedirection:
    """Implements the `forward-redirection`.

    Taken from Pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self, wrapper_module: nn.Module, original_module: nn.Module, method: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass

    def on_after_outer_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None

def get_model(model_name: str, use_liger: bool = True, model_kwargs: Union[Dict[str, Any], None] = None) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if is_liger_available() and use_liger:
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerKernelForCausalLM # type: ignore
        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
def get_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure a non-empty chat template is available. Some base checkpoints expose the attribute
    # but leave it empty; the Trainer relies on a valid template for apply_chat_template.
    template_attr = getattr(tokenizer, "chat_template", None)
    if template_attr is None or template_attr.strip() == "":
        # Heuristic: try the corresponding "-Instruct" checkpoint which usually ships the template.
        instruct_name = None
        if not model_name.endswith("-Instruct"):
            instruct_name = model_name + "-Instruct" if not model_name.endswith("/") else model_name.rstrip("/") + "-Instruct"

        if instruct_name is not None:
            try:
                instruct_tok = AutoTokenizer.from_pretrained(instruct_name, trust_remote_code=True)
                if getattr(instruct_tok, "chat_template", None):
                    tokenizer.chat_template = instruct_tok.chat_template  # type: ignore[attr-defined]
            except Exception:
                # Could not fetch instruct variant – fall through to error below
                pass

    # Final check
    if getattr(tokenizer, "chat_template", None) is None or tokenizer.chat_template.strip() == "":  # type: ignore[attr-defined]
        raise ValueError(
            f"Tokenizer for model '{model_name}' lacks a usable chat_template. Specify a tokenizer that defines one "
            "or manually set tokenizer.chat_template before training."
        )

    return tokenizer
            
def get_model_and_tokenizer(model_name: str, use_liger: bool = True, model_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Any, Any]:
    model = get_model(model_name, use_liger, model_kwargs)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer