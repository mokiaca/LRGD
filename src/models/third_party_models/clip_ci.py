from clip_interrogator import Config, Interrogator


def load_ci_model(clip_model_name="ViT-L-14/openai"):
    """
    Load the CLIP interrogator model.

    Example:
    ```
    ci = load_ci_model()
    desc = ci.interrogate_classic(image)
    neg_desc = ci.interrogate_negative(image)
    ```
    """
    return Interrogator(Config(clip_model_name=clip_model_name, quiet=True))
