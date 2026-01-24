

def get_model_class_name(model):
    # Deal with the case where we have LoRA
    try:
        base_model = model.get_base_model()
        class_name = type(base_model).__name__
    except AttributeError:
        class_name = type(model).__name__
    return class_name


def model_is_evabyte(model):
    class_name = get_model_class_name(model)
    return class_name == "EvaByteModel"


def model_is_llama(model):
    class_name = get_model_class_name(model)
    return class_name == "TPULlamaModel"


def get_model_type(model):
    class_name = get_model_class_name(model)

    if class_name in ("EvaByteForCausalLM", "EvaByteModel"):
        return "evabyte"
    elif class_name in ("TPULlamaForCausalLM", "TPULlamaModel"):
        return "llama"
    else:
        raise ValueError(f"Unsupported model type: {class_name}")
    return class_name
