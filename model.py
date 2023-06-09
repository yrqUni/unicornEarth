from transformers import AutoConfig, ViTForMaskedImageModeling, SwinForMaskedImageModeling

def create_Init_ViT_model(model_name_or_path=None, disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    model = ViTForMaskedImageModeling(config=model_config)
    return model

def create_from_PT_ViT_model(model_name_or_path=None, disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    model = ViTForMaskedImageModeling.from_pretrained(
            model_name_or_path,
            config=model_config,)
    return model

def create_Init_SwinTransV2_model(model_name_or_path=None, disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    model = SwinForMaskedImageModeling(config=model_config)
    return model

def create_from_PT_SwinTransV2_model(model_name_or_path=None, disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    model = SwinForMaskedImageModeling.from_pretrained(
            model_name_or_path,
            config=model_config,)
    return model