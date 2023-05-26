from transformers import AutoConfig, ViTForMaskedImageModeling 

def create_Init_model(model_name_or_path=None,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    model = ViTForMaskedImageModeling(config=model_config)
    
    return model

def create_from_PT_model(model_name_or_path=None,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    model = ViTForMaskedImageModeling.from_pretrained(
            model_name_or_path,
            config=model_config,)
    return model