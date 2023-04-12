from safetensors import safe_open
import safetensors.torch
# from diffusers import StableDiffusionPipeline
import torch
# from diffusers import DPMSolverMultistepScheduler

all = [
    'add_textual_inversion_pt',
]


def add_textual_inversion(pipe, textual_inversion, special_string="token"):
    
    if not isinstance(textual_inversion, list):
        textual_inversion_file_list = [textual_inversion]
    else:
        textual_inversion_file_list = textual_inversion
    
    res_textual_inversion_tokens = []
    for idx, textual_inversion_file in enumerate(textual_inversion_file_list):
        if textual_inversion_file.endswith('safetensor'):
            textual_inversion = safetensors.torch.load_file(textual_inversion_file, device="cpu")
        else:
            textual_inversion = torch.load(textual_inversion_file)
                    
        if 'string_to_param' in textual_inversion:
            string_to_token = list(textual_inversion['string_to_token'].keys())[0]
            textual_inversion_embedding = textual_inversion['string_to_param'][string_to_token].detach()
        else:
            textual_inversion_embedding = textual_inversion['emb_params'].detach()
        raw_tokenizer_length = len(pipe.tokenizer)
        textual_inversion_tokens = [f"<{idx:02d}_{special_string}_{i:03d}>" for i in range(len(textual_inversion_embedding))]
        pipe.tokenizer.add_tokens(textual_inversion_tokens, special_tokens=True)
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        pipe.text_encoder.text_model.embeddings.token_embedding.weight.data[raw_tokenizer_length: raw_tokenizer_length+len(textual_inversion_embedding)] = textual_inversion_embedding
        res_textual_inversion_tokens.append(''.join(textual_inversion_tokens))

    return pipe, res_textual_inversion_tokens


