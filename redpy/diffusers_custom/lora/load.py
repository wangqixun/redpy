from safetensors import safe_open
import torch
from ...utils_redpy.json_utils import load_json

all = [
    'add_lora',
    'add_lora_bin',
    'add_lora_weiUI',
    'add_lora_unet_and_text_encoder'
]

def is_int(d):
    try:
        d = int(d)
        return True
    except Exception as e:
        return False


def add_lora_weiUI(lora_path, pipe, lora_weight=0.5):

    if isinstance(lora_path, str):
        lora_path_list = [lora_path]
    elif isinstance(lora_path, list):
        lora_path_list = lora_path
    else:
        print(f"lora_path must be str or list[str]. Pipeline has not been modified!")
        return pipe

    if isinstance(lora_weight, float) or isinstance(lora_weight, int):
        lora_weight_list = [lora_weight]
    elif isinstance(lora_weight, list):
        lora_weight_list = lora_weight
    else:
        print(f"lora_weight must be a number or list[number]. Pipeline has not been modified!")
        return pipe

    if len(lora_weight_list) == 1:
        lora_weight_list = lora_weight_list * len(lora_path_list)
    if len(lora_weight_list) != len(lora_path_list):
        print(f"The numbers of lora_weight and lora_path does not match. Pipeline has not been modified!")
        return pipe



    device = pipe.device
    for lora_path, lora_weight in zip(lora_path_list, lora_weight_list):
        tensors = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).to(device)

        for k_lora, v_lora in tensors.items():
            if k_lora.startswith('lora_te'):
                model = pipe.text_encoder
                # continue
            elif k_lora.startswith('lora_unet'):
                model = pipe.unet
            else:
                print(k_lora)
            
            # down 跳过
            if '.lora_down.' in k_lora:
                # print('lora_down')
                continue
            if '.alpha' in k_lora:
                # print('alpha')
                continue
            # print('lora_up')

            k_lora_name = k_lora.split('.')[0]
            attr_name_list = k_lora_name.split('_')
            cur_attr = model
            latest_attr_name = ''
            for idx in range(2, len(attr_name_list)):
                attr_name = attr_name_list[idx]
                if is_int(attr_name):
                    cur_attr = cur_attr[int(attr_name)]
                    latest_attr_name = ''
                else:
                    try:
                        if latest_attr_name != '':
                            cur_attr = cur_attr.__getattr__(f"{latest_attr_name}_{attr_name}")
                        else:
                            cur_attr = cur_attr.__getattr__(attr_name)
                        latest_attr_name = ''
                    except Exception as e:
                        if latest_attr_name != '':
                            latest_attr_name = f"{latest_attr_name}_{attr_name}"
                        else:
                            latest_attr_name = attr_name

            w = cur_attr.weight
            up_w = v_lora
            down_w = tensors[k_lora.replace('.lora_up.', '.lora_down.')]
            # print(down_w.shape, up_w.shape, w.shape)
            try:
                alpha_key = k_lora_name + '.alpha'
                alpha_w = tensors[alpha_key]
                wight = alpha_w / up_w.shape[1]
                # print(alpha_w, wight)
            except Exception as e:
                wight = 1
            
            einsum_a = f"ijabcdefg"
            einsum_b = f"jkabcdefg"
            einsum_res = f"ikabcdefg"
            length_shape = len(up_w.shape)
            einsum_str = f"{einsum_a[:length_shape]},{einsum_b[:length_shape]}->{einsum_res[:length_shape]}"
            d_w = torch.einsum(einsum_str, up_w, down_w)
            # print(d_w.shape, wight)

            # wight = 1
            cur_attr.weight.data = cur_attr.weight.data + d_w * wight * lora_weight
            # print('================================= add')
        
        print(f"{lora_path}-{lora_weight} is successfully added.")

    return pipe


def add_lora_bin(lora_path, pipe, lora_weight=1.0):

    if isinstance(lora_path, str):
        lora_path_list = [lora_path]
    elif isinstance(lora_path, list):
        lora_path_list = lora_path
    else:
        print(f"lora_path must be str or list[str]. Pipeline has not been modified!")
        return pipe

    if isinstance(lora_weight, float) or isinstance(lora_weight, int):
        lora_weight_list = [lora_weight]
    elif isinstance(lora_weight, list):
        lora_weight_list = lora_weight
    else:
        print(f"lora_weight must be a number or list[number]. Pipeline has not been modified!")
        return pipe

    if len(lora_weight_list) == 1:
        lora_weight_list = lora_weight_list * len(lora_path_list)
    if len(lora_weight_list) != len(lora_path_list):
        print(f"The numbers of lora_weight and lora_path does not match. Pipeline has not been modified!")
        return pipe

    for lora_path, lora_weight in zip(lora_path_list, lora_weight_list):
        tensors = torch.load(lora_path, map_location=pipe.device)
        unet = pipe.unet 

        for k_lora, v_lora in tensors.items():
            if "_lora.down" in k_lora:
                continue
            k_lora_name = "".join(k_lora.split(".processor"))
            k_lora_name = k_lora_name.replace("_lora.up.weight", "")
            attr_name_list = k_lora_name.split('.')

            cur_attr = unet
            latest_attr_name = ''

            for idx in range(len(attr_name_list)):
                attr_name = attr_name_list[idx]
                if is_int(attr_name):
                    cur_attr = cur_attr[int(attr_name)]
                    latest_attr_name = ''
                else:
                    try:
                        if latest_attr_name != '':
                            cur_attr = cur_attr.__getattr__(f"{latest_attr_name}_{attr_name}")
                        else:
                            cur_attr = cur_attr.__getattr__(attr_name)
                        latest_attr_name = ''
                    except Exception as e:
                        if latest_attr_name != '':
                            latest_attr_name = f"{latest_attr_name}_{attr_name}"
                        else:
                            latest_attr_name = attr_name
            
            up_w = v_lora
            down_w = tensors[k_lora.replace('_lora.up.', '_lora.down.')]
            einsum_a = f"ijabcdefg"
            einsum_b = f"jkabcdefg"
            einsum_res = f"ikabcdefg"
            length_shape = len(up_w.shape)
            einsum_str = f"{einsum_a[:length_shape]},{einsum_b[:length_shape]}->{einsum_res[:length_shape]}"
            d_w = torch.einsum(einsum_str, up_w, down_w)

            try:
                cur_attr.weight.data = cur_attr.weight.data + d_w * lora_weight
            except Exception as e:
                cur_attr[0].weight.data = cur_attr[0].weight.data + d_w * lora_weight
        
        print(f"{lora_path}-{lora_weight} is successfully added.")

    return pipe


def add_lora_unet_and_text_encoder(lora_file, pipe, lora_config_file, lora_weight=1.):

    lora = torch.load(lora_file, map_location='cpu')
    lora_cfg = load_json(lora_config_file)

    # 遍历lora的所有权重
    for k_lora, v_lora in lora.items():
        if k_lora.startswith('text_encoder_model'):
            model = pipe.text_encoder
            lora_alpha = lora_cfg['text_encoder_peft_config']['lora_alpha']
            # continue
        elif k_lora.startswith('model'):
            model = pipe.unet
            lora_alpha = lora_cfg['peft_config']['lora_alpha']
        else:
            print(k_lora)
    
        # down 跳过
        if '.lora_B.' in k_lora:
            continue

        # 定位到层
        cur_attr = model
        attr_name_list = k_lora.split('.')
        for idx in range(1, len(attr_name_list)-2):
            attr_name = attr_name_list[idx]
            if is_int(attr_name):
                cur_attr = cur_attr[int(attr_name)]
            else:
                cur_attr = cur_attr.__getattr__(attr_name)
        device = cur_attr.weight.data.device
        dtype = cur_attr.weight.data.dtype

        # 获取 lora up and down
        down_w = v_lora.to(device=device, dtype=dtype)
        up_w = lora[k_lora.replace('.lora_A.', '.lora_B.')].to(device=device, dtype=dtype)
        einsum_a = f"ijabcdefg"
        einsum_b = f"jkabcdefg"
        einsum_res = f"ikabcdefg"
        length_shape = len(up_w.shape)
        einsum_str = f"{einsum_a[:length_shape]},{einsum_b[:length_shape]}->{einsum_res[:length_shape]}"
        d_w = torch.einsum(einsum_str, up_w, down_w)

        wight = lora_alpha / up_w.shape[1]
        cur_attr.weight.data = cur_attr.weight.data + d_w * wight * lora_weight
        # print('================================= add')

    return pipe



add_lora = add_lora_weiUI




