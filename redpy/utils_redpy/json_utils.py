import json
__all__ = ['load_json', 'write_json']

def write_json(x_struct: dict, json_file: str, indent=4):
    #json_str = json.dumps(x_struct,indent=2,ensure_ascii=False)
    with open(json_file, 'w') as fd:
        json.dump(x_struct, fd, indent=indent, ensure_ascii=False)

def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

