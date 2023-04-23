## Add Lora



```python
from redpy.diffusers_custom import StableDiffusionCommonPipeline
from redpy.diffusers_custom import add_lora

base_model = 'base_model path'
lora_path = 'lora path'
lora_weight = 0.6

pipe = StableDiffusionCommonPipeline.from_pretrained(base_model)
pipe = add_lora(lora_path, pipe, lora_weight)
```
