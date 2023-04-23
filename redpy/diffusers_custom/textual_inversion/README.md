

## Use Textual Inversion

```python
from redpy.diffusers_custom import StableDiffusionCommonPipeline
from redpy.diffusers_custom import add_textual_inversion

base_model = 'base_model path'
textual_inversion_file = 'textual_inversion path'

# load model and textual_inversion
pipe = StableDiffusionCommonPipeline.from_pretrained(base_model, safety_checker=None, feature_extractor=None)
pipe, textual_inversion_tokens = add_textual_inversion(pipe, textual_inversion_file)

# fix prompt or negative_prompt
prompt = textual_inversion_tokens[0] + f"your prompt"
negative_prompt = f"your negative_prompt"

# infer
image = pipe.text2img(prompt=prompt, negative_prompt=negative_prompt, height=768, width=512).images[0]


```