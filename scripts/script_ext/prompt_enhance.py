import re

import gradio as gr


class PromptEnhance:
    def ui(self, is_img2img):
        return [gr.Checkbox(value=True,
                            label="Enable prompt QOL enhancements")]

    def process(self, p, prompt_enhance):
        if prompt_enhance is None or not prompt_enhance:
            return

        for i, prompt in enumerate(p.all_prompts):
            p.all_prompts[i] = re.sub(r'\[([^:]+):([.0-9]+)~([.0-9]+)\]',
                                      r'[:[\1::\3]:\2]',
                                      p.all_prompts[i])
