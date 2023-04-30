import gradio as gr

from modules import shared


class ReverseCFG:
    def ui(self, is_img2img):
        return [gr.Checkbox(label="Swap prompts and negate CFG")]

    def process(self, p, reverse_cfg):
        if reverse_cfg is None or not reverse_cfg:
            return

        p.cfg_scale = -p.cfg_scale
        tmp = p.prompt
        p.prompt = p.negative_prompt
        p.negative_prompt = tmp

        if type(p.prompt) == list:
            p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
        else:
            p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

        if type(p.negative_prompt) == list:
            p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]
