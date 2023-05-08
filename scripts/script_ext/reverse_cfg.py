import re

import gradio as gr

from modules import extra_networks, processing, shared


class ReverseCFG:
    def swap_extra_networks(self, src, dst):
        def get_nets(src):
            nets = ""
            for net in re.findall(extra_networks.re_extra_net, src):
                nets += f"<{net[0]}:{net[1]}>"
            return nets

        if type(src) == list:
            self.all_nets = ""
            for x in src:
                self.all_nets += get_nets(x)
        else:
            self.all_nets = get_nets(src)

        if type(dst) == list:
            for x in dst:
                x += self.all_nets
        else:
            dst += self.all_nets

        return dst

    def ui(self, is_img2img):
        return [gr.Checkbox(label="Swap prompts and negate CFG")]

    def process(self, p, reverse_cfg, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        self.orig_all_prompts = p.all_prompts
        self.orig_all_negative_prompts = p.all_negative_prompts

        p.negative_prompt = self.swap_extra_networks(p.prompt, p.negative_prompt)

        if type(p.negative_prompt) == list:
            p.all_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
        else:
            p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

        if type(p.prompt) == list:
            p.all_negative_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

        self.backup_all_prompts = p.all_prompts
        self.backup_all_negative_prompts = p.all_negative_prompts

    def before_process_batch(self, p, reverse_cfg, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        p.cfg_scale = -p.cfg_scale
        p.all_prompts = self.backup_all_prompts
        p.all_negative_prompts = self.backup_all_negative_prompts

    def postprocess_batch(self, p, reverse_cfg, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        p.cfg_scale = -p.cfg_scale
        p.all_prompts = self.orig_all_prompts
        p.all_negative_prompts = self.orig_all_negative_prompts
