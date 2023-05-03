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

    def new_create_infotext(self, p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0):
        tmp_pos = p.all_prompts
        tmp_neg = p.all_negative_prompts
        p.all_negative_prompts = all_prompts

        index = position_in_batch + iteration * p.batch_size
        prompt_backup = p.all_negative_prompts[index]
        replace_str = self.all_nets if f"{self.all_nets}, " not in p.all_negative_prompts[index] else f"{self.all_nets}, "
        p.all_negative_prompts[index] = p.all_negative_prompts[index].replace(replace_str, "")

        info = self.orig_create_infotext(p, tmp_neg, all_seeds, all_subseeds, comments, iteration, position_in_batch)
        if p.cfg_scale <= 0:
            info = info.replace("CFG scale: -", "CFG scale: ")
        else:
            info = info.replace("CFG scale: ", "CFG scale: -")

        p.all_negative_prompts[index] = prompt_backup
        if ", " in replace_str:
            p.all_negative_prompts[index] = p.all_negative_prompts[index].replace(", ", "", 1)

        p.all_prompts = tmp_pos
        p.all_negative_prompts = tmp_neg
        return info

    def __init__(self):
        self.orig_create_infotext = processing.create_infotext

    def ui(self, is_img2img):
        return [gr.Checkbox(label="Swap prompts and negate CFG")]

    def process(self, p, reverse_cfg):
        if reverse_cfg is None or not reverse_cfg:
            return

        processing.create_infotext = self.new_create_infotext

        p.cfg_scale = -p.cfg_scale

        p.negative_prompt = self.swap_extra_networks(p.prompt, p.negative_prompt)

        if type(p.negative_prompt) == list:
            p.all_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
        else:
            p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

        if type(p.prompt) == list:
            p.all_negative_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
        else:
            p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    def postprocess(self, p, processed, reverse_cfg):
        if reverse_cfg is None or not reverse_cfg:
            return

        processing.create_infotext = self.orig_create_infotext
