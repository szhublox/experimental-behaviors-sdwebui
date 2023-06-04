import re

import gradio as gr

from modules import extra_networks, images, script_callbacks, shared


class ReverseCFG:
    def filename_callback(self, params):
        if not params.p:
            return

        if self.orig_all_negative_prompts[params.p.batch_index]:
            for replace_spaces in (True, False):
                params.filename = params.filename.replace(
                    images.sanitize_filename_part(self.orig_all_negative_prompts[params.p.batch_index],
                                                  replace_spaces),
                    images.sanitize_filename_part(self.orig_all_prompts[params.p.batch_index],
                                                  replace_spaces))
        else:
            ext_index = params.filename.rfind('.')
            filename = params.filename[0:ext_index]
            extension = params.filename[ext_index + 1:]
            params.filename = f"{filename}" \
                f"{images.sanitize_filename_part(self.orig_all_prompts[params.p.batch_index])}" \
                f".{extension}"

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

    def remove_extra_networks(self,src):
        s = src
        for net in re.finditer(extra_networks.re_extra_net, src):
            s = s.replace(net.group(0), '')
            
        s = re.sub(r' +', ' ', s)

        return s

    def ui(self, is_img2img):
        with gr.Row(variant='compact') as tab_enable:
            enabled = gr.Checkbox(label="Swap prompts and negate CFG")
            dirty = gr.Checkbox(label="Keep extra networks as tokens")
        return [enabled, dirty]

    def process(self, p, reverse_cfg, dirty, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        self.orig_all_prompts = p.all_prompts
        self.orig_all_negative_prompts = p.all_negative_prompts

        p.negative_prompt = self.swap_extra_networks(p.prompt, p.negative_prompt)
        if not dirty:
            p.prompt = self.remove_extra_networks(p.prompt)

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
        script_callbacks.on_before_image_saved(self.filename_callback)

    def before_process_batch(self, p, reverse_cfg, dirty, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        p.cfg_scale = -p.cfg_scale
        p.all_prompts = self.backup_all_prompts
        p.all_negative_prompts = self.backup_all_negative_prompts

        n = p.iteration
        p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
        p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
        p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
        p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

        if hasattr(p, "enable_hr") and p.enable_hr:
            p.all_hr_prompts = p.all_prompts
            p.all_hr_negative_prompts = p.all_negative_prompts
            p.hr_prompts = p.all_hr_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.hr_negative_prompts = p.all_hr_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.hr_prompts, p.hr_extra_network_data = \
                extra_networks.parse_prompts(p.hr_prompts)

    def postprocess_batch(self, p, reverse_cfg, dirt, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        p.cfg_scale = -p.cfg_scale
        p.all_prompts = self.orig_all_prompts
        p.all_negative_prompts = self.orig_all_negative_prompts

    def postprocess(self, p, *args, **kwargs):
        script_callbacks.remove_callbacks_for_function(self.filename_callback)
