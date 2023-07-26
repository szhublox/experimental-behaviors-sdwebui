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

    def swap_extra_networks(self, src, dst, z_weight):
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

        if (z_weight > 0.0):
            dst = f"[[{dst}:({self.remove_extra_networks(src)}:{z_weight}):0]:{dst}:1]"

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
            new_method = gr.Checkbox(label="Use cleaner coded version")
        zero_weight = gr.Slider(minimum=0.0, maximum=5.0, step=0.5, value=0, label="Unreverse Negative prompt at step 0 with weight:")
        return [enabled, new_method, zero_weight]

    def process(self, p, reverse_cfg, new_method, zero_weight, **kwargs):
        if reverse_cfg is None or not reverse_cfg or new_method:
            return

        self.orig_all_prompts = p.all_prompts
        self.orig_all_negative_prompts = p.all_negative_prompts

        p.negative_prompt = self.swap_extra_networks(p.prompt, p.negative_prompt, zero_weight)
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

    def do_swap(self, params):
        if len(params.text_cond) > 1:
            # bail, it won't work
            self.orig_cfg_scale = self.p.cfg_scale
            script_callbacks.remove_callbacks_for_function(self.do_swap)

        if self.zero_weight > 0:
            if params.sampling_step == 0:
                self.cond = params.text_cond
                params.text_cond *= self.zero_weight
            if params.sampling_step == 1:
                params.text_cond = self.cond

        if params.sampling_step == 0:
            return

        # attempt to keep AND from crashing. didn't work
        # params.text_cond = params.text_cond.mean(0).unsqueeze(0)
        self.orig_cfg_scale = self.p.cfg_scale
        self.p.cfg_scale = -self.p.cfg_scale
        tmp = params.text_cond
        params.text_cond = params.text_uncond
        params.text_uncond = tmp
        script_callbacks.remove_callbacks_for_function(self.do_swap)

    def process_batch(self, p, reverse_cfg, new_method, zero_weight, **kwargs):
        if reverse_cfg is None or not reverse_cfg or not new_method:
            return

        self.p = p
        self.zero_weight = zero_weight
        script_callbacks.on_cfg_denoiser(self.do_swap)

    def before_process_batch(self, p, reverse_cfg, new_method, zero_weight, **kwargs):
        if reverse_cfg is None or not reverse_cfg or new_method:
            return

        self.orig_cfg_scale = p.cfg_scale
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

    def postprocess_batch(self, p, reverse_cfg, new_method, zero_weight, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        p.cfg_scale = self.orig_cfg_scale

        if not new_method:
            p.all_prompts = self.orig_all_prompts
            p.all_negative_prompts = self.orig_all_negative_prompts

    def postprocess(self, p, *args, **kwargs):
        script_callbacks.remove_callbacks_for_function(self.filename_callback)
