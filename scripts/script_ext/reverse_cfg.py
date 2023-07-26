import gradio as gr

from modules import script_callbacks


class ReverseCFG:
    def ui(self, is_img2img):
        with gr.Row(variant='compact') as tab_enable:
            enabled = gr.Checkbox(label="Swap prompts and negate CFG")
            zero_weight = gr.Slider(minimum=0.0, maximum=5.0, step=0.1,
                                    value=0, label="Multiply step 0 weight")
        return [enabled, zero_weight]

    def do_swap(self, params):
        if self.zero_weight > 0 and params.sampling_step == 0:
            self.cond = params.text_cond
            params.text_cond *= self.zero_weight
        elif self.zero_weight > 0 and params.sampling_step == 1:
            params.text_cond = self.cond

        if params.sampling_step == 0:
            return

        self.p.cfg_scale = -self.p.cfg_scale
        tmp = params.text_cond
        params.text_cond = params.text_uncond
        params.text_uncond = tmp
        script_callbacks.remove_callbacks_for_function(self.do_swap)

    def process_batch(self, p, reverse_cfg, zero_weight, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        self.p = p
        self.zero_weight = zero_weight
        script_callbacks.on_cfg_denoiser(self.do_swap)

    def postprocess_batch(self, p, reverse_cfg, zero_weight, **kwargs):
        if reverse_cfg is None or not reverse_cfg:
            return

        p.cfg_scale = -p.cfg_scale
