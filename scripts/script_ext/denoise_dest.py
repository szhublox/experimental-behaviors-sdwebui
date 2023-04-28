import gradio as gr
import torch

from modules import script_callbacks


class DenoiseDest:
    def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(x_out[0:uncond.shape[0]])

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) \
                    * (weight * cond_scale)

        return denoised

    def replace_combine(self, params: script_callbacks.CFGDenoiserParams):
        if self.reverse_fraction \
                <= params.sampling_step / params.total_sampling_steps:
            self.p.sampler.model_wrap_cfg.combine_denoised \
                = DenoiseDest.new_combine_denoised

    def ui(self, is_img2img):
        return [gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0,
                          label="Reverse denoise for final progress")]

    def process(self, p, reverse_denoise):
        if reverse_denoise is None or reverse_denoise == 0.0:
            return

        self.reverse_fraction = 1 + (-reverse_denoise)

        self.p = p
        script_callbacks.on_cfg_denoised(self.replace_combine)

    def postprocess(self, p, processed, reverse_denoise):
        script_callbacks.remove_callbacks_for_function(self.replace_combine)

    def get_reverse_denoise(self):
        return
