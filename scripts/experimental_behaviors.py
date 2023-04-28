import gradio as gr
import torch

import modules.scripts as scripts
from modules import devices, script_callbacks, sd_samplers, shared
from scripts.script_ext.disable_mean import DisableMean
from scripts.script_ext.latent_cpu import LatentCPU
from scripts.script_ext.dual_denoise import DualDenoise

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


class SkipSteps:
    def clean_stop(self, params: script_callbacks.CFGDenoiserParams):
        if params.sampling_step == self.stop_at:
            def new_combine_denoised(x_out, conds_list, uncond,
                                     cond_scale):
                return x_out[0:uncond.shape[0]]
            self.sampler.model_wrap_cfg.combine_denoised \
                = new_combine_denoised

    def __init__(self):
        self.orig_create_sampler = sd_samplers.create_sampler

    def ui(self, is_img2img):
        return [gr.Slider(minimum=0, maximum=20, step=1, value=0,
                          label="Final sampling steps to skip")]

    def process(self, p, skip_steps):
        if skip_steps is None or skip_steps < 1:
            return

        script_callbacks.on_cfg_denoised(self.clean_stop)

        # off-by-one
        self.stop_at = p.steps - skip_steps - 1
        self.current_create_sampler = sd_samplers.create_sampler

        def new_create_sampler(name, model):
            sampler = self.current_create_sampler(name, model)
            sampler.stop_at = self.stop_at
            self.sampler = sampler
            return sampler

        sd_samplers.create_sampler = new_create_sampler

    def postprocess(self, p, processed, skip_steps):
        sd_samplers.create_sampler = self.orig_create_sampler
        script_callbacks.remove_callbacks_for_function(self.clean_stop)


class WarpClip:
    def __init__(self):
        self.clip_backup = None

    POS_IDS = {
        'Unmodified': None,
        'Ruined':
        [[0,  0,  1,  2,  3,  5,  5,  7,  7,  9, 10, 10, 11, 12,
          14, 15, 15, 16, 18, 18, 20, 20, 21, 23, 23, 25, 25, 26,
          28, 28, 30, 30, 31, 33, 33, 35, 36, 36, 37, 38, 40, 41,
          41, 42, 43, 45, 46, 46, 47, 48, 50, 51, 51, 52, 53, 55,
          56, 56, 57, 58, 60, 61, 61, 62, 63, 65, 66, 66, 67, 68,
          70, 71, 72, 72, 73, 75, 75]],
        'Repaired':
        [list(range(77))]
    }

    def ui(self, is_img2img):
        return [gr.Dropdown(list(WarpClip.POS_IDS),
                           label="Modify CLIP position_ids",
                           value="Unmodified")]

    def process(self, p, pos_ids_mod):
        if pos_ids_mod is None or pos_ids_mod == next(iter(WarpClip.POS_IDS)) \
                or pos_ids_mod == 0:
            return

        self.clip_backup = shared.sd_model.cond_stage_model.wrapped.transformer.text_model.embeddings.position_ids
        device = 'cpu' if shared.cmd_opts.lowvram or shared.cmd_opts.medvram else devices.get_optimal_device_name()

        shared.sd_model.cond_stage_model.wrapped.transformer.text_model.embeddings.position_ids \
            = torch.as_tensor(WarpClip.POS_IDS[pos_ids_mod], device=device)

    def postprocess(self, p, processed, pos_ids_mod):
        if self.clip_backup is not None:
            shared.sd_model.cond_stage_model.wrapped.transformer.text_model.embeddings.position_ids \
                = self.clip_backup
            self.clip_backup = None


experiments = [DenoiseDest, DisableMean, DualDenoise, LatentCPU, SkipSteps, WarpClip]


class Script(scripts.Script):
    def __init__(self):
        self.experiments = [init() for init in experiments]

    def title(self):
        return "Experimental Behaviors"

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False):
            elements = []
            for experiment in self.experiments:
                elements += experiment.ui(is_img2img)
            return elements

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        for i, experiment in enumerate(self.experiments):
            self.experiments[i].process(p, args[i])

    def postprocess(self, p, processed, *args):
        for i, experiment in enumerate(self.experiments):
            self.experiments[i].postprocess(p, processed, args[i])
