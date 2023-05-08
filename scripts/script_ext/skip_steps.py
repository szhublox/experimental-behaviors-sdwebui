import gradio as gr

from modules import script_callbacks, sd_samplers


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

        # off-by-one
        self.stop_at = p.steps - skip_steps - 1

    def process_batch(self, p, skip_steps, **kwargs):
        if skip_steps is None or skip_steps < 1:
            return

        self.current_create_sampler = sd_samplers.create_sampler
        script_callbacks.on_cfg_denoised(self.clean_stop)

        def new_create_sampler(name, model):
            sampler = self.current_create_sampler(name, model)
            sampler.stop_at = self.stop_at
            self.sampler = sampler
            return sampler

        sd_samplers.create_sampler = new_create_sampler

    def postprocess_batch(self, p, skip_steps, **kwargs):
        sd_samplers.create_sampler = self.orig_create_sampler
        script_callbacks.remove_callbacks_for_function(self.clean_stop)
