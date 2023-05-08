import gradio as gr
import torch

from modules import script_callbacks


class DenoiseDest:
    def __init__(self):
        self.p = None
        self.alter_fraction = None
        self.func = None

    def reverse(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(x_out[0:uncond.shape[0]])

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) \
                    * (weight * cond_scale)

        return denoised

    def dual(self, x_out, conds_list, uncond, cond_scale):
        """
        Denoises both cond and uncond Tensors every step and returns the mean of the Tensor
        """
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(x_out[0:uncond.shape[0]])
        # TODO: makes colors more vivid - experiment more
        denoised_uncond_tmp = denoised_uncond  # torch.div(torch.clone(denoised_uncond), 2)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)
                denoised_uncond_tmp[i] += (x_out[cond_index] + denoised[i]) / ((weight * cond_scale) * cond_scale)
                # do not use weight or divide by cond_scale

        # restore_uncond = torch.mul(denoised_uncond_tmp, 2)
        add_tensors = denoised.add(denoised_uncond_tmp)
        tensor_mean_denoised = torch.div(add_tensors, 2)

        return tensor_mean_denoised

    def replace_combine(self, params: script_callbacks.CFGDenoiserParams):
        if self.alter_fraction \
                <= params.sampling_step / params.total_sampling_steps:
            self.p.sampler.model_wrap_cfg.combine_denoised = self.func

    def ui(self, is_img2img):
        return [gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0,
                          label="Alter denoise for final progress"),
                gr.Radio(["Reverse", "Dual"], value="Reverse", label="Denoising method",
                         info="For dual, cond/uncond is processed every step and the mean returned")]

    def process(self, p, alter_denoise, denoise_method, **kwargs):
        if (alter_denoise is None or alter_denoise == 0.0) \
                or (denoise_method is None or denoise_method == ""):
            return

        self.alter_fraction = 1 + (-alter_denoise)

        self.p = p
        self.func = getattr(self, denoise_method.lower())

    def process_batch(self, p, alter_denoise, denoise_method, **kwargs):
        if (alter_denoise is None or alter_denoise == 0.0) \
                or (denoise_method is None or denoise_method == ""):
            return

        script_callbacks.on_cfg_denoised(self.replace_combine)

    def postprocess_batch(self, p, *args, **kwargs):
        self.__init__()
        script_callbacks.remove_callbacks_for_function(self.replace_combine)
