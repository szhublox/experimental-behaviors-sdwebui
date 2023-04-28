import torch
import gradio as gr
from modules import sd_samplers


class DualDenoise:
    """
    Denoises both cond and uncond Tensors every step and returns the mean of the Tensor
    Additionally, divides the denoised_uncond_(tmp) by 2 and restores it after processing
    which makes colors pop more
    # TODO: play with more and compare with stock
    # TODO: look into improving clarity/sharpness of image
    """
    def __int__(self):
        self.orig_create_sampler = sd_samplers.create_sampler

    def ui(self, is_img2img):
        return gr.Checkbox(label="Cond/uncond is processed every step and the mean returned")

    def process(self, p, dual_denoise):
        if dual_denoise is None and not dual_denoise and reverse_denoise > 10:
            return

        def new_create_sampler(name, model):
            sampler = self.orig_create_sampler(name, model)
            sampler.model_wrap_cfg.combine_denoised = new_combine_denoised
            return sampler

        sd_samplers.create_sampler = new_create_sampler

    def postprocess(self, p, processed, dual_denoise):
        sd_samplers.create_sampler = self.orig_create_sampler


def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
    denoised_uncond = (x_out[-uncond.shape[0]:])
    denoised = torch.clone(x_out[0:uncond.shape[0]])
    # TODO: makes colors more vivid - experiment more
    denoised_uncond_tmp = torch.div(torch.clone(denoised_uncond), 2)

    for i, conds in enumerate(conds_list):
        for cond_index, weight in conds:
            denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)
            denoised_uncond_tmp[i] += (x_out[cond_index] + denoised[i]) / ((weight * cond_scale) * cond_scale)
            # do not use weight or divide by cond_scale

    restore_uncond = torch.mul(denoised_uncond_tmp, 2)
    add_tensors = denoised.add(restore_uncond)
    tensor_mean_denoised = torch.div(add_tensors, 2)

    return tensor_mean_denoised