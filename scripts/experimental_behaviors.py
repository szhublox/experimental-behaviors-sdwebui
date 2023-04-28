import gradio as gr
import torch

import modules.scripts as scripts
from modules import devices, sd_samplers, shared
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase


class DenoiseDest:
    def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(x_out[0:uncond.shape[0]])

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) \
                    * (weight * cond_scale)

        return denoised

    def __init__(self):
        self.orig_create_sampler = sd_samplers.create_sampler

    def ui(self, is_img2img):
        return [gr.Checkbox(False, label="Reverse denoising destination")]

    def process(self, p, reverse_denoise):
        if reverse_denoise is None or not reverse_denoise:
            return

        self.current_create_sampler = sd_samplers.create_sampler

        def new_create_sampler(name, model):
            sampler = self.current_create_sampler(name, model)
            sampler.model_wrap_cfg.combine_denoised \
                = DenoiseDest.new_combine_denoised
            return sampler

        sd_samplers.create_sampler = new_create_sampler

    def postprocess(self, p, processed, reverse_denoise):
        sd_samplers.create_sampler = self.orig_create_sampler


class DisableMean:
    def process_tokens(remade_batch_tokens, batch_multipliers):
        tokens = torch.asarray(remade_batch_tokens).to(devices.device)
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad
        z = self.encode_with_transformers(tokens)
        batch_multipliers = torch.asarray(batch_multipliers).to(devices.device)
        z = z * batch_multipliers.reshape(
            batch_multipliers.shape + (1,)).expand(z.shape)
        return z

    def __init__(self):
        self.orig_process_tokens \
            = FrozenCLIPEmbedderWithCustomWordsBase.process_tokens

    def ui(self, is_img2img):
        return [gr.Checkbox(
            False, label="Disable process_tokens mean restoration")]

    def process(self, p, disable_mean):
        if disable_mean or not disable_mean:
            return

        FrozenCLIPEmbedderWithCustomWordsBase.process_tokens \
            = DisableMean.process_tokens

    def postprocess(self, p, processed, disable_mean):
        FrozenCLIPEmbedderWithCustomWordsBase.process_tokens \
            = self.orig_process_tokens


class LatentCPU:
    def cpu_randn(seed, shape):
        torch.manual_seed(seed)
        return torch.randn(shape, device='cpu')

    def cpu_randn_without_seed(shape):
        return torch.randn(shape, device='cpu')

    def __init__(self):
        self.orig_randn = devices.randn
        self.orig_randn_without_seed = devices.randn_without_seed

    def ui(self, is_img2img):
        return [gr.Checkbox(False, label="Generate initial latent on CPU")]

    def process(self, p, latent_cpu):
        if latent_cpu is None or not latent_cpu:
            return

        devices.randn = LatentCPU.cpu_randn
        devices.randn_without_seed = LatentCPU.cpu_randn_without_seed

    def postprocess(self, p, processed, latent_cpu):
        devices.randn = self.orig_randn
        devices.randn_without_seed = self.orig_randn_without_seed


class SkipSteps:
    def __init__(self):
        self.orig_create_sampler = sd_samplers.create_sampler

    def ui(self, is_img2img):
        return [gr.Slider(minimum=0, maximum=20, step=1, value=0,
                          label="Final sampling steps to skip")]

    def process(self, p, skip_steps):
        if skip_steps is None or skip_steps < 1:
            return

        self.current_create_sampler = sd_samplers.create_sampler

        def new_create_sampler(name, model):
            sampler = self.current_create_sampler(name, model)
            # off-by-one
            sampler.stop_at = p.steps - skip_steps - 1
            return sampler

        sd_samplers.create_sampler = new_create_sampler

    def postprocess(self, p, processed, skip_steps):
        sd_samplers.create_sampler = self.orig_create_sampler


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
        if pos_ids_mod is None or pos_ids_mod == next(iter(WarpClip.POS_IDS)):
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


class Script(scripts.Script):
    def __init__(self):
        self.modules = {}
        self.modules['denoise_dest'] = DenoiseDest()
        self.modules['disable_mean'] = DisableMean()
        self.modules['latent_cpu'] = LatentCPU()
        self.modules['skip_steps'] = SkipSteps()
        self.modules['warp_clip'] = WarpClip()

    def title(self):
        return "Experimental Behaviors"

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False):
            elements = []
            for module in self.modules.values():
                elements += module.ui(is_img2img)
            return elements

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, reverse_denoise, disable_mean, latent_cpu,
                skip_steps, pos_ids_mod):
        self.modules['denoise_dest'].process(p, reverse_denoise)
        self.modules['disable_mean'].process(p, disable_mean)
        self.modules['latent_cpu'].process(p, latent_cpu)
        self.modules['skip_steps'].process(p, skip_steps)
        self.modules['warp_clip'].process(p, pos_ids_mod)

    def postprocess(self, p, processed, reverse_denoise, disable_mean,
                    latent_cpu, skip_steps, pos_ids_mod):
        self.modules['denoise_dest'].postprocess(p, processed, reverse_denoise)
        self.modules['disable_mean'].postprocess(p, processed, disable_mean)
        self.modules['latent_cpu'].postprocess(p, processed, latent_cpu)
        self.modules['skip_steps'].postprocess(p, processed, skip_steps)
        self.modules['warp_clip'].postprocess(p, processed, pos_ids_mod)
