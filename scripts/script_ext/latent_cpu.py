import torch
import gradio as gr

from modules import devices


class LatentCPU:
    def __init__(self):
        self.orig_randn = devices.randn
        self.orig_randn_without_seed = devices.randn_without_seed

    def ui(self, is_img2img):
        latent_cpu = gr.Checkbox(label="Generate initial latent on CPU")
        return [latent_cpu]

    def process(self, p, latent_cpu):
        if latent_cpu is None or not latent_cpu:
            return

        devices.randn = lambda seed, shape: (torch.manual_seed(seed), torch.randn(shape, device='cpu'))[1]
        devices.randn_without_seed = lambda shape: torch.randn(shape, device='cpu')

    def postprocess(self, p, *args, **kwargs):
        devices.randn = self.orig_randn
        devices.randn_without_seed = self.orig_randn_without_seed
