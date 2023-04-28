import gradio as gr
import torch

from modules import devices
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase


class DisableMean:
    def __init__(self):
        self.orig_process_tokens = FrozenCLIPEmbedderWithCustomWordsBase.process_tokens

    def ui(self, is_img2img):
        disable_mean = gr.Checkbox(False, label="Disable process_tokens mean restoration")
        return [disable_mean]

    def process(self, p, disable_mean):
        if not disable_mean:
            return

        FrozenCLIPEmbedderWithCustomWordsBase.process_tokens = bypass_process_tokens

    def postprocess(self, p, processed, disable_mean):
        FrozenCLIPEmbedderWithCustomWordsBase.process_tokens = self.orig_process_tokens


def bypass_process_tokens(self, remade_batch_tokens, batch_multipliers):
    tokens = torch.asarray(remade_batch_tokens).to(devices.device)
    if self.id_end != self.id_pad:
        for batch_pos in range(len(remade_batch_tokens)):
            index = remade_batch_tokens[batch_pos].index(self.id_end)
            tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad
    z = self.encode_with_transformers(tokens)
    batch_multipliers = torch.asarray(batch_multipliers).to(devices.device)
    z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
    return z