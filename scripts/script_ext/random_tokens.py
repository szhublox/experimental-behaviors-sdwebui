import random

import gradio as gr
import open_clip.tokenizer
import torch

from modules import devices, sd_models

open_clip_tokenizer = open_clip.tokenizer._tokenizer


class RandomTokens:
    def ui(self, is_img2img):
        return [gr.Slider(minimum=0, maximum=75, step=1, value=0,
                          label="Append random tokens to prompt")]

    def process(self, p, random_tokens):
        if random_tokens is None or random_tokens < 1:
            return

        wrapped = sd_models.model_data.sd_model.cond_stage_model.wrapped
        our_tokenizer = getattr(wrapped, "tokenizer", open_clip_tokenizer)

        for i, prompt in enumerate(p.all_prompts):
            p.all_prompts[i] += " " + our_tokenizer.decode(random.sample(
                range(our_tokenizer.vocab_size), random_tokens))
