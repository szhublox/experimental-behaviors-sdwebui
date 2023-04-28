import gradio as gr
import torch

from modules import shared, devices


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
