import gradio as gr
import torch

from modules import shared, devices


class WarpClip:
    def __init__(self):
        self.clip_backup = None

    def ui(self, is_img2img):
        with gr.Accordion("CLIP",open = False):
            clip_option = gr.Radio(label = "Modify CLIP position_ids",choices = ["Unmodified","Ruins","Reverse","Repaired","Uniform (slider)"], 
                                    value ="Unmodified",type = "value",interactive =True)
            clip_slider = gr.Slider(minimum=0.0, maximum=76.0, step=1, value=0, label="Set all CLIP position_ids to:")
            
        return[clip_option,clip_slider]

    def process(self, p, clip_option, clip_slider, **kwargs):
        if clip_option is None or clip_option == "Unmodified":
            return

        pos_ids_mod = None

        if clip_option == "Ruins":
            pos_ids_mod = [[0,  0,  1,  2,  3,  5,  5,  7,  7,  9, 10, 10, 11, 12,
          14, 15, 15, 16, 18, 18, 20, 20, 21, 23, 23, 25, 25, 26,
          28, 28, 30, 30, 31, 33, 33, 35, 36, 36, 37, 38, 40, 41,
          41, 42, 43, 45, 46, 46, 47, 48, 50, 51, 51, 52, 53, 55,
          56, 56, 57, 58, 60, 61, 61, 62, 63, 65, 66, 66, 67, 68,
          70, 71, 72, 72, 73, 75, 75]]
        
        if clip_option == "Reverse":
            pos_ids_mod = [list(range(76,-1,-1))]
        
        if clip_option == "Repaired":
            pos_ids_mod = [list(range(77))]
        
        if clip_option == "Uniform (slider)":
            pos_ids_mod = [[clip_slider] * 77]

        self.clip_backup = shared.sd_model.cond_stage_model.wrapped.transformer.text_model.embeddings.position_ids
        device = 'cpu' if shared.cmd_opts.lowvram or shared.cmd_opts.medvram else devices.get_optimal_device_name()

        shared.sd_model.cond_stage_model.wrapped.transformer.text_model.embeddings.position_ids \
            = torch.as_tensor(pos_ids_mod, device=device)

    def postprocess(self, p, clip_option, clip_slider, **kwargs):
        if self.clip_backup is not None:
            shared.sd_model.cond_stage_model.wrapped.transformer.text_model.embeddings.position_ids \
                = self.clip_backup
            self.clip_backup = None
