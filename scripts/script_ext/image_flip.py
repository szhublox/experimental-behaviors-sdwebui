import gradio as gr
from PIL import Image
import torch

from modules import script_callbacks


class ImageFlip:
    def init(self):
        self.postprocessed = False

    def ui(self, is_img2img):
        if not is_img2img:
            return []
        return [gr.Checkbox(label="Vertical Flip")]

    def process(self, p, flip, **kwargs):
        if (flip is None or not flip):
            return

        self.postprocessed = False
        for index, image in enumerate(p.init_images):
            p.init_images[index] = p.init_images[index].transpose(Image.FLIP_TOP_BOTTOM)
        if p.image_mask:
            p.image_mask = p.image_mask.transpose(Image.FLIP_TOP_BOTTOM)

    def postprocess_batch(self, p, flip, **kwargs):
        if (flip is None or not flip or self.postprocessed):
            return

        self.postprocessed = True
        for i, overlay in enumerate(p.overlay_images):
            p.overlay_images[i] = overlay.transpose(Image.FLIP_TOP_BOTTOM)

    def postprocess_image(self, p, pp, flip, **kwargs):
        if (flip is None or not flip):
            return

        pp.image = pp.image.transpose(Image.FLIP_TOP_BOTTOM)
