import gradio as gr
import numpy as np
from PIL import Image

from modules import masking, script_callbacks


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
        if p.paste_to:
            p.image_mask = p.image_mask.transpose(Image.FLIP_TOP_BOTTOM)
            crop_region = masking.get_crop_region(np.array(p.image_mask), p.inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, p.image_mask.width, p.image_mask.height)
            x1, y1, x2, y2 = crop_region
            p.paste_to = (x1, y1, x2-x1, y2-y1)

    def postprocess_image(self, p, pp, flip, **kwargs):
        if (flip is None or not flip):
            return

        pp.image = pp.image.transpose(Image.FLIP_TOP_BOTTOM)
