import gradio as gr

from modules import scripts
from scripts.script_ext.denoise_dest import DenoiseDest
from scripts.script_ext.disable_mean import DisableMean
from scripts.script_ext.image_flip import ImageFlip
from scripts.script_ext.latent_cpu import LatentCPU
from scripts.script_ext.prompt_enhance import PromptEnhance
from scripts.script_ext.random_tokens import RandomTokens
from scripts.script_ext.reverse_cfg import ReverseCFG
from scripts.script_ext.skip_steps import SkipSteps
from scripts.script_ext.warp_clip import WarpClip

experiments = [DenoiseDest, DisableMean, ImageFlip, LatentCPU, PromptEnhance,
               RandomTokens, ReverseCFG, SkipSteps, WarpClip]

class Script(scripts.Script):
    def __init__(self):
        self.experiments = [init() for init in experiments]

    def title(self):
        return "Experimental Behaviors"

    def ui(self, is_img2img):
        with gr.Group(elem_id="experimental_behaviors"):
            with gr.Accordion(self.title(), open=False):
                elements = []
                for experiment in self.experiments:
                    current = experiment.ui(is_img2img)
                    elements += current
                    experiment.num_elements = len(current)
                return elements

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run_hook(self, hook, p, *args, extra=None, **kwargs):
        count = 0
        for i, experiment in enumerate(self.experiments):
            num_elements = experiment.num_elements
            func = getattr(experiment, hook, None)
            if func and num_elements > 0:
                if extra:
                    func(p, extra, *args[count:count + num_elements], **kwargs)
                else:
                    func(p, *args[count:count + num_elements], **kwargs)
            count += num_elements

    def process(self, p, *args, **kwargs):
        self.run_hook("process", p, *args, **kwargs)

    def before_process_batch(self, p, *args, **kwargs):
        self.run_hook("before_process_batch", p, *args, **kwargs)

    def process_batch(self, p, *args, **kwargs):
        self.run_hook("process_batch", p, *args, **kwargs)

    def postprocess_batch(self, p, *args, **kwargs):
        self.run_hook("postprocess_batch", p, *args, **kwargs)

    def postprocess(self, p, *args, **kwargs):
        self.run_hook("postprocess", p, *args, **kwargs)

    def postprocess_image(self, p, pp, *args, **kwargs):
        self.run_hook("postprocess_image", p, extra=pp, *args, **kwargs)
