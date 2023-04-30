import gradio as gr

from modules import scripts
from scripts.script_ext.denoise_dest import DenoiseDest
from scripts.script_ext.disable_mean import DisableMean
from scripts.script_ext.latent_cpu import LatentCPU
from scripts.script_ext.reverse_cfg import ReverseCFG
from scripts.script_ext.skip_steps import SkipSteps
from scripts.script_ext.warp_clip import WarpClip

experiments = [DenoiseDest, DisableMean, LatentCPU, ReverseCFG, SkipSteps,
               WarpClip]

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

    def process(self, p, *args):
        count = 0
        for i, experiment in enumerate(self.experiments):
            num_elements = experiment.num_elements
            experiment.process(p, *args[count:count + num_elements])
            count += num_elements

    def postprocess(self, p, processed, *args):
        count = 0
        for i, experiment in enumerate(self.experiments):
            num_elements = experiment.num_elements
            if hasattr(experiment, "postprocess"):
                experiment.postprocess(
                    p, processed, *args[count:count + num_elements])
            count += num_elements
