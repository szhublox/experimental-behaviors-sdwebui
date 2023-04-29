# Add ways to modify image generation procedure

- Change kdiffusion denoise method
- Disable mean restore procedure at the end of process\_tokens()
- Generate initial latent on CPU instead of GPU
- Set sampler stop\_at
- Modify CLIP position\_ids inline

## Notes

**This extension is not fully compatible with the dynamic thresholding extension because they modify some of the same functions.**

The features that are not fully compatible with CompVis is a bug with this extension, not with CompVis samplers.

### Changing kdiffusion denoise

There are two options, "Reverse" and "Dual" with an accompanying slider to determine how far into generation to swap to the alternate method. Set to 0 does nothing, and set to 0.2 will set the change for the last 20% of generation. The CompVis samplers are DDIM, PLMS, and UniPC, and kdiffusion is everything else.

The reverse method simply changes what gets denoised after sampling - cond instead of uncond. This leads to cleaner but less creative images.

The dual method does the standard method, and the reverse method, and averages them.

### Disabling mean restoration

There is a comment in sd\_hijack\_clip.py: "# restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise". This has an effect on (weighted tokens:1.2).

This checkbox removes the last two lines from that function:

```
new_mean = z.mean()
z = z * (original_mean / new_mean)
```

### Generating initial latent on CPU

Generating on GPU gives different results on different cards, but generating on CPU does not. This is also available as a setting in new versions of webui.

### Setting sampler stop\_at

Force the sampler to stop too early. For example, if the slider is set to 4 while generating 28 steps, only the first 24 steps will be taken. Sometimes this is an improvement, especially depending on which checkpoint is loaded. This uses the same trick as the reverse denoise option, so using this with a CompVis sampler will not work well.

### Modifying CLIP position\_ids

Normally this should be a list of numbers from 0-76, but due to model merging, sometimes they become off by a little bit. This option allows to set them correctly, or to set them very wrong on purpose. This does not change the model on disk, only temporarily in memory.
