# LTX-2 Model Integration

## Overview

This integration adds support for the Lightricks LTX-2 text-to-video generation model to vllm-omni. LTX-2 is a state-of-the-art open-source model capable of generating high-quality videos (up to 4K resolution) from text prompts or images.

## Model Information

- **Model ID**: `Lightricks/LTX-2`
- **Model Type**: Text-to-Video (T2V) and Image-to-Video (I2V)
- **Architecture**: Diffusion Transformer (DiT) with 19B parameters
- **HuggingFace**: https://huggingface.co/Lightricks/LTX-2
- **GitHub**: https://github.com/Lightricks/LTX-2

## Quick Start

### Basic Usage

```python
from vllm_omni.entrypoints.omni import Omni

# Initialize the model
omni = Omni(model="Lightricks/LTX-2")

# Generate a video
output = omni.generate(
    prompt="A panda riding a bicycle through a forest, cinematic lighting",
    height=512,
    width=768,
    num_frames=121,
    num_inference_steps=40,
    guidance_scale=4.0,
)
```

### Using the Example Script

```bash
cd examples/offline_inference/ltx2

python text_to_video.py \
    --prompt "A serene lakeside sunrise with mist over the water" \
    --height 512 \
    --width 768 \
    --num_frames 121 \
    --output my_video.mp4
```

See the [example documentation](examples/offline_inference/ltx2/text_to_video.md) for more details.

## Parameters

### Video Dimensions
- **height**: Video height in pixels (must be divisible by 32)
- **width**: Video width in pixels (must be divisible by 32)
- Default: 512x768

### Frame Count
- **num_frames**: Number of frames (should follow pattern 8*n+1)
- Valid examples: 25, 41, 81, 121, 161
- Default: 121 frames (approximately 5 seconds at 24fps)

### Generation Quality
- **num_inference_steps**: Number of denoising steps (default: 40)
  - More steps = higher quality but slower generation
  - Range: 20-50 steps typically
  
- **guidance_scale**: Classifier-free guidance strength (default: 4.0)
  - Higher values follow prompt more strictly
  - Range: 3.0-5.0 recommended

### Other Options
- **negative_prompt**: Text describing what to avoid in the video
- **seed**: Random seed for reproducibility
- **fps**: Output video frame rate (default: 24)

## Implementation Details

### Architecture

The integration follows vllm-omni's diffusion model patterns:

```
vllm_omni/diffusion/models/ltx2/
├── __init__.py                  # Module exports
└── pipeline_ltx2.py             # Main pipeline implementation
```

### Key Components

1. **LTX2Pipeline**: Wraps diffusers' `LTXVideoPipeline`
   - Handles parameter validation
   - Ensures dimension constraints (divisible by 32)
   - Manages frame count convention (8*n+1)
   
2. **Pre-processing**: `get_ltx2_pre_process_func`
   - Loads and validates image inputs for I2V mode
   
3. **Post-processing**: `get_ltx2_post_process_func`
   - Converts video tensors to numpy arrays
   - Handles different output formats

### Registry Integration

The model is registered in `vllm_omni/diffusion/registry.py`:
- Model class: `LTX2Pipeline`
- Pre-process function: `get_ltx2_pre_process_func`
- Post-process function: `get_ltx2_post_process_func`

## Requirements

- **Python**: >= 3.10
- **GPU**: CUDA-capable GPU with >= 24GB VRAM recommended
- **Dependencies**: diffusers >= 0.36.0 (already included in vllm-omni)

### Model Download

On first use, the model will be automatically downloaded from HuggingFace:
- Size: Approximately 20-30GB
- Location: `~/.cache/huggingface/hub/`

## Performance Tips

1. **Start Small**: Test with lower resolutions (512x768) and fewer frames (25-81)
2. **Increase Gradually**: Scale up resolution and frame count based on your GPU
3. **Optimize Steps**: 40 steps provide good quality; adjust based on speed/quality needs
4. **Batch Processing**: For multiple videos, reuse the same Omni instance to avoid reloading

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
- Reduce video dimensions (e.g., 480x640 instead of 512x768)
- Reduce frame count (e.g., 25 or 41 instead of 121)
- Enable VAE tiling/slicing if supported

### Slow Generation

For faster generation:
- Reduce `num_inference_steps` (e.g., 30 instead of 40)
- Use smaller dimensions
- Generate fewer frames

### Quality Issues

For better quality:
- Increase `num_inference_steps` (e.g., 50)
- Adjust `guidance_scale` (try 3.5-5.0)
- Use more descriptive prompts with style keywords

## Example Prompts

```python
# Cinematic nature scene
"A serene lakeside sunrise with mist over the water, cinematic, high quality"

# Action scene
"A futuristic cityscape with flying cars, neon lights, cyberpunk style, 4k"

# Animated style
"A cartoon panda eating bamboo in a forest, Pixar style animation"

# Close-up shot
"Close-up of raindrops falling on a window, soft focus, moody lighting"
```

## References

- [LTX-2 on HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [LTX-2 GitHub Repository](https://github.com/Lightricks/LTX-2)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video)
- [Example Usage](examples/offline_inference/ltx2/text_to_video.md)
