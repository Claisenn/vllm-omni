# LTX-2 Text-to-Video Generation

This example demonstrates how to use Lightricks LTX-2 model for text-to-video generation with vLLM-Omni.

## Model Information

LTX-2 is a state-of-the-art text-to-video model by Lightricks that can generate high-quality videos from text prompts. It supports:
- Text-to-video generation
- Image-to-video generation (conditioned on an input image)
- High-resolution video generation (up to 4K)
- Synchronized audio-video generation

For more information, see:
- [Lightricks/LTX-2 on HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [LTX-2 GitHub Repository](https://github.com/Lightricks/LTX-2)

## Requirements

- Python >= 3.10
- diffusers >= 0.36.0 (already included in vllm-omni dependencies)
- GPU with sufficient VRAM (recommended: >= 24GB for high-quality generation)

## Usage

### Basic Text-to-Video Generation

```bash
python text_to_video.py \
    --model Lightricks/LTX-2 \
    --prompt "A panda riding a bicycle through a forest, cinematic lighting" \
    --height 512 \
    --width 768 \
    --num_frames 121 \
    --num_inference_steps 40 \
    --guidance_scale 4.0 \
    --output output.mp4
```

### Parameters

- `--model`: Model ID or local path (default: `Lightricks/LTX-2`)
- `--prompt`: Text description of the video to generate
- `--negative_prompt`: Text describing what to avoid in the video
- `--height`: Video height in pixels, must be divisible by 32 (default: 512)
- `--width`: Video width in pixels, must be divisible by 32 (default: 768)
- `--num_frames`: Number of frames, should be 8*n+1 (e.g., 25, 81, 121) (default: 121)
- `--num_inference_steps`: Number of denoising steps (default: 40)
- `--guidance_scale`: Classifier-free guidance scale (default: 4.0)
- `--seed`: Random seed for reproducibility (default: 42)
- `--fps`: Output video frames per second (default: 24)
- `--output`: Output video file path (default: ltx2_output.mp4)

### Example Prompts

```bash
# Cinematic scene
python text_to_video.py --prompt "A serene lakeside sunrise with mist over the water, cinematic"

# Action scene
python text_to_video.py --prompt "A futuristic cityscape with flying cars, neon lights, cyberpunk style"

# Nature scene
python text_to_video.py --prompt "A waterfall in a lush rainforest, birds flying overhead"
```

## Performance Tips

1. **Resolution**: Start with lower resolutions (512x768) for faster generation. Higher resolutions require more VRAM.
2. **Frames**: More frames take longer to generate. Start with 25 or 81 frames for testing.
3. **Inference Steps**: 40 steps provide good quality. Fewer steps are faster but may reduce quality.
4. **Guidance Scale**: Values between 3.0-5.0 typically work well. Higher values follow the prompt more strictly.

## Notes

- The first run will download the model from HuggingFace (approximately 20-30GB).
- Generation time depends on your GPU and the parameters chosen.
- For best results, use descriptive prompts with style keywords (e.g., "cinematic", "high quality", "detailed").
