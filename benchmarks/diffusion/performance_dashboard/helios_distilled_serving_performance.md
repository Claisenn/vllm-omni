# Helios-Distilled Serving Performance Dashboard

This document describes how to deploy and benchmark **BestWishYsh/Helios-Distilled** using vLLM-Omni. It follows the same serving benchmark structure as the Qwen-Image and Wan2.2 dashboards.

---

# 1. Overview

Helios-Distilled is a text-to-video diffusion model. Helios uses an autoregressive chunking pattern that generates 33 frames per chunk. For stable comparisons, choose `num_frames` as a multiple of 33, such as 99.

This document covers:

* Service launch configuration
* Benchmark scripts and usage
* Dataset and workload settings
* Performance and accuracy comparison methodology
* Reproducibility guidelines

---

# 2. Test Environment

| Component | Specification |
|------------|----------------|
| GPU | NVIDIA H100 for CI accuracy/perf gate; local bring-up also used L20X |
| Diffusion Attention Backend | FlashAttention |
| Model | `BestWishYsh/Helios-Distilled` |

---

# 3. Service Launch Configuration

## 3.1 Basic Serving Command

```bash
vllm serve BestWishYsh/Helios-Distilled --omni \
    --port 8091
```

## 3.2 Key Parameters

| Parameter | Description |
| --- | --- |
| `--port` | Serving port |
| `--max-num-seqs` | Maximum number of active sequences |
| `--enable-diffusion-pipeline-profiler` | Optional stage-level profiling |
| `--profiler-config` | Optional torch profiler configuration |

Record these parameters when reporting performance results.

---

# 4. Benchmark Scripts

## 4.1 Online Serving Benchmark

Use the shared diffusion serving benchmark, as with Qwen-Image and Wan2.2:

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8091 \
    --endpoint /v1/videos \
    --model BestWishYsh/Helios-Distilled \
    --dataset random \
    --task t2v \
    --num-prompts 1 \
    --max-concurrency 1 \
    --disable-tqdm \
    --random-request-config '[
        {
            "width": 640,
            "height": 384,
            "num_inference_steps": 50,
            "num_frames": 99,
            "fps": 16,
            "guidance_scale": 1.0,
            "is_enable_stage2": true,
            "pyramid_num_stages": 3,
            "pyramid_num_inference_steps_list": [1, 1, 1],
            "is_amplify_first_chunk": false,
            "weight": 1
        }
    ]'
```

`--random-request-config` forwards standard fields such as `width`, `height`, `num_frames`, `fps`, and `num_inference_steps` as first-class request fields. Model-specific fields are forwarded in the request body, which is required for Helios stage-2 settings.

## 4.2 Warmed Serving Benchmark

For local bring-up, use the same shared serving benchmark with one warmup request and one measured request:

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8091 \
    --endpoint /v1/videos \
    --model BestWishYsh/Helios-Distilled \
    --dataset random \
    --task t2v \
    --num-prompts 1 \
    --max-concurrency 1 \
    --warmup-requests 1 \
    --warmup-num-inference-steps 1 \
    --random-prompt "A cat wearing sunglasses dances on a beach at sunset, cinematic lighting." \
    --seed 42 \
    --disable-tqdm \
    --output-file metrics.json \
    --save-response-dir responses \
    --random-request-config '[
        {
            "width": 640,
            "height": 384,
            "num_inference_steps": 50,
            "num_frames": 99,
            "fps": 16,
            "guidance_scale": 1.0,
            "is_enable_stage2": true,
            "pyramid_num_stages": 3,
            "pyramid_num_inference_steps_list": [1, 1, 1],
            "is_amplify_first_chunk": false,
            "weight": 1
        }
    ]'
```

## 4.3 PR-vs-Main Accuracy and Performance Comparison

The PR-vs-main accuracy/performance gate lives in the Helios accuracy pytest. It checks out `origin/main`, starts a server for each checkout, and drives both with `diffusion_benchmark_serving.py`:

```bash
pytest -s -v tests/e2e/accuracy/test_helios_distilled.py::test_helios_distilled_pr_matches_main_acc_perf \
    -m 'full_model and benchmark' \
    --run-level 'full_model'
```

---

# 5. Dataset & Workload Settings

## 5.1 Recommended Evaluation Configurations

### Dataset A (99 frames, 384x640)

* Dataset: `random`
* Task: t2v
* Concurrency: 1
* Frame count: 99 (`33 x 3`)
* FPS: 16
* Stage-2 steps: `[1, 1, 1]`

```json
[
  {
    "width": 640,
    "height": 384,
    "num_inference_steps": 50,
    "num_frames": 99,
    "fps": 16,
    "guidance_scale": 1.0,
    "is_enable_stage2": true,
    "pyramid_num_stages": 3,
    "pyramid_num_inference_steps_list": [1, 1, 1],
    "is_amplify_first_chunk": false,
    "weight": 1
  }
]
```

---

# 6. Performance Metrics

| Metric | Description | Unit |
| --- | --- | --- |
| Mean Latency | Mean request latency | seconds |
| P99 Latency | P99 request latency | seconds |
| Generated FPS | `num_frames / measured_time_s` | frames/second |
| Candidate/reference latency ratio | PR latency divided by reference latency | ratio |

---

# 7. Accuracy Metrics

The PR-vs-main gate compares encoded video frames extracted from MP4 outputs:

| Metric | Threshold | Direction |
| --- | --- | --- |
| PSNR | `>= 35 dB` | Higher is better |
| MAE | `<= 3.0 / 255` | Lower is better |
| Candidate/reference latency ratio | `<= 1.03` | Lower is better |

Local L20X bring-up for this PR produced:

| Variant | Measured Latency (ms) | Generated FPS |
| --- | --- | --- |
| `origin/main` | 5317.95 | 18.62 |
| Candidate PR | 5283.27 | 18.74 |

Encoded-video similarity:

| Metric | Value |
| --- | --- |
| PSNR | 38.66 dB |
| MAE | 1.89 / 255 |
| Cosine similarity | 0.99978 |

---

# 8. Reproducibility Checklist

To ensure consistent and comparable benchmark results:

* Record GPU type
* Record vLLM and vLLM-Omni versions
* Record benchmark parameters: resolution, frame count, FPS, seed, concurrency, and stage-2 steps
* Use frame counts that are multiples of 33
* Keep prompt, seed, resolution, and sampling parameters identical when comparing PR vs main
* Ensure no background workload on GPUs during testing
* Use the same MP4 export and frame extraction path for both variants

---

This document serves as the Helios-Distilled serving performance reference under vLLM-Omni.
