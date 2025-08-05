#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "qwen-vl-utils",
#   "rich",
#   "torch",
#   "torchcodec",
#   "torchvision",
#   "transformers>=4.51.3",
#   "pydantic",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# ///

"""Example inference using vllm.

Example:

```shell
./inference_vllm.py
```
"""

import argparse
import pydantic
from rich import print
import transformers
import vllm
import  qwen_vl_utils

class VisionConfig(pydantic.BaseModel):
    fps: int = pydantic.Field(default=1, description="FPS of the video")
    max_pixels: int = pydantic.Field(default=81920, description="Max pixels of the video")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="User prompt message")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.",
        help="System prompt message",
    )
    parser.add_argument("--images", type=str, nargs="*", help="Image paths")
    parser.add_argument("--videos", type=str, nargs="*", help="Video paths")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model name (https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)",
    )
    parser.add_argument("--vision", type=str, help="Vision config json file")
    parser.add_argument("--sampling", type=str, help="Sampling config json file")
    args = parser.parse_args()
    
    images = args.images or []
    videos = args.videos or []
    if args.vision is not None:
        vision_config = VisionConfig.model_validate_json(open(args.vision, "rb").read())
    else:
        vision_config = VisionConfig()

    llm = vllm.LLM(
        model=args.model,
        limit_mm_per_prompt={"image": len(images), "video": len(videos)},
    )

    if args.sampling is not None:
        sampling_params = vllm.SamplingParams.model_validate_json(open(args.sampling, "rb").read())
    else:
        sampling_params = vllm.SamplingParams(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=4096,
        )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ("Is it safe to turn right?")},
                {
                    "type": "video",
                    "video": "assets/sample.mp4",
                    "fps": 4,
                },
            ],
        },
    ]

    processor = transformers.AutoProcessor.from_pretrained(args.model)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        messages, return_video_kwargs=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)

if __name__ == "__main__":
    main()