#!/usr/bin/env python
"""
Send query to localhost VLLM server, for test purposes.
"""


import json
import time

import fire
from openai import OpenAI


def send_query(
    prompt_file: str,
    temperature: float = 0,
):
    """
    Send query to localhost LLM server and get response.
    """

    time_start = time.time()
    print("Load input file:", prompt_file)
    print("Using temperature:", temperature)
    with open(prompt_file, "r", encoding="UTF8") as f:
        messages = json.load(f)

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    completion = client.chat.completions.create(
        messages=messages,
        model="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        max_tokens=256,
        temperature=temperature,
    )

    print("Completion result:", completion)
    time_end = time.time()
    print(f"Time taken: {round(time_end - time_start, 2)} seconds.")

    out = completion.choices[0].message.content
    out = out.encode("utf-8").decode("unicode-escape")
    print("LLM OUTPUT:", out)


if __name__ == "__main__":
    fire.Fire(send_query)
