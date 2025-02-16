#!/usr/bin/env python
"""
Generate LLM queries to solve simple math problem.
Use to verify determinism of LLM.

Use set of JSON files instead of list so easier to run
in parallel with GNU parallel.
"""

import json

import fire


def generate_queries(
    out_file_prefix: str,
    num_gen: int,
):
    """
    Main function that runs the program.
    """

    for i in range(num_gen):
        r = [
            {
                "role": "user",
                "content": (
                    "Solve the equation 3x + 2 = 17 for x."
                    "\n\nShow and explain all steps."
                ),
            },
        ]

        out_file = f"{out_file_prefix}_{i:03}.json"
        print("Save file:", out_file)
        with open(out_file, "w", encoding="UTF8") as f:
            json.dump(r, f)


if __name__ == "__main__":
    fire.Fire(generate_queries)
