# Test vLLM Sampler

Some tests of vLLM sampler under various conditions.

## Integral Test for vLLM Determinism

Goal is to determine if LLM is asked to solve an integral showing steps,
is it consistent each time at zero temperature?

### Hardware

This test was done on a Lambda Labs gpu_1x_a10 server in the Virginia, USA region.

The 4-bit GPTQ quantized Llama 3.1 8B LLM was used for this test.

### Environment Setup

Setup a virtual environment and install vLLM and other dependencies within it:

```bash
python -m venv vllm-env
source vllm-env/bin/activate
python -m pip install -U pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install vllm openai fire

sudo apt install parallel
```

Clone the repo:

```bash
git bashne https://github.com/StanHatko/test_vllm_sampler
cd test_vllm_sampler
```

### Run Server

Start vLLM server:

```bash
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
  --host 127.0.0.1 \
  --port 8000 \
  --max_model_len 16000 \
  --enable-prefix-caching
```

### Results with vLLM 0.7.2

Initial test of vLLM:

```bash
./generate_detect_num_list.py /tmp/llm_test_basic 10 0
./send_local_llm_query.py /tmp/llm_test_basic_000.json

ls -1 /tmp/llm_test_basic_*.json
ls -1 /tmp/llm_test_basic_*.json | parallel -j 10 ./send_local_llm_query.py
time ( ls -1 /tmp/llm_test_basic_*.json | parallel -j 10 ./send_local_llm_query.py )
```
