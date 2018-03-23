# Relevant pytorch snippets

## Select the GPU for variables

- Pre-execution: `CUDA_VISIBLE_DEVIECS=1,2 python myscript.py`
- Context manager: `with torch.cuda.device(1):`
- Manually: `model.cuda(1)`
