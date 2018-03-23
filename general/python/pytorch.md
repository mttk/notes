# Relevant pytorch snippets

## Select the GPU for variables

- Pre-execution: 
```python
CUDA_VISIBLE_DEVIECS=1,2 python myscript.py
```
- Context manager: 
```python
with torch.cuda.device(1):
```
- Manually: 
```python
model.cuda(1)
```

## Dropout in RNN params

Is just dropout on the outputs.
