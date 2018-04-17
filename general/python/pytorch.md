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

## Blog posts / tutorials

http://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour/

## Memory size of parameters
Credits to @apazske
```python
total_size = 0
for tensor in model.state_dict().values():
   total_size += tensor.numel() * tensor.element_size()
```
