# TROUBLESHOOTING COMMON PROBLEMS

### RuntimeError: mat1 and mat2 shapes cannot be multiplied (AxB and CxD)

Compare the actor and critic input size to the observation size. They need to be identical. Remember that changing team size and self play can change the observation size.

### Blue Screen of Death error (related to wandb)

1) Rollback your Nvidia driver to WHQL certified September 2021

OR 2) Comment out the following lines of code in the wandb repo:

```python
try:
    pynvml.nvmlInit()
    self.gpu_count = pynvml.nvmlDeviceGetCount()
except pynvml.NVMLError:
```

### Can't get Redis to work

- WSL2 is probably the easiest way to get things working.
- Double check that you can ping the redis server locally.

### There are no errors but changes I'm making don't seem to be affecting anything

- Double check that observations, rewards, and action parsers are the same on both the learner and workers.

### wandb is not working properly or giving you hint errors in your IDE

- Check that there isn’t a folder created called wandb in your project.
- This is created automatically by wandb, you can change the name in the init call so it doesn’t interfere.
