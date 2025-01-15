import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

# Create the log directory if it doesn't exist
log_dir = "./log/resnet18"

# Profile for multiple iterations and save the trace data for TensorBoard
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
             on_trace_ready=tensorboard_trace_handler(log_dir)) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

# You can then run TensorBoard to visualize the results:
# tensorboard --logdir=./log
