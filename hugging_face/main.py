from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

# Each process runs on 1 GPU device specified by the local_rank argument.
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
torch.distributed.init_process_group(backend='nccl')

# Encapsulate the model on the GPU assigned to the current process
device = torch.device('cuda', arg.local_rank)
model = model.to(device)
distrib_model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

# Restricts data loading to a subset of the dataset exclusive to the current process
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

dataset = datasets.MNIST(root, train=train, transform=transform, download=True)
sampler = DistributedSampler(dataset)

dataloader = DataLoader(dataset, sampler=sampler)
for inputs, labels in dataloader:
    predictions = distrib_model(inputs.to(device))         # Forward pass
    loss = loss_function(predictions, labels.to(device))   # Compute loss function
    loss.backward()                                        # Backward pass
    optimizer.step()                                       # Optimizer step
