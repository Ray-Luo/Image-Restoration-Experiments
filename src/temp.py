import torch
import torchvision.models as models

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define an example input tensor
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image size

# Export the model to a .pt file
torch.save(model, 'resnet18.pt')

# You can also use JIT (Just-In-Time) scripting to optimize the model further
scripted_model = torch.jit.trace(model, dummy_input)
torch.jit.save(scripted_model, '/home/luoleyouluole/resnet18_scripted.pt')
