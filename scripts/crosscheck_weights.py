import torch
import torch.nn as nn
import torchvision.models as models
# from models.model import ImageNetModel, CLIPModel
from timm.models import vit_base_patch16_224

# resnet18 = models.resnet18(pretrained=True)
vit16 = vit_base_patch16_224(pretrained=True)
# print(vit16)

# full_model = ImageNetModel(resnet18, num_classes=2)
full_model = vit_base_patch16_224(pretrained=True)
full_model.head = nn.Linear(full_model.head.in_features, 2)
full_model.load_state_dict(torch.load('/home/santosh.sanjeev/model-soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/vit_v2_full_finetuning.pth')['model_state_dict'])
full_model.eval()
# print(full_model)
# Check if the weights are the same
# weights_are_same = torch.allclose(vit16(torch.randn(1, 3, 224, 224)),
#                                   full_model.feature_extractor(torch.randn(1, 3, 224, 224)))

# Get the state dictionaries of the models
vit16_state_dict = vit16.state_dict()
full_model_state_dict = full_model.state_dict()

# Compare the state dictionaries
same_weights_layers = {}
different_weights_layers = {}


for name, params in vit16_state_dict.items():
    if name in full_model_state_dict and params.shape == full_model_state_dict[name].shape:
        if torch.allclose(params, full_model_state_dict[name]):
            same_weights_layers[name] = params
        else:
            different_weights_layers[name] = (params, full_model_state_dict[name])
    else:
        different_weights_layers[name] = None

for name, params in full_model_state_dict.items():
    if name not in vit16_state_dict:
        different_weights_layers[name] = None

print("Layers with the same weights:")
for name in same_weights_layers.keys():
    print(name)

print(different_weights_layers)
print("\nLayers with different weights:")
for name, params in different_weights_layers.items():
    if params is not None:
        if isinstance(params, tuple):
            print(f"Layer '{name}' has different weights.")
        else:
            print(f"Layer '{name}' is not present in one of the models.")

