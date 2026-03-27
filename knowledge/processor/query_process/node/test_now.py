def item_name_filter(validated_item_names):
    quoted = ", ".join(f'"{v}"' for v in validated_item_names)
    return f"item_name in [{quoted}]"


validated_item_names = ["iPhone15", "iPhone14", "HuaweiMate60"]
print(validated_item_names)
print(item_name_filter(validated_item_names))

import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))