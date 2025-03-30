import torch.nn as nn

def print_model_stats(model):
    # Compute total parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    
    # Collect modules that have parameters (treat these as layers)
    layer_modules = [module for module in model.modules() if any(p.numel() > 0 for p in module.parameters(recurse=False))]
    
    # Separate modules into groups
    linear_modules = [m for m in layer_modules if isinstance(m, nn.Linear)]
    conv_modules   = [m for m in layer_modules if isinstance(m, nn.Conv2d)]
    other_modules  = [m for m in layer_modules if not isinstance(m, (nn.Linear, nn.Conv2d))]
    
    # Helper function: sum parameters for a list of modules
    def group_params(modules):
        return sum(sum(p.numel() for p in module.parameters()) for module in modules)
    
    linear_params = group_params(linear_modules)
    conv_params   = group_params(conv_modules)
    # For other modules, we subtract linear and conv parameters from the total
    other_params  = total_params - linear_params - conv_params
    
    # Print concise summary stats
    print("Model Stats Summary:")
    print("-------------------------")
    print(f"Total parameters: {total_params}")
    print(f"Linear layers: {len(linear_modules)} layers | {linear_params} parameters ({(linear_params/total_params)*100:.2f}%)")
    print(f"Conv layers:   {len(conv_modules)} layers | {conv_params} parameters ({(conv_params/total_params)*100:.2f}%)")
    print(f"Other layers:  {len(other_modules)} layers | {other_params} parameters ({(other_params/total_params)*100:.2f}%)")
