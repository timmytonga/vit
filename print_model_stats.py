import torch.nn as nn

def print_model_stats(model):
    # Compute total parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    
    # Collect modules that have parameters (treat these as layers)
    layer_modules = [module for module in model.modules() if any(p.numel() > 0 for p in module.parameters(recurse=False))]
    total_layers = len(layer_modules)
    
    # Separate out Linear and Conv modules and collect details for each
    linear_modules = []
    conv_modules = []
    other_modules = []
    
    for module in layer_modules:
        if isinstance(module, nn.Linear):
            linear_modules.append(module)
        elif isinstance(module, nn.Conv2d):
            conv_modules.append(module)
        else:
            other_modules.append(module)
    
    # Helper: sum of parameters for a list of modules
    def params_sum(modules):
        return sum(sum(p.numel() for p in module.parameters()) for module in modules)
    
    linear_params = params_sum(linear_modules)
    conv_params = params_sum(conv_modules)
    other_params = total_params - linear_params - conv_params
    
    # Print summary stats
    print("Model Statistics:")
    print("----------------------")
    print(f"Total parameters: {total_params}")
    print(f"Total layers (with parameters): {total_layers}\n")
    
    # Linear layers details
    print(f"Linear layers: {len(linear_modules)}")
    print(f"  Combined Linear parameters: {linear_params} ({(linear_params/total_params*100):.2f}%)")
    for idx, module in enumerate(linear_modules):
        p_count = sum(p.numel() for p in module.parameters())
        perc = p_count / total_params * 100
        print(f"    Linear layer {idx+1} ({module.__class__.__name__}): {p_count} parameters, {perc:.2f}% of total")
    print("")
    
    # Convolutional layers details
    print(f"Convolutional layers: {len(conv_modules)}")
    print(f"  Combined Conv parameters: {conv_params} ({(conv_params/total_params*100):.2f}%)")
    for idx, module in enumerate(conv_modules):
        p_count = sum(p.numel() for p in module.parameters())
        perc = p_count / total_params * 100
        print(f"    Conv layer {idx+1} ({module.__class__.__name__}): {p_count} parameters, {perc:.2f}% of total")
    print("")
    
    # Other layers details
    print(f"Other layers: {len(other_modules)}")
    print(f"  Combined Other parameters: {other_params} ({(other_params/total_params*100):.2f}%)")
    for idx, module in enumerate(other_modules):
        p_count = sum(p.numel() for p in module.parameters())
        # Only print modules with parameters for clarity
        if p_count > 0:
            perc = p_count / total_params * 100
            print(f"    Other layer {idx+1} ({module.__class__.__name__}): {p_count} parameters, {perc:.2f}% of total")
