import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm  # make sure to install timm: pip install timm
import wandb
from tqdm import tqdm
from print_model_stats import print_model_stats

TQDM_NCOLS=60


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    log_every = 100
    print(f"[Epoch {epoch}] Training...")
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=TQDM_NCOLS):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % log_every == 0:
            wandb.log({"Loss": loss.item() * inputs.size(0)})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    wandb.log({"Train Loss": epoch_loss, "Train Acc": epoch_acc, "Epoch": epoch})
    print(f"[Epoch {epoch}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    print(f"[Epoch {epoch}] Eval...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, total=len(val_loader), ncols=TQDM_NCOLS):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    wandb.log({"Val Loss": epoch_loss, "Val Acc": epoch_acc, "Epoch": epoch})
    print(f"[Epoch {epoch}] Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR100 ViT Training with WandB Logging")
    parser.add_argument("--optimizer", type=str, default="adam", 
                        choices=["adam", "sgd", "adamw_sn", "adamw_snsm"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=200)
    
    args = parser.parse_args()
    
    run_name = f"{args.optimizer}_lr{args.lr}_wd{args.wd}"
    if args.optimizer == "adamw_snsm":
        run_name += f"r{args.rank}ug{args.update_proj_gap}"
    # Initialize wandb logging
    wandb.init(project="cifar100_vit_experiment", 
               config=vars(args),
               name=run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms: resize to 224 to match ViT input requirements
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR100 dataset
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create the Vision Transformer model (using timm)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
    model.to(device)

    print_model_stats(model)
    # Choose optimizer based on argument
    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer.lower() == "adamw_sn":
        from adamw_sn import AdamWSN
        optimizer = AdamWSN(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer.lower() == "adamw_snsm":
        from adamw_snsm import AdamwSNSM
        linear_modules = [module.weight for module in model.modules() if isinstance(module, nn.Linear)]
        regular_params = [p for p in model.parameters() if id(p) not in [id(p) for p in linear_modules]]
        snsm_params = {'params': linear_modules, "rank": args.rank, "update_proj_gap": args.update_proj_gap}
        param_groups = [
            {'params': regular_params},   # this is just sn
            snsm_params
        ]
        optimizer = AdamwSNSM(param_groups, lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")

    print(optimizer)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        validate(model, device, test_loader, criterion, epoch)

    wandb.finish()

if __name__ == "__main__":
    main()
