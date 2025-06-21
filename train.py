import time
import torch
import random
import argparse
import importlib
import torchmetrics
import torch.utils
import torch.utils.data
import MRIDataset
import SMDataset
import ImageDataset
import GeneDataset
import NewGeneDataset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.utils.class_weight import compute_class_weight

# # Validation/Test function for images only
# def validate(model, device, val_loader, criterion, names, metrics):
#     model.eval()
#     running_loss = 0.0

#     with torch.no_grad():
#         for data in val_loader:
#             inputs = data['image']
#             labels = data['label']
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#             for metric in metrics:
#                 x = metric(outputs, labels)
    
#     for name, metric in zip(names,metrics):
#         x = metric.compute()
#         print(f"{name}: {x}")
#         metric.reset()
#     print(f"Validation Loss (Average): {loss/len(val_loader)}")

# Validation/Test function for genes only
def validate(model, device, val_loader, criterion, names, metrics):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            inputs = data['genes']
            labels = data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            for metric in metrics:
                x = metric(outputs, labels)
    
    for name, metric in zip(names,metrics):
        x = metric.compute()
        print(f"{name}: {x}")
        metric.reset()
    print(f"Validation Loss (Average): {loss/len(val_loader)}")

# Validation/Testing for multi-modal data
# def validate(model, device, val_loader, criterion, names, metrics):
#     model.eval()
#     running_loss = 0.0

#     with torch.no_grad():
#         for data in val_loader:
#             inputs = data['image']
#             snps = data['genes']
#             labels = data['label']
#             inputs, snps, labels = inputs.to(device), snps.to(device), labels.to(device)

#             outputs = model(inputs, snps)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#             for metric in metrics:
#                 x = metric(outputs, labels)
        
#     for name, metric in zip(names,metrics):
#         x = metric.compute()
#         print(f"{name}: {x}")
#         metric.reset()
#     print(f"Validation Loss (Average): {loss/len(val_loader)}")

# Training function Imagess
# def train(model, device, train_loader, criterion, optimizer, epoch, running_loss = 0.0):
#     model.train()
#     start = time.time()
#     correct = 0
#     total = 0
    
#     for data in train_loader:
#         inputs = data['image']
#         labels = data['label']
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     accuracy = correct / total * 100
#     end = time.time()
#     print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%, Time: {end-start}s')

#GENE
def train(model, device, train_loader, criterion, optimizer, epoch, running_loss = 0.0):
    model.train()
    start = time.time()
    correct = 0
    total = 0
    
    for data in train_loader:
        inputs = data['genes']
        labels = data['label']
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    end = time.time()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%, Time: {end-start}s')

# Multi-modal training function
# def train(model, device, train_loader, criterion, optimizer, epoch, running_loss = 0.0):
#     model.train()
#     start = time.time()
#     correct = 0
#     total = 0
    
#     for data in train_loader:
#         inputs = data['image']
#         snps = data['genes']
#         labels = data['label']
#         inputs, snps, labels = inputs.to(device), snps.to(device), labels.to(device)
#         optimizer.zero_grad()

#         outputs = model(inputs, snps)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     accuracy = correct / total * 100
#     end = time.time()
#     print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%, Time: {end-start}s')

def save_model(model, filename='checkpoint.pth'):
    weights = {'model_state_dict': model.state_dict()}
    filename = filename.replace('checkpoint','trained')
    torch.save(weights, filename)
    print(f'Saved model weights at {filename}')

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    filename = filename.replace('trained','checkpoint')
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved at epoch {epoch}')

def main(args):
    """Initializes dataset, trains, validates models

    Args:
        args (dict): A dictionary containing user defined arguments
    """
    
    # Print the parsed arguments for clarity and user verification
    print(f"Data Directory: {args.data}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.epochs}")
    print(f"Number of workers: {args.workers}")
    print(f"Train Test Validation Split: {args.ttv_split}")
    print(f"Randomization Control: {args.random_seed}")
    print(f"Optimizer: {args.optim}")
    print(f"Optimizer Parameters: {args.optim_params}")
    print(f"Model Name: {args.model}")
    print(f"Checkpoint File: {args.checkpoint}")
    print(f"Final Model Save File: {args.save}")
    print(f"Mode: {args.mode}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dynamically import the model class based on user input
    module_name, class_name = args.model.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    # Instantiate the model
    model = model_class(16,8).to(device)
    print(f"Model instance created: {model}")

    # Set the random seed to limit variation
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch_generator = torch.Generator().manual_seed(args.random_seed)

    # Default transformation
    transform = transforms.Compose([
		transforms.Resize([192,192]),        
		transforms.ToTensor(),
    ])

    # Prepare data and dataloaders
    # dataset = MRIDataset.MRI2DDataset(args.data,transform=transform)
    # trainset, testset, valset = torch.utils.data.random_split(dataset, args.ttv_split, torch_generator)

    # Prepare data for each modality
    # dataset = ImageDataset.ImageDataset(args.data, transform=transform)
    # trainset, testset, valset = torch.utils.data.random_split(dataset, args.ttv_split, torch_generator)

    # dataset = GeneDataset.GeneDataset(args.data)
    # trainset, testset, valset = torch.utils.data.random_split(dataset, args.ttv_split, torch_generator)

    dataset = NewGeneDataset.NewGeneDataset(args.data)
    # print(len(dataset))
    trainset, testset , valset = torch.utils.data.random_split(dataset, args.ttv_split, torch_generator)

    # Prepare data and dataloader for multi-modal data
    # dataset = SMDataset.SMDataset(args.data,transform=transform)
    # trainset, testset, valset = torch.utils.data.random_split(dataset, args.ttv_split, torch_generator)

    print(f"Size of Training Set: {len(trainset)}")
    print(f"Size of Testing Set: {len(testset)}")
    print(f"Size of Validation Set: {len(valset)}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # print(trainloader[0]["genes"])
    # Carefully uncomment to add the ability for weighted loss
    # labels = [data['label'] for data in dataset]

    # class_weights = compute_class_weight('balanced', classes=[0,1,2], y=labels)
    # class_weights = torch.tensor([1.3, 1.0, 1.1], dtype=torch.float).to(device)

    # Initialize the optimizer and loss criterion
    criterion = nn.CrossEntropyLoss(weight=None)
    
    # Optional ability to add weights to CrossEntropyLoss
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # print(f"Class_Weights: {class_weights}")

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = args.optim_params[0], weight_decay=args.optim_params[1])
    else:
        optimizer = optim.SGD(model.parameters(), lr = args.optim_params[0], weight_decay=args.optim_params[1], momentum=args.optim_params[2])

    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes = 3).to(device)
    precision = torchmetrics.classification.MulticlassPrecision(num_classes = 3).to(device)
    recall = torchmetrics.classification.MulticlassRecall(num_classes = 3).to(device)
    confusion = torchmetrics.classification.MulticlassConfusionMatrix(num_classes = 3).to(device)

    metrics = [accuracy, precision, recall, confusion]
    names = ["Accuracy", "Precision", "Recall", "Confusion"]

    # Resume training from checkpoint or load pre-trained weights
    # Alternatively can be used to only evaluate performance of a model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Not in evaluation mode
        # Can be used to train with pre-trained weights
        if args.mode == 'CH':
            print(f'\nResuming training from checkpoint at: {args.checkpoint}')
            
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                for i in range(epoch,args.epochs):
                    train(model, device, trainloader, criterion, optimizer, i, loss)
                    if i % 3 == 0:
                        validate(model, device, valloader, criterion,names,metrics)

            except KeyboardInterrupt:
                print(f'\nKeyboard Interrupt Handled')
                save_checkpoint(model, optimizer, i, loss, args.checkpoint)
                return 0

            except Exception as e:
                print(f'\nThere was an exception during training. Due to {str(e)}')
                save_checkpoint(model, optimizer, i, loss, args.checkpoint)
                return 0


            # Test model after training
            print(f'\nTesting Model...')
            validate(model, device, testloader,criterion,names,metrics)

            # Save model after training
            print(f'Saving model #{args.checkpoint.split("_")[1]}')
            save_model(model,args.checkpoint)
        
        # Train model loaded with Pre-trained weights
        elif args.mode == 'PT':
            print(f'\nModel weights loaded from: {args.checkpoint}')
            print(f'Beginning Training.')

            try:
                for i in range(0, args.epochs):
                    loss = 0.0
                    train(model, device, trainloader, criterion, optimizer, i, loss)
                    if i % 3 == 0:
                        validate(model, device, valloader, criterion,names,metrics)
            
            except KeyboardInterrupt:
                print(f'\nKeyboard Interrupt Handled')
                save_checkpoint(model, optimizer, i, loss, args.save)
                return 0

            except Exception as e:
                print(f'\nThere was an exception during training. Due to {str(e)}')
                save_checkpoint(model, optimizer, i, loss, args.save)
                return 0


            # Test model after training
            print(f'\nTesting Model...')
            validate(model, device, testloader,criterion,names,metrics)

            # Save model after training
            print(f'Saving model #{args.save.split("_")[1]}')
            save_model(model,args.save)

        # Evaluate the trained model
        elif args.mode == 'E':
            print(f'\nModel weights loaded from: {args.checkpoint}')
            print(f'Evaluating model.')
            validate(model, device, testloader,criterion,names,metrics)
    
    # Model needs to be trained from scratch
    else:
        print('\nNo model weights found. Initializing Training.')
        try:
            for i in range(0, args.epochs):
                    loss = 0.0
                    train(model, device, trainloader, criterion, optimizer, i, loss)
                    if i % 3 == 0:
                        validate(model, device, valloader, criterion,names,metrics)

        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {i} due to KeyboardInterrupt")
            save_checkpoint(model, optimizer, i, loss, args.save)
            return 0
    
        except Exception as e:
            print(f"\nTraining interrupted at epoch {i} due to {str(e)}")
            save_checkpoint(model, optimizer, i, loss, args.save)
            return 0
        
        # Test model after training
        print(f'\nTesting Model...')
        validate(model, device, testloader, criterion,names,metrics)

        print(f'Saving model #{args.save.split("_")[1]}')
        save_model(model,args.save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with specified parameters.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--data", type=str, default='./Dataset', help="Path to data directory.\n"
                        "Default: ./Dataset")
    parser.add_argument("--workers", type=int, default=0, help="Number of multiprocessing workers.")
    parser.add_argument("--batch_size", type=int, required=True, help="Size of each training batch.")
    parser.add_argument("--epochs", type=int, required=True, help="Enter the number of epochs to train for")
    parser.add_argument("--ttv_split", type=float, nargs='+', required=True, help="Enter the Train Test Validation split.\n"
                        "Enter 2 values for train test split. Use testing data for validation."
                        "Example Usage: --ttv_split .80 .10 .10")
    parser.add_argument("--random_seed", type=int, default=42, help="Enter your preferred seed.\n"
                        "Default: 42")
    parser.add_argument("--optim", type=str, default='adam', choices=["adam", "sgd"], help="Optimizer to use for training.\n"
                        "adam - Use the Adam Optimizer.\n"
                        "sgd - Use the Stochastic Gradient Descent.\n"
                        "Deafult: adam")
    parser.add_argument("--optim_params", type=float, nargs='+', required=True, help="Learning parameters for the optimizer. Enter each parameter with a space separating them.\n"
                        "Parameters for Adam: Learning Rate, Weight Decay.\n"
                        "Parameters for SGD: Learning Rate, Weight Decay, Momentum")
    parser.add_argument("--model", type=str, required=True, help="Name of the model class to use.\n"
                        "Example Usage: --model module_name.Class_name")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path to resume training from.[Deafult Behavior]\n"
                        "Path to load a trained model for evaluation. [Requires mode flag to be set to 'E'].\n"
                        "Path to pre-trained model weights for further training and evaluation. [Requires mode flag to be set to 'PT']\n"
                        "Warning: Not setting the mode flag appropriately might result in unexpected behavior.")
    parser.add_argument("--save", type=str, default=f'./runs/trained/model_{random.randint(0,1000000)}.pth', help="Path to save the final trained model.")
    parser.add_argument("--mode", type=str, default='CH', choices=['PT','E','CH'], help="Set eval flag to True to evaluate the model.\n"
                        "E: Set the model to Evaluate mode. Reports accuracy metrics for the model.\n"
                        "PT: Set the model to Pre-Trained mode. Loads pre-trained weights and starts training.\n"
                        "CH: Continue training from a checkpoint. [Default Behavior]")

    args = parser.parse_args()
    main(args)