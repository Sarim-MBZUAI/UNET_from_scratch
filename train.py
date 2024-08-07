# import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm
# import torch.nn as nn
# import torch.optim as optim
# from model import UNET
# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_imgs,
# )
# import wandb
# import logging
# import os
# from datetime import datetime


# LEARNING_RATE = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16
# NUM_EPOCHS = 100
# NUM_WORKERS = 4
# IMAGE_HEIGHT = 160
# IMAGE_WIDTH = 240
# PIN_MEMORY = True
# LOAD_MODEL = False
# TRAIN_IMG_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/train_images"
# TRAIN_MASK_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/train_masks"
# VAL_IMG_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/val_images"
# VAL_MASK_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/val_masks"

# def setup_logging():
#     log_dir = "/home/sarim.hashmi/Downloads/unet_from_scratch/UNET_from_scratch_old/logs"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
#     logging.basicConfig(
#         filename=log_file,
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     logging.getLogger('').addHandler(console)

# def log_info(message):
#     logging.info(message)
#     wandb.run.log({"info": message})

# def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
#     model.train()
#     loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
#     total_loss = 0

#     for batch_idx, (data, targets) in enumerate(loop):
#         data = data.to(device=DEVICE)
#         targets = targets.float().unsqueeze(1).to(device=DEVICE)

#         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)

#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         total_loss += loss.item()
#         loop.set_postfix(loss=loss.item())


#         wandb.log({
#             "batch_loss": loss.item(),
#             "batch": batch_idx + epoch * len(loader),
#             "epoch": epoch + 1
#         })

#     avg_loss = total_loss / len(loader)
#     log_info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Train Loss: {avg_loss:.4f}")
#     wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

#     return avg_loss

# def main():
#     setup_logging()
    
#     wandb.init(project="unet-segmentation", config={
#         "learning_rate": LEARNING_RATE,
#         "epochs": NUM_EPOCHS,
#         "batch_size": BATCH_SIZE,
#         "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
#         "model": "UNET",
#         "optimizer": "Adam",
#         "loss_function": "BCEWithLogitsLoss",
#     })

#     log_info("Starting training process")
#     log_info(f"Device: {DEVICE}")

#     train_transform = A.Compose(
#         [
#             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#             A.Rotate(limit=35, p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.1),
#             A.Normalize(
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 max_pixel_value=255.0,
#             ),
#             ToTensorV2(),
#         ],
#     )

#     val_transforms = A.Compose(
#         [
#             A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#             A.Normalize(
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 max_pixel_value=255.0,
#             ),
#             ToTensorV2(),
#         ],
#     )

#     model = UNET(in_channels=3, out_channels=1).to(DEVICE)
#     wandb.watch(model, log="all", log_freq=100)
#     loss_fn = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     train_loader, val_loader = get_loaders(
#         TRAIN_IMG_DIR,
#         TRAIN_MASK_DIR,
#         VAL_IMG_DIR,
#         VAL_MASK_DIR,
#         BATCH_SIZE,
#         train_transform,
#         val_transforms,
#         NUM_WORKERS,
#         PIN_MEMORY,
#     )

#     log_info(f"Train dataset size: {len(train_loader.dataset)}")
#     log_info(f"Validation dataset size: {len(val_loader.dataset)}")

#     if LOAD_MODEL:
#         load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
#         log_info("Loaded pre-trained model")

#     scaler = torch.amp.GradScaler()

#     best_val_score = 0
#     for epoch in range(NUM_EPOCHS):
#         train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)
        
#         model.eval()
#         with torch.no_grad():
#             val_score = check_accuracy(val_loader, model, device=DEVICE)
        
#         if val_score is not None:
#             log_info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Score: {val_score:.4f}")
#             wandb.log({"val_score": val_score, "epoch": epoch + 1})

#             if val_score > best_val_score:
#                 best_val_score = val_score
#                 checkpoint = {
#                     "state_dict": model.state_dict(),
#                     "optimizer": optimizer.state_dict(),
#                 }
#                 save_checkpoint(checkpoint)
#                 log_info(f"New best model saved. Best validation score: {best_val_score:.4f}")
#         else:
#             log_info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation score not available")

#         save_predictions_as_imgs(
#             val_loader, model, folder="saved_images/", device=DEVICE
#         )
        
#         wandb.log({
#             "epoch": epoch + 1,
#             "train_loss": train_loss,
#             "val_score": val_score if val_score is not None else 0,
#             "learning_rate": optimizer.param_groups[0]['lr']
#         })

#     log_info("Training completed")
#     wandb.finish()

# if __name__ == "__main__":
#     main()


import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import wandb
import os
from datetime import datetime


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 4
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/train_images"
TRAIN_MASK_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/train_masks"
VAL_IMG_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/val_images"
VAL_MASK_DIR = "/home/sarim.hashmi/Downloads/unet_large_files_/original_dataset/val_masks"

log_messages = []

def log_info(message):
    log_messages.append(message)
    wandb.run.log({"info": message})

def save_log_to_file(epoch):
    log_dir = "/home/sarim.hashmi/Downloads/unet_from_scratch/UNET_from_scratch_old/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch+1}.txt")
    with open(log_file, 'w') as f:
        for message in log_messages:
            f.write(message + '\n')
    log_messages.clear()

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        wandb.log({
            "batch_loss": loss.item(),
            "batch": batch_idx + epoch * len(loader),
            "epoch": epoch + 1
        })

    avg_loss = total_loss / len(loader)
    log_info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Train Loss: {avg_loss:.4f}")
    wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

    return avg_loss

def main():
    wandb.init(project="unet-segmentation", config={
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "model": "UNET",
        "optimizer": "Adam",
        "loss_function": "BCEWithLogitsLoss",
    })

    log_info("Starting training process")
    log_info(f"Device: {DEVICE}")

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    wandb.watch(model, log="all", log_freq=100)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    log_info(f"Train dataset size: {len(train_loader.dataset)}")
    log_info(f"Validation dataset size: {len(val_loader.dataset)}")

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        log_info("Loaded pre-trained model")

    scaler = torch.amp.GradScaler()

    best_val_score = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)
        
        model.eval()
        with torch.no_grad():
            val_score = check_accuracy(val_loader, model, device=DEVICE)
        
        if val_score is not None:
            log_info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Score: {val_score:.4f}")
            wandb.log({"val_score": val_score, "epoch": epoch + 1})

            if val_score > best_val_score:
                best_val_score = val_score
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)
                log_info(f"New best model saved. Best validation score: {best_val_score:.4f}")
        else:
            log_info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation score not available")

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_score": val_score if val_score is not None else 0,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        save_log_to_file(epoch)

    log_info("Training completed")
    save_log_to_file(NUM_EPOCHS)
    wandb.finish()

if __name__ == "__main__":
    main()
