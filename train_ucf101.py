import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import make_grid
from accelerate import Accelerator, load_checkpoint_in_model
from sosora import sampler
from sosora.model import SoSora, LevelSpec, GlobalAttentionSpec, NeighborhoodAttentionSpec, MappingSpec


def make_model() -> SoSora:
    d_head, dropout = 64, 0.05
    return SoSora(
        levels=[
            LevelSpec(2, 256, 256 * 3, NeighborhoodAttentionSpec(d_head, 7), dropout, (2, 2, 2)),
            LevelSpec(4, 1024, 1024 * 3, GlobalAttentionSpec(d_head), dropout, (2, 2, 2)),
            LevelSpec(12, 3072, 3072 * 3, GlobalAttentionSpec(d_head), dropout),
        ],
        mapping=MappingSpec(2, 256, 768, 0.0),
        in_channels=3,
        out_channels=3,
        patch_size=(2, 2, 2),
        num_classes=101,
    )


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def loop_loader(d: DataLoader):
    while True:
        for x, c in d:
            yield x, c


def make_data(path_data: str, path_split: str, h: int = 48, w: int = 64, seq_len: int = 16, batch_size: int = 32) -> DataLoader:
    transform = v2.Compose(
        [
            v2.Resize((h, w)),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,)),
            v2.Lambda(lambda x: x.permute(0, 2, 3, 1)),
        ]
    )
    dataset = datasets.UCF101(
        path_data,
        path_split,
        frames_per_clip=seq_len,
        step_between_clips=seq_len,
        transform=transform,
        output_format="TCHW"
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)


def make_video_grid(x: torch.Tensor, nrow: int):
    sample = (x * 0.5 + 0.5).clamp(0, 1)
    sample = sample.permute(1, 0, 4, 2, 3) # BTHWC -> TBCHW
    frames = []
    for batch in sample:
        grid = make_grid(batch.float(), nrow=nrow).permute(1, 2, 0)
        frame = (grid * 255).type(torch.uint8).cpu().numpy()
        frames.append(Image.fromarray(frame))
    return frames


if __name__ == "__main__":
    SAVE_DIR = "checkpoints_ucf"
    os.makedirs(SAVE_DIR, exist_ok=True)
    LR, GRAD_NORM_CLIP = 5e-4, 0.5
    BS, SEQ_LEN, H, W = 64, 64, 48, 64
    model, dataloader = make_model(), make_data("UCF-101", "ucfTrainTestlist", H, W, SEQ_LEN, BS)
    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    accelerator.init_trackers(project_name="sosora", init_kwargs={"wandb": {"dir": "../../wandb"}})
    optimizer = optim.AdamW(model.param_groups(LR), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)
    load_checkpoint_in_model(model, SAVE_DIR) # restart from checkpoint
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    generator = loop_loader(dataloader)
    for i in tqdm(range(100000)):
        x, c = next(generator)
        optimizer.zero_grad()
        loss = sampler.rf_forward(model, x, c)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
        optimizer.step()
        accelerator.log({"train_loss": loss.item()})

        if i % 500 == 0:
            model.eval()
            with torch.inference_mode():
                z = torch.randn(25, SEQ_LEN, H, W, 3).to(accelerator.device)
                cond = torch.arange(0, 25).to(accelerator.device) % 51
                n_cond = torch.ones_like(cond).to(accelerator.device) * 51
                images = sampler.rf_sample(model, z, cond, n_cond)
                frames = make_video_grid(images, 5)
                frames = np.array([np.array(f) for f in frames]).transpose(0, 3, 1, 2)
                accelerator.log({"video": wandb.Video(frames, fps=16)})
            model.train()
            #accelerator.save_state(SAVE_DIR)
            accelerator.save_model(model, SAVE_DIR)

    accelerator.end_training()
