import torch
from LogoModel import LogoModel
from LogoDataset import LogoDataModule
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

batch_size = 128
num_workers = 8
EPOCHS = 150
lr = (2e-6, 2e-4)
data_dir = 'dataset/LogoDet-3K_preprocessed'
ocr_model_path="ocr_best.ckpt"

torch.set_float32_matmul_precision('medium')

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    filename='model-{epoch:02d}-{val_acc:.5f}',
    save_top_k=3,
    mode='max',
    save_last=True
)

model = LogoModel(lr=lr)
data_module = LogoDataModule(batch_size=batch_size, num_workers=num_workers, data_dir=data_dir, transform=model.preprocess)

trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    devices=1,
    accelerator="gpu",
    strategy=trainer_strategy,
    precision="16-mixed",
    callbacks=[checkpoint_callback],
)
trainer.fit(model, data_module)

