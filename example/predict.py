import modis
import modis.utils
import modis.utils.data
from src.generate_dataset import get_datasets

# Prepare the data

config = modis.load_config('config/semisupervised.yaml')
checkpoint_file = "saved/checkpoints/intersim_2_delta/intersim_semisupervised/20250720_181713/checkpoint_best.pth"

datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'test',
    data_dir = './data',
    include_sample_ids = False
)
dataloaders = modis.utils.data.get_dataloaders(datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)
x, y = list(zip(*[modis.utils.data.get_samples_from_dataloader(dataloader, num_samples=100, device='cpu')
                  for dataloader in dataloaders]))

num_modalities = len(dataloaders)

# Load checkpoint
model = modis.Model(config).cpu()
model.load_from_checkpoint(checkpoint_file)

# Predict
pred = [model.predict(x[i], input_modality=i) for i in range(num_modalities)]
print(f"Modality predictions: {[p.numpy().shape for p in pred]}")

# Translate
for i in range(num_modalities):
    for j in range(num_modalities):
        if i == j: continue
        translation = model.translate(x[i], input_modality=i, output_modality=j)
        print(f"Shape for translation from modality {i} to {j}: {translation.shape}")

# Latents
latents = model.get_latents(x[0], input_modality=0)
print(f"Latents shape: {latents.numpy().shape}")