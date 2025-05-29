# Report Generation Setup

## HPC Configuration

When running the `run_rep_gen.sh` script on the HPC:

1. **File Permissions**: Ensure you have permissions to access:
   - Input files
   - Model files
   - Docker/Singularity image
   - Python configuration file

2. **Output Directory**: Make sure you are saving the outputs to a directory you have write permissions for.

## Model and Dataset Configuration

### Model Path
```
PATH_TO_MODEL = "/d/hpc/projects/onj_fri/brainstorm/FineTune_models/GaMS-9B-Instruct/checkpoints/checkpoint-epoch-3"
```

### Test Dataset Path
```
PATH_TO_DATASET = "/d/hpc/projects/onj_fri/brainstorm/not_dataset_folder/not_test_data.csv"
```
