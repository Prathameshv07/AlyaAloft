# AlyaAloft Quickstart Guide

This guide will help you quickly set up and run AlyaAloft with the enhanced FLAN-T5 model.

## Step 1: Environment Setup

Create and activate a python or conda virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# create a conda virtual environment
conda create -n venv python=3.9
conda activate venv
```

## Step 2: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

For improved performance with CUDA-enabled GPUs, make sure to install bitsandbytes:

```bash
pip install bitsandbytes
```

## Step 3: Download the T5 Model

AlyaAloft uses the FLAN-T5 model from Hugging Face. Use our script to download it:

```bash
# Download the default model (flan-t5-base)
python scripts/download_t5_model.py

# For a smaller model with lower resource requirements:
python scripts/download_t5_model.py --model google/flan-t5-small

# For better quality with higher resource requirements:
python scripts/download_t5_model.py --model google/flan-t5-large
```

## Step 4: Run the Application

Start the application with default settings:

```bash
python start_app.py
```

Or customize with command-line options:

```bash
# Run on a different port with debug logging
python start_app.py --port 9000 --log-level DEBUG

# Run on all network interfaces (accessible from other devices)
python start_app.py --host 0.0.0.0
```

## Step 5: Access the Web Interface

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

(Or use the custom port if you specified one)

## Step 6: Upload Documents and Ask Questions

1. Click the "Upload Document" button to upload a PDF
2. Wait for the document to be processed
3. Type your questions in the chat interface
4. AlyaAloft will respond with answers based on the document's content

## Model Performance Settings

If you're experiencing slow performance or memory issues, you can adjust the following settings in `app/config/model_config.py`:

- Reduce `max_length` and `min_length` in `T5_CONFIG`
- Reduce the number of beams in `num_beams`
- Reduce `max_context_chunks` in `PIPELINE_CONFIG`

## GPU Acceleration

AlyaAloft automatically uses CUDA if available. To check if your GPU is being used:

1. Run the application with debug logging:
   ```bash
   python start_app.py --log-level DEBUG
   ```
2. Check the console for a message confirming CUDA availability

## Getting Help

If you encounter issues:

- Check the console output for error messages
- Refer to the README.md and Troubleshooting section
- Make sure your environment meets the minimum requirements (Python 3.8+, sufficient RAM)

Enjoy using AlyaAloft with enhanced FLAN-T5 prompting! 