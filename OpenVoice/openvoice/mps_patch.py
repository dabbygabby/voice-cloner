import torch

def patch_mps_for_openvoice():
    """
    Patch to disable MPS for OpenVoice on Apple Silicon
    This prevents runtime errors during TTS processing
    """
    if torch.backends.mps.is_available():
        print("Applying MPS patch for Apple Silicon compatibility...")
        torch.backends.mps.is_available = lambda: False
        print("MPS disabled for OpenVoice compatibility")

# Apply patch when imported
patch_mps_for_openvoice()