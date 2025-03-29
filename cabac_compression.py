"""
CABAC (Context-Adaptive Binary Arithmetic Coding) for StyleGAN3-HVAE Compression

This script implements CABAC entropy coding for compressing the latent codes from 
StyleGAN3-HVAE. CABAC is an advanced entropy coding technique that:
- Adapts probability models to local contexts
- Provides superior compression compared to standard entropy coders
- Is used in modern video codecs like H.264/AVC and H.265/HEVC

The implementation includes:
1. Context modeling for latent code compression
2. Probability estimation based on neighborhood patterns
3. Binary arithmetic coding for the final compression
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
from tqdm import tqdm
import struct
from collections import defaultdict
import time

from stylegan3_hvae_full import HVAE_VGG_Encoder, StyleGAN3Compressor, save_tensor_as_image
from gumbel_softmax_compression import GumbelSoftmaxDiscretization, GumbelSoftmaxCompressor

# Import our custom binarization layer
try:
    from gumbel_softmax_compression import GumbelSoftmaxDiscretization
except ImportError:
    print("Warning: gumbel_softmax_compression.py not found. Using simplified version.")
    
    # Simplified discretization class if gumbel_softmax_compression.py is not available
    class GumbelSoftmaxDiscretization(nn.Module):
        def __init__(self, latent_dim=512, n_embeddings=256, **kwargs):
            super().__init__()
            self.latent_dim = latent_dim
            self.n_embeddings = n_embeddings
            self.register_buffer('codebook', torch.linspace(-1, 1, n_embeddings).float())
            
        def forward(self, z, hard=True):
            batch_size, num_ws, w_dim = z.shape
            flat_z = z.reshape(-1, 1)
            distances = torch.abs(flat_z - self.codebook.reshape(1, -1))
            encoding_indices = torch.argmin(distances, dim=1)
            
            # Get the corresponding values from the codebook
            w_discrete_flat = self.codebook[encoding_indices]
            w_discrete = w_discrete_flat.reshape(batch_size, num_ws, w_dim)
            
            return w_discrete, torch.tensor(0.0), encoding_indices


# CABAC Context Modeling
class ContextModel:
    """
    Context model for CABAC encoding/decoding.
    
    Tracks local context statistics to predict symbol probabilities.
    """
    def __init__(self, n_symbols=256, context_size=5, adaptation_rate=0.05):
        self.n_symbols = n_symbols
        self.context_size = context_size
        self.adaptation_rate = adaptation_rate
        
        # Initialize context models
        # We'll use a dictionary to store contexts and their probability models
        self.context_models = defaultdict(lambda: np.ones(n_symbols) / n_symbols)
        
        # Track context statistics
        self.context_counts = defaultdict(int)
    
    def get_context(self, data, pos, shape):
        """
        Extract context around a position.
        
        Args:
            data: Array of symbol indices
            pos: Current position (flat index)
            shape: Original shape of data
            
        Returns:
            Context key (tuple of neighboring symbol indices)
        """
        # Convert flat position to multi-dimensional index
        if len(shape) == 3:  # [batch, num_ws, w_dim]
            batch, ws, dim = np.unravel_index(pos, shape)
            
            # Context includes elements from same batch:
            # 1. Previous symbols in the same w vector
            # 2. Same position in previous w vectors
            # 3. Nearby dimensions in the current w vector
            
            context = []
            
            # Previous element in the same w vector
            if dim > 0:
                context.append(data[batch, ws, dim-1])
            else:
                context.append(-1)  # Sentinel value
                
            # Same position in previous w vector
            if ws > 0:
                context.append(data[batch, ws-1, dim])
            else:
                context.append(-1)  # Sentinel value
                
            # Form context key
            return tuple(context)
        else:
            # Fallback for different shapes
            return ()
    
    def update_model(self, context, symbol):
        """
        Update probability model for a context given observed symbol.
        
        Args:
            context: Context key
            symbol: Observed symbol
        """
        # Get current probability model
        probs = self.context_models[context]
        
        # Update with exponential moving average
        new_probs = probs.copy()
        new_probs[symbol] += self.adaptation_rate * (1.0 - new_probs[symbol])
        
        # Normalize other probabilities
        total_others = new_probs.sum() - new_probs[symbol]
        scale_factor = (1.0 - new_probs[symbol]) / total_others if total_others > 0 else 0
        
        for i in range(self.n_symbols):
            if i != symbol:
                new_probs[i] *= scale_factor
        
        # Update model
        self.context_models[context] = new_probs
        self.context_counts[context] += 1
    
    def get_probability(self, context, symbol=None):
        """
        Get probability for a symbol in a given context.
        
        Args:
            context: Context key
            symbol: Symbol to get probability for (optional)
            
        Returns:
            Probability distribution or probability of specific symbol
        """
        probs = self.context_models[context]
        
        if symbol is not None:
            return probs[symbol]
        else:
            return probs


# Arithmetic Coding Implementation
class ArithmeticCoder:
    """
    Arithmetic coder for CABAC implementation.
    
    Implements binary arithmetic coding for compressing symbol streams.
    """
    def __init__(self):
        # Precision and range parameters
        self.precision = 32
        self.full_range = 1 << self.precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.minimum_range = self.full_range >> 16
        
        # Encoding state
        self.low = 0
        self.high = self.full_range - 1
        self.outstanding_bytes = 0
        
        # Output/input streams
        self.encoded_bytes = bytearray()
        self.current_bit_pos = 0
    
    def _renormalize_encoder(self):
        """Renormalize encoder state and output bits."""
        while (self.high & self.half_range) == (self.low & self.half_range):
            bit = self.high >> (self.precision - 1)
            self.encoded_bytes.append(bit)
            
            # Output outstanding bits
            for _ in range(self.outstanding_bytes):
                self.encoded_bytes.append(1 - bit)
            self.outstanding_bytes = 0
            
            # Scale range
            self.low = (self.low << 1) & (self.full_range - 1)
            self.high = ((self.high << 1) & (self.full_range - 1)) | 1
    
    def _handle_underflow(self):
        """Handle underflow cases in arithmetic coding."""
        while ((self.low & self.quarter_range) != 0) and \
              ((self.high & self.quarter_range) == 0):
            self.outstanding_bytes += 1
            self.low = (self.low << 1) & (self.half_range - 1)
            self.high = ((self.high << 1) & (self.half_range - 1)) | self.full_range | 1
    
    def encode_symbol(self, symbol_prob, cumulative_probs):
        """
        Encode a symbol using arithmetic coding.
        
        Args:
            symbol_prob: Probability of the symbol
            cumulative_probs: Cumulative probabilities for all symbols
        """
        range_size = self.high - self.low + 1
        
        # Update bounds based on probability
        self.high = self.low + int(range_size * cumulative_probs[1] - 1)
        self.low = self.low + int(range_size * cumulative_probs[0])
        
        # Renormalize
        self._renormalize_encoder()
        self._handle_underflow()
    
    def finish_encoding(self):
        """Finish encoding and flush any remaining bits."""
        # Output enough bits to distinguish the final range
        self.outstanding_bytes += 1
        
        if (self.low & self.quarter_range) != 0:
            self.encoded_bytes.append(1)
            for _ in range(self.outstanding_bytes):
                self.encoded_bytes.append(0)
        else:
            self.encoded_bytes.append(0)
            for _ in range(self.outstanding_bytes):
                self.encoded_bytes.append(1)
        
        # Convert bits to bytes
        return bytes(self.encoded_bytes)
    
    def start_decoding(self, encoded_bytes):
        """Initialize decoder with encoded data."""
        self.encoded_bytes = encoded_bytes
        self.current_bit_pos = 0
        self.low = 0
        self.high = self.full_range - 1
        
        # Initialize code value
        self.code_value = 0
        for _ in range(self.precision):
            bit = self._read_bit()
            self.code_value = (self.code_value << 1) | bit
    
    def _read_bit(self):
        """Read a bit from the encoded stream."""
        byte_index = self.current_bit_pos // 8
        bit_index = 7 - (self.current_bit_pos % 8)
        
        if byte_index >= len(self.encoded_bytes):
            return 0
        
        bit = (self.encoded_bytes[byte_index] >> bit_index) & 1
        self.current_bit_pos += 1
        return bit
    
    def decode_symbol(self, cumulative_probs):
        """
        Decode a symbol using arithmetic coding.
        
        Args:
            cumulative_probs: Cumulative probabilities for all symbols
            
        Returns:
            Decoded symbol index
        """
        range_size = self.high - self.low + 1
        
        # Find symbol based on code value
        scaled_value = ((self.code_value - self.low + 1) * 1.0 / range_size) - 1e-10
        
        # Binary search to find the symbol
        symbol = np.searchsorted(cumulative_probs, scaled_value) - 1
        
        # Update bounds
        self.high = self.low + int(range_size * cumulative_probs[symbol+1] - 1)
        self.low = self.low + int(range_size * cumulative_probs[symbol])
        
        # Renormalize
        while (self.high & self.half_range) == (self.low & self.half_range):
            if (self.high & self.half_range) != 0:
                self.low = (self.low << 1) & (self.full_range - 1)
                self.high = ((self.high << 1) & (self.full_range - 1)) | 1
                self.code_value = ((self.code_value << 1) & (self.full_range - 1)) | self._read_bit()
            else:
                self.low = (self.low << 1) & (self.full_range - 1)
                self.high = ((self.high << 1) & (self.full_range - 1)) | 1
                self.code_value = ((self.code_value << 1) & (self.full_range - 1)) | self._read_bit()
        
        # Handle underflow
        while ((self.low & self.quarter_range) != 0) and ((self.high & self.quarter_range) == 0):
            self.low = (self.low << 1) & (self.half_range - 1)
            self.high = ((self.high << 1) & (self.half_range - 1)) | self.full_range | 1
            self.code_value = ((self.code_value ^ self.quarter_range) << 1) | self._read_bit()
        
        return symbol


# CABAC Encoder
def cabac_encode(data, context_model):
    """
    Encode data using CABAC.
    
    Args:
        data: Numpy array of symbol indices
        context_model: ContextModel instance
        
    Returns:
        Encoded bytes
    """
    # Initialize arithmetic coder
    coder = ArithmeticCoder()
    
    # Original shape for context modeling
    original_shape = data.shape
    
    # Flatten data for sequential encoding
    flat_data = data.reshape(-1)
    total_symbols = len(flat_data)
    
    # Encode each symbol
    for i in tqdm(range(total_symbols), desc="CABAC Encoding"):
        # Get context
        context = context_model.get_context(data, i, original_shape)
        
        # Get probabilities
        symbol = flat_data[i]
        probs = context_model.get_probability(context)
        
        # Calculate cumulative probabilities
        cum_probs = np.zeros(len(probs) + 1)
        cum_probs[1:] = np.cumsum(probs)
        
        # Encode symbol
        coder.encode_symbol(probs[symbol], (cum_probs[symbol], cum_probs[symbol+1]))
        
        # Update context model
        context_model.update_model(context, symbol)
    
    # Finish encoding
    encoded_bytes = coder.finish_encoding()
    
    # Return encoded bytes and metadata
    return encoded_bytes


# CABAC Decoder
def cabac_decode(encoded_bytes, context_model, shape):
    """
    Decode data using CABAC.
    
    Args:
        encoded_bytes: Encoded byte stream
        context_model: ContextModel instance
        shape: Shape of the original data
        
    Returns:
        Decoded numpy array
    """
    # Initialize arithmetic decoder
    coder = ArithmeticCoder()
    coder.start_decoding(encoded_bytes)
    
    # Create empty array to hold decoded data
    total_symbols = np.prod(shape)
    decoded_flat = np.zeros(total_symbols, dtype=np.int32)
    
    # Reshape to original shape for context modeling
    decoded = decoded_flat.reshape(shape)
    
    # Decode each symbol
    for i in tqdm(range(total_symbols), desc="CABAC Decoding"):
        # Get context
        context = context_model.get_context(decoded, i, shape)
        
        # Get probabilities
        probs = context_model.get_probability(context)
        
        # Calculate cumulative probabilities
        cum_probs = np.zeros(len(probs) + 1)
        cum_probs[1:] = np.cumsum(probs)
        
        # Decode symbol
        symbol = coder.decode_symbol(cum_probs)
        decoded_flat[i] = symbol
        
        # Update context model
        context_model.update_model(context, symbol)
    
    # Reshape to original shape
    return decoded.reshape(shape)


class CABACCompressor:
    """
    Combines StyleGAN3-HVAE with CABAC compression.
    
    Uses the HVAE encoder and StyleGAN3 generator, with CABAC
    entropy coding for optimized latent code compression.
    """
    def __init__(
        self, 
        encoder, 
        generator, 
        discretization=None,
        n_embeddings=256,
        training_resolution=None
    ):
        self.encoder = encoder
        self.generator = generator
        self.training_resolution = training_resolution
        
        # Create or use discretization layer
        if discretization is None:
            self.discretization = GumbelSoftmaxDiscretization(
                latent_dim=encoder.w_dim,
                n_embeddings=n_embeddings
            )
        else:
            self.discretization = discretization
        
        # Context model for CABAC
        self.context_model = ContextModel(n_symbols=n_embeddings)
    
    def encode(self, x, deterministic=True):
        """Encode an image to discretized W+ space."""
        w_plus, means, _ = self.encoder(x)
        
        if deterministic:
            w_discrete, _, _ = self.discretization(means, hard=True)
        else:
            w_discrete, _, _ = self.discretization(w_plus, hard=True)
            
        return w_discrete
    
    def compress(self, x, use_cabac=True):
        """
        Compress an image to encoded latent representation.
        
        Args:
            x: Input image tensor
            use_cabac: Whether to use CABAC entropy coding
            
        Returns:
            Compressed bytes, metadata
        """
        with torch.no_grad():
            # Encode image to get w+ vector
            w_plus, means, _ = self.encoder(x)
            
            # Discretize latent
            _, _, indices = self.discretization(means, hard=True)
            
            # Convert to numpy array for CABAC encoding
            batch_size, num_ws, w_dim = w_plus.shape
            codes = indices.reshape(batch_size, num_ws, w_dim).cpu().numpy().astype(np.int32)
            
            # Calculate original size (before compression)
            orig_size = codes.size * np.log2(self.discretization.n_embeddings) / 8
            
            if use_cabac:
                # CABAC encode
                encoded_bytes = cabac_encode(codes, self.context_model)
                comp_size = len(encoded_bytes)
            else:
                # Simple encoding (for comparison)
                encoded_bytes = codes.tobytes()
                comp_size = len(encoded_bytes)
            
            # Create metadata
            metadata = {
                'shape': codes.shape,
                'n_embeddings': self.discretization.n_embeddings,
                'use_cabac': use_cabac,
                'orig_size': orig_size,
                'comp_size': comp_size,
                'compression_ratio': orig_size / comp_size
            }
            
            return encoded_bytes, metadata
    
    def decompress(self, encoded_bytes, metadata, noise_mode='const'):
        """
        Decompress latent representation to an image.
        
        Args:
            encoded_bytes: Compressed bytes
            metadata: Compression metadata
            noise_mode: Noise mode for StyleGAN3 synthesis
            
        Returns:
            Reconstructed image
        """
        with torch.no_grad():
            # Get shape and parameters from metadata
            shape = metadata['shape']
            use_cabac = metadata.get('use_cabac', True)
            
            # Decode latent codes
            if use_cabac:
                # CABAC decode
                codes = cabac_decode(encoded_bytes, self.context_model, shape)
            else:
                # Simple decoding
                codes = np.frombuffer(encoded_bytes, dtype=np.int32).reshape(shape)
            
            # Convert to tensor
            device = next(self.generator.parameters()).device
            codes_tensor = torch.from_numpy(codes).to(device)
            
            # Convert indices to latent values using codebook
            batch_size, num_ws, w_dim = shape
            flat_codes = codes_tensor.reshape(-1)
            
            # Look up values in codebook
            w_discrete_flat = self.discretization.codebook[flat_codes]
            w_discrete = w_discrete_flat.reshape(batch_size, num_ws, w_dim)
            
            # Generate image using StyleGAN3
            img = self.generator.synthesis(w_discrete, noise_mode=noise_mode)
            
            return img
    
    def save_compressed(self, x, filename, use_cabac=True):
        """
        Save compressed representation of an image.
        
        Args:
            x: Input image tensor
            filename: Output filename (.cabac)
            use_cabac: Whether to use CABAC encoding
            
        Returns:
            Compression statistics
        """
        # Compress image
        encoded_bytes, metadata = self.compress(x, use_cabac=use_cabac)
        
        # Write compressed data
        with open(filename, 'wb') as f:
            # Write metadata
            f.write(struct.pack('I', len(metadata)))
            f.write(pickle.dumps(metadata))
            
            # Write compressed data
            f.write(encoded_bytes)
        
        return metadata['orig_size'], metadata['comp_size'], metadata['compression_ratio']
    
    def load_compressed(self, filename, noise_mode='const'):
        """
        Load and decompress an image from a compressed file.
        
        Args:
            filename: Input filename (.cabac)
            noise_mode: Noise mode for StyleGAN3 synthesis
            
        Returns:
            Reconstructed image, compression ratio
        """
        # Load compressed data
        with open(filename, 'rb') as f:
            # Read metadata
            metadata_len = struct.unpack('I', f.read(4))[0]
            metadata = pickle.loads(f.read(metadata_len))
            
            # Read compressed data
            encoded_bytes = f.read()
        
        # Decompress
        img = self.decompress(encoded_bytes, metadata, noise_mode=noise_mode)
        
        return img, metadata['compression_ratio']


# Training or loading a model
def load_or_train_cabac_compressor(
    generator_pkl,
    output_dir='./output_cabac',
    checkpoint_path=None,
    training_resolution=256,
    device_override=None,
    use_gumbel=True,
    n_embeddings=256
):
    """
    Load or train a compressor with CABAC support.
    
    Args:
        generator_pkl: Path to StyleGAN3 generator pickle
        output_dir: Directory to save results
        checkpoint_path: Path to existing checkpoint
        training_resolution: Training resolution
        device_override: Override device selection
        use_gumbel: Whether to use Gumbel-Softmax discretization
        n_embeddings: Number of discrete embeddings
        
    Returns:
        Compressor with CABAC support
    """
    import os.path as osp
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    if device_override is not None:
        device = torch.device(device_override)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) on Mac")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print(f"Using device: {device}")
    
    # Load StyleGAN3 generator
    print(f"Loading StyleGAN3 generator from {generator_pkl}")
    with open(generator_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    
    # Load or create encoder and discretization
    if checkpoint_path is not None and osp.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create encoder
        encoder = HVAE_VGG_Encoder(
            img_resolution=1024,
            img_channels=checkpoint['config']['img_channels'],
            w_dim=checkpoint['config']['w_dim'],
            num_ws=checkpoint['config']['num_ws'],
            block_split=checkpoint['config']['block_split'],
            channel_base=32768,
            channel_max=512,
        ).to(device)
        
        # Load encoder weights
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Create discretization layer
        if 'discretization_state_dict' in checkpoint and use_gumbel:
            discretization = GumbelSoftmaxDiscretization(
                latent_dim=encoder.w_dim,
                n_embeddings=checkpoint['config'].get('n_embeddings', n_embeddings),
            ).to(device)
            discretization.load_state_dict(checkpoint['discretization_state_dict'])
            print("Loaded Gumbel-Softmax discretization from checkpoint")
        else:
            discretization = GumbelSoftmaxDiscretization(
                latent_dim=encoder.w_dim,
                n_embeddings=n_embeddings,
            ).to(device)
            print("Created new discretization layer")
        
        print("Loaded model from checkpoint")
    else:
        print("No checkpoint provided, creating new encoder")
        encoder = HVAE_VGG_Encoder(
            img_resolution=1024,
            img_channels=G.img_channels,
            w_dim=G.w_dim,
            num_ws=G.num_ws,
            block_split=(5, 12),
            channel_base=32768,
            channel_max=512,
        ).to(device)
        
        discretization = GumbelSoftmaxDiscretization(
            latent_dim=encoder.w_dim,
            n_embeddings=n_embeddings,
        ).to(device)
    
    # Create CABAC compressor
    compressor = CABACCompressor(
        encoder, 
        G, 
        discretization=discretization,
        n_embeddings=n_embeddings,
        training_resolution=training_resolution
    )
    
    return compressor


def compress_image(
    compressor,
    image_path,
    output_path,
    use_cabac=True,
    resolution=256
):
    """
    Compress an image using the CABAC compressor.
    
    Args:
        compressor: CABACCompressor instance
        image_path: Path to input image
        output_path: Path to output compressed file
        use_cabac: Whether to use CABAC encoding
        resolution: Resolution for compression
        
    Returns:
        Compression statistics
    """
    from PIL import Image
    from torchvision import transforms
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Convert to tensor
    img_tensor = transform(img).unsqueeze(0)
    
    # Move to correct device
    device = next(compressor.encoder.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Compress
    start_time = time.time()
    orig_size, comp_size, compression_ratio = compressor.save_compressed(
        img_tensor, output_path, use_cabac=use_cabac
    )
    compress_time = time.time() - start_time
    
    # Print statistics
    print(f"Compression results for {image_path}:")
    print(f"  Original size: {orig_size / 1024:.2f} KB")
    print(f"  Compressed size: {comp_size / 1024:.2f} KB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Compression time: {compress_time:.2f} seconds")
    
    return orig_size, comp_size, compression_ratio


def decompress_image(
    compressor,
    compressed_path,
    output_path,
    noise_mode='const'
):
    """
    Decompress an image using the CABAC compressor.
    
    Args:
        compressor: CABACCompressor instance
        compressed_path: Path to compressed file
        output_path: Path to output image
        noise_mode: Noise mode for StyleGAN3 synthesis
        
    Returns:
        Decompression statistics
    """
    from torchvision.utils import save_image
    
    # Decompress
    start_time = time.time()
    img, compression_ratio = compressor.load_compressed(
        compressed_path, noise_mode=noise_mode
    )
    decompress_time = time.time() - start_time
    
    # Save image
    save_image((img.cpu() + 1) / 2, output_path)
    
    # Print statistics
    print(f"Decompression results for {compressed_path}:")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Decompression time: {decompress_time:.2f} seconds")
    
    return compression_ratio, decompress_time


def compare_compression_methods(
    compressor,
    image_path,
    output_dir,
    resolution=256
):
    """
    Compare different compression methods on an image.
    
    Args:
        compressor: CABACCompressor instance
        image_path: Path to input image
        output_dir: Directory to save outputs
        resolution: Resolution for compression
        
    Returns:
        Compression statistics
    """
    from PIL import Image
    import os.path as osp
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    # Original file size
    orig_file_size = os.path.getsize(image_path)
    
    # Save as PNG and JPEG for comparison
    img_resized = img.resize((resolution, resolution), Image.LANCZOS)
    
    png_path = osp.join(output_dir, 'comparison_png.png')
    jpg_path = osp.join(output_dir, 'comparison_jpg.jpg')
    
    img_resized.save(png_path, format='PNG')
    img_resized.save(jpg_path, format='JPEG', quality=90)
    
    png_size = os.path.getsize(png_path)
    jpg_size = os.path.getsize(jpg_path)
    
    # HVAE with standard entropy coding
    hvae_path = osp.join(output_dir, 'comparison_hvae.cabac')
    _, hvae_size, hvae_ratio = compress_image(
        compressor, image_path, hvae_path, use_cabac=False, resolution=resolution
    )
    
    # HVAE with CABAC
    cabac_path = osp.join(output_dir, 'comparison_cabac.cabac')
    _, cabac_size, cabac_ratio = compress_image(
        compressor, image_path, cabac_path, use_cabac=True, resolution=resolution
    )
    
    # Decompress and save reconstructed images
    decompress_image(
        compressor, hvae_path, osp.join(output_dir, 'comparison_hvae_reconstructed.png')
    )
    
    decompress_image(
        compressor, cabac_path, osp.join(output_dir, 'comparison_cabac_reconstructed.png')
    )
    
    # Print comparison
    print("\nCompression Method Comparison:")
    print(f"Original file: {orig_file_size / 1024:.2f} KB")
    print(f"PNG (lossless): {png_size / 1024:.2f} KB, {orig_file_size / png_size:.2f}x ratio")
    print(f"JPEG (quality 90): {jpg_size / 1024:.2f} KB, {orig_file_size / jpg_size:.2f}x ratio")
    print(f"HVAE standard: {hvae_size / 1024:.2f} KB, {hvae_ratio:.2f}x ratio")
    print(f"HVAE with CABAC: {cabac_size / 1024:.2f} KB, {cabac_ratio:.2f}x ratio")
    print(f"CABAC improvement over standard: {hvae_size / cabac_size:.2f}x")
    
    return {
        'original': orig_file_size,
        'png': png_size,
        'jpg': jpg_size,
        'hvae': hvae_size,
        'cabac': cabac_size,
        'hvae_ratio': hvae_ratio,
        'cabac_ratio': cabac_ratio,
        'cabac_vs_hvae': hvae_size / cabac_size
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CABAC Compression for StyleGAN3-HVAE")
    parser.add_argument("--generator", type=str, default="models/stylegan3-t-ffhq-1024x1024.pkl",
                       help="Path to StyleGAN3 generator pickle")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to encoder checkpoint (optional)")
    parser.add_argument("--output", type=str, default="./output_cabac",
                       help="Output directory")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to input image (for compression)")
    parser.add_argument("--compressed", type=str, default=None,
                       help="Path to compressed file (for decompression)")
    parser.add_argument("--resolution", type=int, default=256,
                       help="Resolution for compression/decompression")
    parser.add_argument("--no_cabac", action="store_true",
                       help="Disable CABAC encoding (for comparison)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare different compression methods")
    parser.add_argument("--device", type=str, default=None,
                       help="Override device selection")
    
    args = parser.parse_args()
    
    # Load compressor
    compressor = load_or_train_cabac_compressor(
        generator_pkl=args.generator,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        training_resolution=args.resolution,
        device_override=args.device
    )
    
    # Perform compression or decompression
    if args.compare and args.image:
        # Compare different compression methods
        compare_compression_methods(
            compressor,
            args.image,
            args.output,
            resolution=args.resolution
        )
    elif args.compressed:
        # Decompress
        output_image = osp.join(args.output, 'decompressed.png')
        decompress_image(
            compressor,
            args.compressed,
            output_image
        )
        print(f"Decompressed image saved to {output_image}")
    elif args.image:
        # Compress
        output_compressed = osp.join(args.output, 'compressed.cabac')
        compress_image(
            compressor,
            args.image,
            output_compressed,
            use_cabac=not args.no_cabac,
            resolution=args.resolution
        )
        print(f"Compressed file saved to {output_compressed}")
    else:
        print("Please provide either --image or --compressed argument.")
        print("For comparison, use --compare --image <path>")
        
    print("CABAC compression completed!")