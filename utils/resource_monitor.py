"""Benchmarking utilities for monitoring GPU memory and runtime."""

import time
import torch


class ResourceMonitor:
    """Monitor GPU memory and runtime."""
    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.max_memory = 0
        self.start_time = None
        
    def start(self):
        """Start monitoring."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self.start_time = time.time()
        
    def stop(self):
        """Stop monitoring and return stats."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        elapsed_time = time.time() - self.start_time
        
        if self.is_cuda:
            # Use max_memory_reserved to match nvidia-smi (includes cached memory)
            max_memory_mb = torch.cuda.max_memory_reserved(self.device) / 1024 / 1024
            current_memory_mb = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            # Also track allocated (actual tensor memory)
            max_allocated_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        else:
            max_memory_mb = 0
            current_memory_mb = 0
            max_allocated_mb = 0
            
        return {
            'elapsed_time': elapsed_time,
            'max_memory_mb': max_memory_mb,  # Total reserved (matches nvidia-smi)
            'current_memory_mb': current_memory_mb,
            'max_allocated_mb': max_allocated_mb  # Actual tensor memory
        }
    
    def get_current_memory(self):
        """Get current GPU memory usage in MB (reserved, matches nvidia-smi)."""
        if self.is_cuda:
            return torch.cuda.memory_reserved(self.device) / 1024 / 1024
        return 0

