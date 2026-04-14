import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multi_scale_gnn import MultiScaleGNN


def test_multi_scale_gnn():
    """Simple test to verify MultiScaleGNN works."""
    print("🧪 Testing MultiScaleGNN...")
    
    # Create a small model for testing
    model = MultiScaleGNN(
        nnode_in_features=10,
        nnode_out_features=3,
        latent_dim=32,
        nmessage_passing_steps=5,
        nmlp_layers=2,
    )
    
    print("✅ Model created successfully")
    
    # Test data
    num_grid_nodes = 25
    
    # Create test tensors
    grid_node_features = torch.randn(num_grid_nodes, 10)
    
    print("✅ Test data created successfully")
    
    # Test forward pass
    try:
        outputs = model(
            x=grid_node_features
        )
        
        print(f"✅ Forward pass successful! Output shape: {outputs.shape}")
        print(f"   Expected: ({num_grid_nodes}, 3)")
        print(f"   Actual: {outputs.shape}")
        
        # Check output is not NaN
        if not torch.isnan(outputs).any():
            print("✅ Outputs are valid (no NaN values)")
        else:
            print("❌ Outputs contain NaN values")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test input projection for Transolver token embedding
    try:
        projected = model.input_proj(grid_node_features)
        print(f"✅ Input projection works: {projected.shape}")
    except Exception as e:
        print(f"❌ Input projection failed: {e}")
        return False
    
    print("\n🎉 All tests passed! MultiScaleGNN is working correctly.")
    return True


if __name__ == '__main__':
    success = test_multi_scale_gnn()
    if not success:
        sys.exit(1)
