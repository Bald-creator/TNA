"""
Atlas Configuration Module
Centralized configuration for brain atlases (CC200, AAL116, etc.)
"""

# Atlas configurations with metadata
ATLAS_CONFIGS = {
    'cc200': {
        'num_nodes': 200,
        'rda_file': 'craddock200.rda',
        'rda_key': 'craddock200',
        'rearrange_file': 'CC200_Yeo7_rearrangement.pkl',
        'comm_boundaries': [0, 48, 71, 91, 113, 129, 152, 200],  # Yeo-7 network
        'num_communities': 7,
    },
    'aal116': {
        'num_nodes': 116,
        'rda_file': 'aal116.rda',
        'rda_key': 'aal116',
        'rearrange_file': 'AAL_Yeo7_rearrangement.pkl',
        'comm_boundaries': [0, 40, 52, 57, 69, 82, 91, 116],  # Yeo-7 network
        'num_communities': 7,
    }
}


def get_atlas_config(atlas_name='cc200'):
    """
    Get configuration for specified atlas
    
    Args:
        atlas_name: Name of the atlas ('cc200' or 'aal116')
        
    Returns:
        dict: Atlas configuration
        
    Raises:
        ValueError: If atlas_name is not supported
    """
    if atlas_name not in ATLAS_CONFIGS:
        raise ValueError(
            f"Unknown atlas: {atlas_name}. "
            f"Available: {list(ATLAS_CONFIGS.keys())}"
        )
    return ATLAS_CONFIGS[atlas_name]


def get_available_atlases():
    """Get list of available atlas names"""
    return list(ATLAS_CONFIGS.keys())

