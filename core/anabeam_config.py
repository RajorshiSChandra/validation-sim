import json
from pathlib import Path

def read_analytic_beam_map(map_file: Path) -> tuple[dict[int, dict], dict[int, int]]:
    """
    Read an analytic beam map YAML file.
    
    Returns
    -------
    beam_definitions : dict[int, dict]
        Mapping from beam_id to full beam config dict (class, ref_freq, beam_coeffs, etc.)
    antenna_to_beamid : dict[int, int]
        Mapping from antenna number to beam_id
    """
    import yaml
    with open(map_file) as f:
        data = yaml.safe_load(f)
    
    defaults = data.get("defaults", {})
    beam_defs_raw = data.get("beam_definitions", {})
    antenna_mapping = data.get("antenna_mapping", {})
    default_beam_id = data.get("default_beam_id", 0)
    
    # Merge defaults with per-beam overrides
    beam_definitions = {}
    for beam_id, beam_params in beam_defs_raw.items():
        full_config = dict(defaults)
        full_config.update(beam_params)
        beam_definitions[int(beam_id)] = full_config
    
    antenna_to_beamid = {int(ant): int(bid) for ant, bid in antenna_mapping.items()}
    beam_definitions["_default_beam_id"] = default_beam_id
    
    return beam_definitions, antenna_to_beamid

def build_analytic_beam_config(
    beam_class: str,
    diameter: float = 14.0,
    sigma: float = 0.15,
    ref_freq: float = 1e8,
    spectral_index: float = -0.6975,
    coeffs_file: Path | None = None,
    preset: str | None = None,
) -> dict:
    """Build analytical beam configuration dict for YAML output."""
    
    config = {"class": beam_class}
    
    if beam_class == "AiryBeam":
        config["diameter"] = diameter
        
    elif beam_class == "GaussianBeam":
        config["sigma"] = sigma
        
    elif beam_class in ("hera_sim.beams.PolyBeam", "hera_sim.beams.ZernikeBeam", 
                        "hera_sim.beams.PerturbedPolyBeam"):
        config["ref_freq"] = ref_freq
        config["spectral_index"] = spectral_index
        
        # Get beam coefficients
        if preset == "fagnoni19":
            config["beam_coeffs"] = [
                0.29778665, -0.44821433, 0.27338272, -0.10030698,
                -0.01195859, 0.06063853, -0.04593295, 0.0107879,
                0.01390283, -0.01881641, -0.00177106, 0.01265177,
                -0.00568299, -0.00333975, 0.00452368, 0.00151808,
                -0.00593812, 0.00351559,
            ]
        elif coeffs_file is not None:
            # Load from file (JSON or YAML)
            with open(coeffs_file) as f:
                if str(coeffs_file).endswith('.json'):
                    data = json.load(f)
                else:
                    import yaml
                    data = yaml.safe_load(f)
                config["beam_coeffs"] = data.get("beam_coeffs", data.get("coeffs", data))
        else:
            raise ValueError(
                f"For {beam_class}, provide --analytic-beam-preset or --analytic-beam-coeffs-file"
            )
    
    return config