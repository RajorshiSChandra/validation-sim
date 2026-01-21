#!/usr/bin/env python3
"""Generate Zernike beam coefficient files."""
import argparse
import json
import numpy as np
from pathlib import Path

# Zernike term names for reference
ZERNIKE_NAMES = {
    0: "Piston (amplitude)",
    1: "X-Tilt (pointing X)",
    2: "Y-Tilt (pointing Y)",
    3: "Oblique Astigmatism (45° elongation)",
    4: "Defocus (beam width)",
    5: "Vertical Astigmatism (X/Y elongation)",
    6: "Vertical Trefoil",
    7: "Vertical Coma (X)",
    8: "Horizontal Coma (Y)",
    9: "Oblique Trefoil",
    10: "Oblique Quadrafoil",
    11: "Oblique 2nd Astigmatism",
    12: "Primary Spherical (sidelobe level)",
    13: "Vertical 2nd Astigmatism",
    14: "Vertical Quadrafoil",
}

def make_symmetric_beam(defocus=-0.25, spherical=0.03, n_coeffs=15):
    """Create a simple symmetric beam."""
    coeffs = np.zeros(n_coeffs)
    coeffs[0] = 1.0       # Piston (amplitude)
    coeffs[4] = defocus   # Defocus (beam width)
    coeffs[12] = spherical if n_coeffs > 12 else 0  # Spherical (sidelobes)
    return coeffs.tolist()

def make_elliptical_beam(defocus=-0.25, astigmatism=0.1, spherical=0.03, n_coeffs=15):
    """Create an elliptical beam."""
    coeffs = np.zeros(n_coeffs)
    coeffs[0] = 1.0
    coeffs[4] = defocus
    coeffs[5] = astigmatism  # Vertical astigmatism
    coeffs[12] = spherical if n_coeffs > 12 else 0
    return coeffs.tolist()

def make_tilted_beam(x_tilt=0.05, y_tilt=0.02, defocus=-0.25, n_coeffs=15):
    """Create a beam with pointing error."""
    coeffs = np.zeros(n_coeffs)
    coeffs[0] = 1.0
    coeffs[1] = x_tilt
    coeffs[2] = y_tilt
    coeffs[4] = defocus
    return coeffs.tolist()

def make_coma_beam(defocus=-0.25, coma_x=0.05, coma_y=0.03, n_coeffs=15):
    """Create a beam with coma aberration."""
    coeffs = np.zeros(n_coeffs)
    coeffs[0] = 1.0
    coeffs[4] = defocus
    coeffs[7] = coma_x   # Vertical coma
    coeffs[8] = coma_y   # Horizontal coma
    return coeffs.tolist()

def make_random_perturbation(seed=42, scale=0.05, n_coeffs=15):
    """Create random perturbations on top of a base beam."""
    rng = np.random.default_rng(seed)
    coeffs = np.zeros(n_coeffs)
    coeffs[0] = 1.0
    coeffs[4] = -0.25  # Base defocus
    # Add small random perturbations to higher-order terms
    coeffs[1:] += rng.normal(0, scale, n_coeffs - 1)
    coeffs[0] = 1.0  # Keep piston fixed
    return coeffs.tolist()

def main():
    parser = argparse.ArgumentParser(description="Generate Zernike beam coefficient files")
    parser.add_argument("--type", choices=["symmetric", "elliptical", "tilted", "coma", "random", "custom"],
                        default="symmetric", help="Type of beam to generate")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output file path (.json or .yaml)")
    parser.add_argument("--n-coeffs", type=int, default=15, help="Number of Zernike coefficients")
    parser.add_argument("--defocus", type=float, default=-0.25, help="Defocus coefficient (beam width)")
    parser.add_argument("--astigmatism", type=float, default=0.1, help="Astigmatism coefficient")
    parser.add_argument("--x-tilt", type=float, default=0.05, help="X-tilt (pointing error)")
    parser.add_argument("--y-tilt", type=float, default=0.02, help="Y-tilt (pointing error)")
    parser.add_argument("--spherical", type=float, default=0.03, help="Spherical aberration (sidelobes)")
    parser.add_argument("--coma-x", type=float, default=0.05, help="Vertical coma")
    parser.add_argument("--coma-y", type=float, default=0.03, help="Horizontal coma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random type")
    parser.add_argument("--scale", type=float, default=0.05, help="Scale of random perturbations")
    parser.add_argument("--coeffs", type=float, nargs="+", help="Custom coefficients (for --type custom)")
    parser.add_argument("--list-terms", action="store_true", help="List Zernike term names and exit")
    
    args = parser.parse_args()
    
    if args.list_terms:
        print("\nZernike Terms (index: name):")
        print("-" * 40)
        for i, name in ZERNIKE_NAMES.items():
            print(f"  [{i:2d}] {name}")
        print("-" * 40)
        return
    
    # Generate coefficients
    if args.type == "symmetric":
        coeffs = make_symmetric_beam(args.defocus, args.spherical, args.n_coeffs)
    elif args.type == "elliptical":
        coeffs = make_elliptical_beam(args.defocus, args.astigmatism, args.spherical, args.n_coeffs)
    elif args.type == "tilted":
        coeffs = make_tilted_beam(args.x_tilt, args.y_tilt, args.defocus, args.n_coeffs)
    elif args.type == "coma":
        coeffs = make_coma_beam(args.defocus, args.coma_x, args.coma_y, args.n_coeffs)
    elif args.type == "random":
        coeffs = make_random_perturbation(args.seed, args.scale, args.n_coeffs)
    elif args.type == "custom":
        if args.coeffs is None:
            parser.error("--coeffs required for --type custom")
        coeffs = list(args.coeffs)
    
    # Create output data
    data = {"beam_coeffs": coeffs}
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    if str(args.output).endswith(".yaml") or str(args.output).endswith(".yml"):
        import yaml
        with open(args.output, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
    
    print(f"Wrote {len(coeffs)} Zernike coefficients to {args.output}")
    print(f"Coefficients: {coeffs}")

if __name__ == "__main__":
    main()