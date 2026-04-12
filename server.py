from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import torch
import numpy as np
import uvicorn
from src.numerical.solver import TorusSpectralSolver

app = FastAPI()

# Allow requests from the web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global solver instance (can be re-initialized per request if needed)
# Defaulting to 256x256 for high fidelity
solver = TorusSpectralSolver(R=1.5, r=0.5, c=1.0, N_theta=256, N_phi=256, CFL=0.1)

@app.get("/simulate")
async def simulate(theta0: float = 0.0, phi0: float = 0.0, steps: int = 100):
    """
    Runs the spectral simulation and returns the final frame as binary Float32 data.
    """
    def source_fn(t, dev):
        if t > 0.1: return torch.zeros((solver.N_theta, solver.N_phi), device=dev)
        
        # Grid setup for source
        theta_1d = torch.linspace(0, 2*np.pi, solver.N_theta + 1, device=dev)[:-1]
        phi_1d = torch.linspace(0, 2*np.pi, solver.N_phi + 1, device=dev)[:-1]
        THETA, PHI = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
        
        # Periodic distance
        dtheta = (THETA - theta0 + np.pi) % (2*np.pi) - np.pi
        dphi = (PHI - phi0 + np.pi) % (2*np.pi) - np.pi
        
        dist_sq = dtheta**2 + dphi**2
        return torch.exp(-dist_sq / (2 * 0.1**2)) * 10.0

    device = torch.device('cpu') # Use CPU for stability in simple backend
    # Run simulation
    # We only take the last frame for this simple bridge
    history = solver.simulate(num_steps=steps, source_fn=source_fn, device=device, record_every=steps)
    
    last_frame = history[-1].numpy().astype(np.float32)
    
    # Return as raw bytes
    return Response(content=last_frame.tobytes(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
