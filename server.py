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

# Global solver instance and state
solver = TorusSpectralSolver(R=1.5, r=0.5, c=1.0, N_theta=256, N_phi=256, CFL=0.1)
device = torch.device('cpu') 

P_prev = torch.zeros((solver.N_theta, solver.N_phi), device=device)
P_curr = torch.zeros((solver.N_theta, solver.N_phi), device=device)

@app.get("/reset")
async def reset():
    global P_prev, P_curr
    P_prev.zero_()
    P_curr.zero_()
    return {"status": "ok"}

@app.get("/step")
async def step(theta0: float = -1.0, phi0: float = -1.0, steps: int = 15):
    """
    Advances the global simulation by 'steps' iterations and streams back the current frame.
    Optionally injects a pulse at (theta0, phi0) for one internal step.
    """
    global P_prev, P_curr
    
    for i in range(steps):
        S_curr = torch.zeros_like(P_curr)
        if theta0 >= 0 and phi0 >= 0 and i == 0:
            # Inject pulse only on the first step of this batch
            theta_1d = torch.linspace(0, 2*np.pi, solver.N_theta + 1, device=device)[:-1]
            phi_1d = torch.linspace(0, 2*np.pi, solver.N_phi + 1, device=device)[:-1]
            THETA, PHI = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
            
            dtheta = (THETA - theta0 + np.pi) % (2*np.pi) - np.pi
            dphi = (PHI - phi0 + np.pi) % (2*np.pi) - np.pi
            
            dist_sq = dtheta**2 + dphi**2
            sigma = 0.05
            r_sq_over_sigma_sq = dist_sq / (sigma ** 2)
            # Ricker / Mexican Hat: (2 - r^2/sigma^2) * exp(-r^2/(2*sigma^2))
            spatial = (2.0 - r_sq_over_sigma_sq) * torch.exp(-dist_sq / (2 * sigma ** 2))
            # Subtract mean to ensure zero-mean baseline
            spatial = spatial - spatial.mean()
            S_curr += spatial * 10000.0
            
        laplacian = solver.compute_laplace_beltrami(P_curr)
        accel = (solver.c**2) * (laplacian + S_curr)
        P_next = 2 * P_curr - P_prev + (solver.dt**2) * accel
        
        P_prev = P_curr
        P_curr = P_next
        
    last_frame = P_curr.cpu().numpy().astype(np.float32)
    return Response(content=last_frame.tobytes(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

