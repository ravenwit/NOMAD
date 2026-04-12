import argparse
import torch

from src.simulator import TorusAcousticSimulator
from src.models import AutoregressiveGenerator, Discriminator, SINDyAutoencoder
from src.train import PhysicsInformedTrainer, TorusWaveDataset
from torch.utils.data import DataLoader
from src.geometry import TorusGeometry
from src.discovery import discover_pde

def main():
    parser = argparse.ArgumentParser(description="Acoustic Wave Torus Geometric Deep Learning Framework")
    parser.add_argument('--mode', type=str, choices=['simulate', 'train-gan', 'discover', 'export-onnx'], required=True)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    if args.mode == 'simulate':
        print("Starting Acoustic Wave Simulation on Torus...")
        simulator = TorusAcousticSimulator(R=1.0, r=0.3, c=1.0, N_theta=64, N_phi=64, dt=0.005)
        
        # Simple Gaussian impact source
        def source_func(t, d):
            return simulator.generate_gaussian_source(t, t0=0.05, sigma_t=0.01, 
                                                      theta0=3.14, phi0=3.14, sigma_s=0.5, 
                                                      amplitude=torch.ones(3, device=d), device=d)
        
        P_seq, S_seq = simulator.simulate(num_steps=500, source_generator_fn=source_func, device=device)
        print(f"Simulation completed. P shape: {P_seq.shape}")
        simulator.save_to_h5(P_seq, S_seq, "torus_simulation.h5")
        
    elif args.mode == 'train-gan':
        print("Initializing Physics-Informed GAN Training...")
        generator = AutoregressiveGenerator(input_channels=3).to(device)
        discriminator = Discriminator(input_channels=3).to(device)
        
        # Load Dataset
        try:
            dataset = TorusWaveDataset("torus_simulation.h5")
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            print(f"Loaded dataset with {len(dataset)} samples.")
        except FileNotFoundError:
            print("Error: simulation_data.h5 not found. Run --mode simulate first.")
            return

        trainer = PhysicsInformedTrainer(generator, discriminator, R=1.0, r=0.3, c=1.0)
        trainer.train_epochs(dataloader, epochs=10)
        
        # Save trained weights
        torch.save(generator.state_dict(), "generator_weights.pth")
        print("Training complete. Weights saved to generator_weights.pth")
        
    elif args.mode == 'discover':
        print("Initializing Symbolic PDE Discovery...")
        autoencoder = SINDyAutoencoder(input_dim=(64, 64), channels=3, latent_dim=8).to(device)
        
        # Mock sequence testing
        P_sequence = torch.randn(2, 50, 3, 64, 64, device=device) 
        
        result = discover_pde(autoencoder, P_sequence, dt=0.005)
        print(f"Discovery Result: {result}")
        
    elif args.mode == 'export-onnx':
        print("Exporting AutoregressiveGenerator to ONNX...")
        generator = AutoregressiveGenerator(input_channels=3).to(device)
        # Note: Load your trained weights via generator.load_state_dict(...) before exporting in production!
        generator.export_to_onnx("acoustic_torus.onnx", device=device)

if __name__ == "__main__":
    main()
