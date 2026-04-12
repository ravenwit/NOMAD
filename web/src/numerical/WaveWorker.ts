import { TSSolver } from "./SpectralSolver";

let solver: TSSolver | null = null;

self.onmessage = (e: MessageEvent) => {
  const { type, ...payload } = e.data;

  switch (type) {
    case 'init':
      solver = new TSSolver(
        payload.R,
        payload.r,
        payload.c,
        payload.width,
        payload.height,
        payload.CFL
      );
      break;

    case 'step':
      if (solver) {
        solver.step(payload.steps || 1);
        const data = solver.getFloat32Array();
        // Use Transferable to avoid copy overhead
        (self as any).postMessage({ type: 'frame', data }, [data.buffer]);
      }
      break;

    case 'pulse':
      if (solver) {
        solver.injectPulse(payload.theta0, payload.phi0, payload.impulse);
      }
      break;

    case 'reset':
      if (solver) {
        solver.reset();
      }
      break;
  }
};
