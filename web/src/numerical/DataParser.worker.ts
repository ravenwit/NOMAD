// DataParser.worker.ts

// This worker downloads and parses large JSON datasets into Float32Arrays 
// so the main thread doesn't freeze.

self.onmessage = async (e: MessageEvent) => {
    const { url, type } = e.data;
    
    try {
        self.postMessage({ type: 'progress', status: 'Fetching data...', progress: 0 });
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
        }
        
        // Wait, for 150MB+ files we can use standard .json() 
        // since we are in a worker and won't freeze the UI.
        self.postMessage({ type: 'progress', status: 'Parsing JSON...', progress: 30 });
        const jsonText = await response.text();
        
        self.postMessage({ type: 'progress', status: 'Decoding data...', progress: 60 });
        const data = JSON.parse(jsonText);
        
        self.postMessage({ type: 'progress', status: 'Flattening to binary...', progress: 80 });
        
        // Assuming data is of shape [frames][256][256] or [frames][128][128]
        // Let's dynamically detect dimensions
        const frames = data.length;
        if (frames === 0) throw new Error("Empty data array");
        const height = data[0].length;
        const width = data[0][0].length;
        
        const totalSize = frames * height * width;
        const flatBuffer = new Float32Array(totalSize);
        
        let idx = 0;
        for (let t = 0; t < frames; t++) {
            const frame = data[t];
            for (let j = 0; j < height; j++) {
                const row = frame[j];
                for (let i = 0; i < width; i++) {
                    flatBuffer[idx++] = row[i];
                }
            }
        }
        
        // Send back the flattened Float32Array and dimensions
        // Use Transferable object for zero-copy
        self.postMessage({
            type: 'complete',
            payload: {
                buffer: flatBuffer,
                frames,
                height,
                width,
                dataType: type
            }
        }, { transfer: [flatBuffer.buffer] });
        
    } catch (err: any) {
        self.postMessage({ type: 'error', error: err.message });
    }
};
