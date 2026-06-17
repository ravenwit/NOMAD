import { Client } from "@gradio/client";

async function test() {
    try {
        console.log("Connecting to HF space...");
        const client = await Client.connect("raven-shakir/nomad");
        console.log("Connected. Sending predict...");
        const result = await client.predict("/run_simulation", {
            theta: Math.PI,
            phi: Math.PI
        });
        const data = result.data[0];
        console.log("Data type:", typeof data);
        if (Array.isArray(data)) {
            console.log("Is array. Length:", data.length);
            if (data.length > 0) {
                console.log("First element type:", typeof data[0]);
                if (Array.isArray(data[0])) {
                    console.log("First element length:", data[0].length);
                }
            }
        }
    } catch (e) {
        console.error("Error:", e);
    }
}
test();
