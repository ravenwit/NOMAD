const fragmentShader = `
  uniform float intensity;
  uniform sampler2D dataTexture;
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewPosition;

  // Premium aesthetic palette
  vec3 colorZero = vec3(0.04, 0.05, 0.08); // Obsidian dark base
  vec3 colorPos  = vec3(1.0, 0.45, 0.35);  // Rose gold for positive peaks
  vec3 colorNeg  = vec3(0.1, 0.95, 0.7);   // Bioluminescent mint for negative troughs

  void main() {
    float wave = texture2D(dataTexture, vUv).r;
    float gain = 30.0;
    
    vec3 materialColor;
    if (wave > 0.0) {
        float t = clamp(wave * gain, 0.0, 1.0);
        materialColor = mix(colorZero, colorPos, smoothstep(0.0, 1.0, t));
    } else {
        float t = clamp(-wave * gain, 0.0, 1.0);
        materialColor = mix(colorZero, colorNeg, smoothstep(0.0, 1.0, t));
    }
    
    // Procedural Lighting setup
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition);

    vec3 lightDir = normalize(vec3(1.0, 1.5, 1.0)); // Top-right
    vec3 lightColor = vec3(1.0, 0.98, 0.95);
    vec3 ambient = vec3(0.2, 0.25, 0.3); // Soft ambient
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular Highlight
    vec3 halfVector = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVector), 0.0), 64.0);
    vec3 specular = 0.6 * spec * lightColor;
    
    // Rim Light
    float rimDot = 1.0 - max(dot(viewDir, normal), 0.0);
    float rimAmount = smoothstep(0.6, 1.0, rimDot);
    vec3 rimLight = vec3(0.4, 0.5, 0.8) * rimAmount * 0.4;

    vec3 finalColor = materialColor * (ambient + diffuse) + specular + rimLight;
    
    gl_FragColor = vec4(finalColor * intensity, 1.0);
  }
`;

const res = 'uniform float isErrorMode;\nuniform float gain;\n' +
            `
            vec3 heatmap(float t) {
                vec3 c0 = vec3(0.02, 0.0, 0.05);
                vec3 c1 = vec3(0.8, 0.1, 0.2);
                vec3 c2 = vec3(1.0, 0.8, 0.1);
                vec3 c3 = vec3(1.0, 1.0, 1.0);
                
                if (t < 0.33) return mix(c0, c1, t * 3.0);
                if (t < 0.66) return mix(c1, c2, (t - 0.33) * 3.0);
                return mix(c2, c3, (t - 0.66) * 3.0);
            }
            ` +
            fragmentShader.replace(
                'float gain = 30.0;',
                '' // Remove hardcoded gain, use the uniform
            ).replace(
                'gl_FragColor = vec4(finalColor * intensity, 1.0);',
                `
                if (isErrorMode > 0.5) {
                    float t = clamp(abs(wave) * gain, 0.0, 1.0);
                    vec3 errColor = heatmap(t);
                    vec3 finalErr = errColor * (ambient + diffuse) + specular + rimLight;
                    gl_FragColor = vec4(finalErr * intensity, 1.0);
                } else {
                    gl_FragColor = vec4(finalColor * intensity, 1.0);
                }
                `
            );

console.log("Replaced float gain:", res.includes('float gain = 30.0;'));
console.log("Replaced gl_FragColor:", res.includes('gl_FragColor = vec4(finalColor * intensity, 1.0);'));
console.log(res);
