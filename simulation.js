'use strict';

const canvas = document.getElementsByTagName('canvas')[0];
const DESKTOP_TEXTURE_SIZE = 200;
const MOBILE_TEXTURE_SIZE = 100;
const FIELD_W = 240;
const FIELD_H = Math.floor(window.screen.availHeight / window.screen.availWidth * FIELD_W);

let texture_size = DESKTOP_TEXTURE_SIZE;
const regex = /Mobi|Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
if (regex.test(navigator.userAgent)) {
    texture_size = MOBILE_TEXTURE_SIZE;
}

let particle_count = texture_size * texture_size;

const config = {
    N_SUBSTEPS: 3,
    DELTA_TIME: 0.004,
    SPLAT_RADIUS: 0.06,
    HEATMAP_INTENSITY: 0.04,
    PARTICLE_OPACITY: 1.0,
    GRAVITY: 2.0,
    REST_DENSITY: 30.0,
    GAS_CONSTANT: 15.0,
    NEAR_PRESSURE_MULTIPLIER: 0.0,
    VISCOSITY: 0.0,
    SHOW_DENSITY: true,
    SHOW_PARTICLES: true
};

const params = { alpha: true, depth: false, stencil: false, antialias: false };
let gl = canvas.getContext('webgl2', params);

// Linear filtering for floats
gl.getExtension('EXT_color_buffer_float');
gl.getExtension('OES_texture_float_linear');

const baseVS = `#version 300 es
    precision highp float;

    layout(location = 0) in vec2 aPosition;
    out vec2 vUv;

    void main() {
        vUv = aPosition * 0.5 + 0.5;
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }`;

const simFS = `#version 300 es
    precision highp float;

    uniform sampler2D uParticle;
    uniform sampler2D uDensity;
    uniform sampler2D uPressure;
    uniform sampler2D uViscosity;
    uniform sampler2D uNearPressure;

    uniform float dT;
    uniform float uRestDensity;
    uniform float uGasConstant;
    uniform float uNearPressureMultiplier;
    uniform float uGravity;
    uniform float uViscosityStrength;
    out vec4 outPos;

    void main() {
        vec2 uv = gl_FragCoord.xy / vec2(${texture_size}.0);
        vec2 pos = texture(uParticle, uv).xy;
        vec2 vel = texture(uParticle, uv).zw;

        vec2 fieldUV = pos * 0.5 + 0.5; // (-1, 1) range to (0, 1)

        // density is 1 from the particle itself
        // the max shouldnt be necessary - just for safety
        float density = texture(uDensity, fieldUV).x;
        float density_term = density - uRestDensity;

        float nearDensity = texture(uDensity, fieldUV).y;

        vec2 grad = texture(uPressure, fieldUV).xy;
        vec2 invDensityGrad = texture(uPressure, fieldUV).zw;
        vec2 pressureForce = uGasConstant * ((density_term * invDensityGrad) + grad);
        // pressureForce = normalize(pressureForce) * max(0.0, length(pressureForce) - (uGasConstant * ((2.0 * density - uRestDensity) / density)));

        vec2 nearGrad = texture(uNearPressure, fieldUV).xy;
        vec2 invDensityNearGrad = texture(uNearPressure, fieldUV).zw;
        vec2 nearPressureForce = uNearPressureMultiplier * ((density_term * invDensityNearGrad) + nearGrad);

        vec4 viscosity = texture(uViscosity, fieldUV);
        vec2 viscosityForce = -uViscosityStrength * (viscosity.xy - (vel * viscosity.z));

        vel *= 0.99; // damping

        vel += pressureForce * dT;
        vel += nearPressureForce * dT;
        vel += viscosityForce * dT;
        vel.y -= uGravity * dT;
        pos += vel * dT;

        // Boundaries
        if (pos.x > 1.0) {
            pos.x = 1.0;
            vel.x *= -0.5;
        } else if (pos.x < -1.0) {
            pos.x = -1.0;
            vel.x *= -0.5;
        }

        if (pos.y > 1.0) {
            pos.y = 1.0;
            vel.y *= -0.5;
        } else if (pos.y < -1.0) {
            pos.y = -1.0;
            vel.y *= -0.5;
        }

        outPos = vec4(pos, vel);
    }`;

const splatVS = `#version 300 es
    precision highp float;

    uniform sampler2D uParticle;
    uniform float uRadius;
    out vec4 vParticleInfo;

    void main() {
        int id = gl_VertexID;
        vec2 uv = vec2(id % ${texture_size}, id / ${texture_size}) / vec2(${texture_size}.0);
        vParticleInfo = texture(uParticle, uv);
        vec2 pos = vParticleInfo.xy;
        gl_Position = vec4(pos, 0.0, 1.0);
        gl_PointSize = uRadius * ${FIELD_H}.0; // TODO
    }`;

const densityFS = `#version 300 es
    precision highp float;

    out vec4 outColor;

    void main() {
        vec2 vecFromCentre = gl_PointCoord - 0.5;
        vecFromCentre.y *= -1.0; // flip y for correct orientation
        float distance = length(vecFromCentre) * 2.0; // normalised so that all distances are between 0 and 1
        if (distance > 1.0) discard;
        if (distance < 0.05) discard;
        float density = pow(1.0 - distance, 2.0); // spikyPow2
        float nearDensity = pow(1.0 - distance, 3.0); // spikyPow3
        outColor = vec4(density, nearDensity, 0.0, 1.0);
    }`;

const pressureFS = `#version 300 es
    precision highp float;

    uniform sampler2D uDensity;
    out vec4 outColor;
    in vec4 vParticleInfo;

    void main() {
        vec2 vecFromCentre = gl_PointCoord - 0.5;
        vecFromCentre.y *= -1.0; // flip y for correct orientation
        float distance = length(vecFromCentre) * 2.0;
        if (distance > 1.0) discard;
        if (distance < 0.05) discard;

        float strength = 1.0 - distance;
        float lenSq = dot(vecFromCentre, vecFromCentre);
        // having 1e-3 be fairly large seems to stop it getting to energetic
        vec2 grad = vecFromCentre * inversesqrt(max(lenSq, 1e-8)) * strength; // safe normalize

        vec2 fieldUV = vParticleInfo.xy * 0.5 + 0.5; // (-1, 1) world pos range to (0, 1) for UV
        float density = texture(uDensity, fieldUV).x;
        float invDensity = 1.0 / (density + 1e-6);
        vec2 invDensityGrad = grad * invDensity;

        outColor = vec4(grad.x, grad.y, invDensityGrad.x, invDensityGrad.y);
    }`;

const nearPressureFS = `#version 300 es
    precision highp float;

    uniform sampler2D uDensity;
    out vec4 outColor;
    in vec4 vParticleInfo;

    void main() {
        vec2 vecFromCentre = gl_PointCoord - 0.5;
        vecFromCentre.y *= -1.0; // flip y for correct orientation
        float distance = length(vecFromCentre) * 2.0;
        if (distance > 1.0) discard;
        if (distance < 0.05) discard;

        float strength = pow(1.0 - distance, 2.0);
        float lenSq = dot(vecFromCentre, vecFromCentre);
        vec2 grad = vecFromCentre * inversesqrt(max(lenSq, 1e-3)) * strength; // safe normalize

        vec2 fieldUV = vParticleInfo.xy * 0.5 + 0.5; // (-1, 1) range to (0, 1)
        float nearDensity = texture(uDensity, fieldUV).y;
        float invNearDensity = 1.0 / (nearDensity + 1e-6);
        vec2 invNearDensityGrad = grad * invNearDensity;

        outColor = vec4(grad.x, grad.y, invNearDensityGrad.x, invNearDensityGrad.y);
    }`;

const viscosityFS = `#version 300 es
    precision highp float;

    uniform sampler2D uParticle;
    uniform sampler2D uDensity;
    out vec4 outColor;
    in vec4 vParticleInfo;

    void main() {
        vec2 vecFromCentre = gl_PointCoord - 0.5;
        vecFromCentre.y *= -1.0; // flip y for correct orientation
        float distance = length(vecFromCentre) * 2.0;
        if (distance > 1.0) discard;

        float v = 1.0 - (distance * distance);
        float strength = distance * distance * distance;

        vec2 fieldUV = vParticleInfo.xy * 0.5 + 0.5; // (-1, 1) range to (0, 1)
        float density = texture(uDensity, fieldUV).x;
        float invDensity = 1.0 / (density + 1e-6);

        vec2 vel = vParticleInfo.zw;
        float magnitude = invDensity * strength;
        vec2 v = magnitude * vel;
        outColor = vec4(v.x, v.y, magnitude, 0.0);
    }`;

const displayFS = `#version 300 es
    precision highp float;

    uniform sampler2D uDensity;
    uniform float uIntensity;
    in vec2 vUv;
    out vec4 outColor;

    void main() {
        float d = texture(uDensity, vUv).r * uIntensity;
        // Heatmap Gradient: Dark Blue -> Cyan -> White
        vec3 color = mix(vec3(0.01, 0.01, 0.04), vec3(0.1, 0.5, 1.0), d);
        color = mix(color, vec3(1.0, 1.0, 1.0), smoothstep(0.7, 1.3, d));
        outColor = vec4(color, 1.0);
    }`;

const particleVS = `#version 300 es
    precision highp float;

    uniform sampler2D uParticle;

    void main() {
        int id = gl_VertexID;
        vec2 uv = vec2(id % ${texture_size}, id / ${texture_size}) / vec2(${texture_size}.0);
        vec2 pos = texture(uParticle, uv).xy;
        gl_Position = vec4(pos, 0.0, 1.0);
        gl_PointSize = 1.5;
    }`;

const particleFS = `#version 300 es
    precision highp float;

    uniform float uOpacity;
    out vec4 outColor;

    void main() {
        outColor = vec4(1.0, 1.0, 1.0, uOpacity);
    }`;

class Program {
    constructor(vsSource, fsSource) {
        const vs = this.compile(gl.VERTEX_SHADER, vsSource);
        const fs = this.compile(gl.FRAGMENT_SHADER, fsSource);
        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);
        this.uniforms = {};
        let count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < count; i++) {
            let name = gl.getActiveUniform(this.program, i).name;
            this.uniforms[name] = gl.getUniformLocation(this.program, name);
        }
    }
    compile(type, source) {
        const s = gl.createShader(type);
        gl.shaderSource(s, source);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
        return s;
    }
    bind() { gl.useProgram(this.program); }
}

function createFBO(w, h, filter) {
    gl.activeTexture(gl.TEXTURE0);
    let tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo };
}

const simProgram = new Program(baseVS, simFS);
const densityProgram = new Program(splatVS, densityFS);
const pressureProgram = new Program(splatVS, pressureFS);
const nearPressureProgram = new Program(splatVS, nearPressureFS);
const viscosityProgram = new Program(splatVS, viscosityFS);
const displayProgram = new Program(baseVS, displayFS);
const particleProgram = new Program(particleVS, particleFS);

// two particle textures for ping-pong
let particleFBO = [
    createFBO(texture_size, texture_size, gl.NEAREST),
    createFBO(texture_size, texture_size, gl.NEAREST)
];

// render targets
let densityFBO = createFBO(FIELD_W, FIELD_H, gl.LINEAR);
let pressureFBO = createFBO(FIELD_W, FIELD_H, gl.LINEAR);
let nearPressureFBO = createFBO(FIELD_W, FIELD_H, gl.LINEAR);
let viscosityFBO = createFBO(FIELD_W, FIELD_H, gl.LINEAR);

function randomRange(min, max) {
    return Math.random() * (max - min) + min;
}

// initialise particles
const initialData = new Float32Array(particle_count * 4);
for (let i = 0; i < particle_count; i++) {
    // position
    initialData[i * 4] = randomRange(-0.9, 0.4); // x
    initialData[i * 4 + 1] = randomRange(0.4, 0.9); // y

    // velocity
    initialData[i * 4 + 2] = 0.0;
    initialData[i * 4 + 3] = 0.0;
}
gl.bindTexture(gl.TEXTURE_2D, particleFBO[0].tex);
gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, texture_size, texture_size, gl.RGBA, gl.FLOAT, initialData);

// Fullscreen quad
const quadVAO = gl.createVertexArray();
gl.bindVertexArray(quadVAO);
const quadVBO = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

// Empty VAO for gl_VertexID based calls
const emptyVAO = gl.createVertexArray();

const gui = new dat.GUI();
gui.add(config, 'SHOW_DENSITY').name('Show Heatmap');
gui.add(config, 'SHOW_PARTICLES').name('Show Particles');
gui.add(config, 'DELTA_TIME', 0, 0.2).name('Delta Time');
gui.add(config, 'SPLAT_RADIUS', 0.01, 0.1).name('Density Radius');
gui.add(config, 'HEATMAP_INTENSITY', 0.01, 0.1).name('Heatmap Brightness');
gui.add(config, 'PARTICLE_OPACITY', 0, 1.0).name('Particle Opacity');
gui.add(config, 'REST_DENSITY', 0.0, 30.5).name('Rest Density');
gui.add(config, 'GAS_CONSTANT', 0.0, 50.0).name('Gas Constant');
gui.add(config, 'NEAR_PRESSURE_MULTIPLIER', 0.0, 20.0).name('Near Pressure Multiplier');
gui.add(config, 'GRAVITY', 0.0, 10.0).name('Gravity');
gui.add(config, 'VISCOSITY', 0.0, 100.0).name('Viscosity');

// Initialise stats for FPS
const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

// FPS display - top left corner
stats.dom.style.position = 'absolute';
stats.dom.style.top = '0px';
stats.dom.style.left = '0px';

function substep() {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, particleFBO[0].tex);

    //// Density pass ////
    gl.viewport(0, 0, FIELD_W, FIELD_H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, densityFBO.fbo); // output
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE); // Additive blending for density accumulation
    densityProgram.bind();
    gl.uniform1i(densityProgram.uniforms.uParticle, 0);
    gl.uniform1f(densityProgram.uniforms.uRadius, config.SPLAT_RADIUS);
    gl.bindVertexArray(emptyVAO);
    gl.drawArrays(gl.POINTS, 0, particle_count);

    //// Pressure pass ////
    gl.bindFramebuffer(gl.FRAMEBUFFER, pressureFBO.fbo); // output
    gl.clear(gl.COLOR_BUFFER_BIT);

    pressureProgram.bind();

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, densityFBO.tex);

    gl.uniform1i(pressureProgram.uniforms.uDensity, 1);
    gl.uniform1f(pressureProgram.uniforms.uRadius, config.SPLAT_RADIUS);
    gl.bindVertexArray(emptyVAO);
    gl.drawArrays(gl.POINTS, 0, particle_count);

    //// Near pressure pass ////
    if (config.NEAR_PRESSURE_MULTIPLIER > 0.0) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, nearPressureFBO.fbo); // output
        gl.clear(gl.COLOR_BUFFER_BIT);

        nearPressureProgram.bind();

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, densityFBO.tex);

        gl.uniform1i(nearPressureProgram.uniforms.uDensity, 1);
        gl.uniform1f(nearPressureProgram.uniforms.uRadius, config.SPLAT_RADIUS);
        gl.bindVertexArray(emptyVAO);
        gl.drawArrays(gl.POINTS, 0, particle_count);
    }

    //// Viscosity pass ////
    if (config.VISCOSITY > 0.0) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, viscosityFBO.fbo); // output
        gl.clear(gl.COLOR_BUFFER_BIT);

        viscosityProgram.bind();
        gl.activeTexture(gl.TEXTURE1); // ?
        gl.bindTexture(gl.TEXTURE_2D, densityFBO.tex); // ?
        gl.uniform1i(viscosityProgram.uniforms.uParticle, 0);
        gl.uniform1f(viscosityProgram.uniforms.uRadius, config.SPLAT_RADIUS);
        gl.bindVertexArray(emptyVAO);
        gl.drawArrays(gl.POINTS, 0, particle_count);
    }

    gl.disable(gl.BLEND);

    //// Simulation pass ////
    gl.viewport(0, 0, texture_size, texture_size);
    gl.bindFramebuffer(gl.FRAMEBUFFER, particleFBO[1].fbo); // output
    simProgram.bind();

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, pressureFBO.tex);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, viscosityFBO.tex);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, nearPressureFBO.tex);

    gl.uniform1i(simProgram.uniforms.uParticle, 0);
    gl.uniform1i(simProgram.uniforms.uDensity, 1);
    gl.uniform1i(simProgram.uniforms.uPressure, 2);
    gl.uniform1i(simProgram.uniforms.uViscosity, 3);
    gl.uniform1i(simProgram.uniforms.uNearPressure, 4);
    gl.uniform1f(simProgram.uniforms.dT, config.DELTA_TIME);
    gl.uniform1f(simProgram.uniforms.uRestDensity, config.REST_DENSITY);
    gl.uniform1f(simProgram.uniforms.uGasConstant, config.GAS_CONSTANT);
    gl.uniform1f(simProgram.uniforms.uNearPressureMultiplier, config.NEAR_PRESSURE_MULTIPLIER);
    gl.uniform1f(simProgram.uniforms.uGravity, config.GRAVITY);
    gl.uniform1f(simProgram.uniforms.uViscosityStrength, config.VISCOSITY);

    gl.bindVertexArray(quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    particleFBO.reverse(); // for ping-pong
}

// Render loop
function update(time) {
    stats.begin();

    if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    for (let i=0; i < config.N_SUBSTEPS; i++) {
        substep();
    }

    //// Render pass ////
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Draw density field (if enabled)
    if (config.SHOW_DENSITY) {
        displayProgram.bind();
        gl.uniform1i(displayProgram.uniforms.uDensity, 1);
        gl.uniform1f(displayProgram.uniforms.uIntensity, config.HEATMAP_INTENSITY);
        gl.bindVertexArray(quadVAO);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // Draw particles (if enabled)
    if (config.SHOW_PARTICLES) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        particleProgram.bind();
        gl.uniform1i(particleProgram.uniforms.uParticle, 0);
        gl.uniform1f(particleProgram.uniforms.uOpacity, config.PARTICLE_OPACITY);
        gl.bindVertexArray(emptyVAO);
        gl.drawArrays(gl.POINTS, 0, particle_count);
        gl.disable(gl.BLEND);
    }

    stats.end();
    requestAnimationFrame(update);
}

requestAnimationFrame(update);