"use strict";

// Get the canvas and its WebGL context
const canvas = document.getElementById("canvas");

// --- NEW: Configuration Object with more descriptive names ---
// We will link this to our dat.gui panel
const config = {
    TEXTURE_DOWNSAMPLE: 1,
    DENSITY_DISSIPATION: 0.98,
    VELOCITY_DISSIPATION: 0.99,
    PRESSURE_DISSIPATION: 0.8,
    PRESSURE_ITERATIONS: 25,
    CURL: 30, // Vorticity (swirliness)
    SPLAT_RADIUS: 0.35 // Splash radius, as a percentage of the shortest screen dimension
};

// --- WebGL setup - largely unchanged but cleaned up ---
const { gl, ext, support_linear_float } = getWebGLContext(canvas);

function getWebGLContext(canvas) {
    const params = { alpha: false, depth: false, stencil: false, antialias: false };
    let gl = canvas.getContext("webgl2", params);
    const isWebGL2 = !!gl;
    if (!isWebGL2) {
        gl = canvas.getContext("webgl", params) || canvas.getContext("experimental-webgl", params);
    }

    const halfFloat = gl.getExtension("OES_texture_half_float");
    let support_linear_float = gl.getExtension("OES_texture_half_float_linear");
    if (isWebGL2) {
        gl.getExtension("EXT_color_buffer_float");
        support_linear_float = gl.getExtension("OES_texture_float_linear");
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    const internalFormat = isWebGL2 ? gl.RGBA16F : gl.RGBA;
    const internalFormatRG = isWebGL2 ? gl.RG16F : gl.RGBA;
    const formatRG = isWebGL2 ? gl.RG : gl.RGBA;
    const texType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;

    return { gl, ext: { internalFormat, internalFormatRG, formatRG, texType }, support_linear_float };
}

// --- Shader Compilation - unchanged ---
// (The long shader source code strings are omitted here for brevity, but they are identical to your original file)
const baseVertexShader = compileShader(gl.VERTEX_SHADER, "precision highp float; precision mediump sampler2D; attribute vec2 aPosition; varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB; uniform vec2 texelSize; void main () {     vUv = aPosition * 0.5 + 0.5;     vL = vUv - vec2(texelSize.x, 0.0);     vR = vUv + vec2(texelSize.x, 0.0);     vT = vUv + vec2(0.0, texelSize.y);     vB = vUv - vec2(0.0, texelSize.y);     gl_Position = vec4(aPosition, 0.0, 1.0); }");
const clearShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; uniform sampler2D uTexture; uniform float value; void main () {     gl_FragColor = value * texture2D(uTexture, vUv); }");
const displayShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; uniform sampler2D uTexture; void main () {     gl_FragColor = texture2D(uTexture, vUv); }");
const splatShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; uniform sampler2D uTarget; uniform float aspectRatio; uniform vec3 color; uniform vec2 point; uniform float radius; void main () {     vec2 p = vUv - point.xy;     p.x *= aspectRatio;     vec3 splat = exp(-dot(p, p) / radius) * color;     vec3 base = texture2D(uTarget, vUv).xyz;     gl_FragColor = vec4(base + splat, 1.0); }");
const advectionShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; uniform sampler2D uVelocity; uniform sampler2D uSource; uniform vec2 texelSize; uniform float dt; uniform float dissipation; void main () {     vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;     gl_FragColor = dissipation * texture2D(uSource, coord); }");
const advectionManualFilteringShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; uniform sampler2D uVelocity; uniform sampler2D uSource; uniform vec2 texelSize; uniform float dt; uniform float dissipation; vec4 bilerp (in sampler2D sam, in vec2 p) {     vec4 st;     st.xy = floor(p - 0.5) + 0.5;     st.zw = st.xy + 1.0;     vec4 uv = st * texelSize.xyxy;     vec4 a = texture2D(sam, uv.xy);     vec4 b = texture2D(sam, uv.zy);     vec4 c = texture2D(sam, uv.xw);     vec4 d = texture2D(sam, uv.zw);     vec2 f = p - st.xy;     return mix(mix(a, b, f.x), mix(c, d, f.x), f.y); } void main () {     vec2 coord = gl_FragCoord.xy - dt * texture2D(uVelocity, vUv).xy;     gl_FragColor = dissipation * bilerp(uSource, coord);     gl_FragColor.a = 1.0; }");
const divergenceShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB; uniform sampler2D uVelocity; vec2 sampleVelocity (in vec2 uv) {     vec2 multiplier = vec2(1.0, 1.0);     if (uv.x < 0.0) { uv.x = 0.0; multiplier.x = -1.0; }     if (uv.x > 1.0) { uv.x = 1.0; multiplier.x = -1.0; }     if (uv.y < 0.0) { uv.y = 0.0; multiplier.y = -1.0; }     if (uv.y > 1.0) { uv.y = 1.0; multiplier.y = -1.0; }     return multiplier * texture2D(uVelocity, uv).xy; } void main () {     float L = sampleVelocity(vL).x;     float R = sampleVelocity(vR).x;     float T = sampleVelocity(vT).y;     float B = sampleVelocity(vB).y;     float div = 0.5 * (R - L + T - B);     gl_FragColor = vec4(div, 0.0, 0.0, 1.0); }");
const curlShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB; uniform sampler2D uVelocity; void main () {     float L = texture2D(uVelocity, vL).y;     float R = texture2D(uVelocity, vR).y;     float T = texture2D(uVelocity, vT).x;     float B = texture2D(uVelocity, vB).x;     float vorticity = R - L - T + B;     gl_FragColor = vec4(vorticity, 0.0, 0.0, 1.0); }");
const vorticityShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB; uniform sampler2D uVelocity; uniform sampler2D uCurl; uniform float curl; uniform float dt; void main () {     float L = texture2D(uCurl, vL).y;     float R = texture2D(uCurl, vR).y;     float T = texture2D(uCurl, vT).x;     float B = texture2D(uCurl, vB).x;     float C = texture2D(uCurl, vUv).x;     vec2 force = vec2(abs(T) - abs(B), abs(R) - abs(L));     force *= 1.0 / length(force + 0.00001) * curl * C;     vec2 vel = texture2D(uVelocity, vUv).xy;     gl_FragColor = vec4(vel + force * dt, 0.0, 1.0); }");
const pressureShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB; uniform sampler2D uPressure; uniform sampler2D uDivergence; vec2 boundary (in vec2 uv) {     uv = min(max(uv, 0.0), 1.0);     return uv; } void main () {     float L = texture2D(uPressure, boundary(vL)).x;     float R = texture2D(uPressure, boundary(vR)).x;     float T = texture2D(uPressure, boundary(vT)).x;     float B = texture2D(uPressure, boundary(vB)).x;     float C = texture2D(uPressure, vUv).x;     float divergence = texture2D(uDivergence, vUv).x;     float pressure = (L + R + B + T - divergence) * 0.25;     gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0); }");
const gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, "precision highp float; precision mediump sampler2D; varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB; uniform sampler2D uPressure; uniform sampler2D uVelocity; vec2 boundary (in vec2 uv) {     uv = min(max(uv, 0.0), 1.0);     return uv; } void main () {     float L = texture2D(uPressure, boundary(vL)).x;     float R = texture2D(uPressure, boundary(vR)).x;     float T = texture2D(uPressure, boundary(vT)).x;     float B = texture2D(uPressure, boundary(vB)).x;     vec2 velocity = texture2D(uVelocity, vUv).xy;     velocity.xy -= vec2(R - L, T - B);     gl_FragColor = vec4(velocity, 0.0, 1.0); }");

function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(shader);
    return shader;
};

// --- GL Program Class, FBO creation, etc. - largely unchanged ---
class GLProgram {
    constructor(vertexShader, fragmentShader) {
        this.uniforms = {};
        this.program = gl.createProgram();
        gl.attachShader(this.program, vertexShader);
        gl.attachShader(this.program, fragmentShader);
        gl.linkProgram(this.program);
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw gl.getProgramInfoLog(this.program);
        const uniformCount = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < uniformCount; i++) {
            const uniformName = gl.getActiveUniform(this.program, i).name;
            this.uniforms[uniformName] = gl.getUniformLocation(this.program, uniformName);
        }
    }
    bind() {
        gl.useProgram(this.program);
    }
}

// --- Pointer & Color tracking ---
const pointers = [new pointerPrototype()];
let hue=0;

function pointerPrototype() {
    this.id = -1;
    this.x = 0;
    this.y = 0;
    this.dx = 0;
    this.dy = 0;
    this.down = false;
    this.moved = false;
    this.color = [30, 0, 300];
}

// --- Framebuffer objects and GL programs ---
let textureWidth, textureHeight, density, velocity, divergence, curl, pressure;
initFramebuffers();

const clearProgram = new GLProgram(baseVertexShader, clearShader);
const displayProgram = new GLProgram(baseVertexShader, displayShader);
const splatProgram = new GLProgram(baseVertexShader, splatShader);
const advectionProgram = new GLProgram(baseVertexShader, support_linear_float ? advectionShader : advectionManualFilteringShader);
const divergenceProgram = new GLProgram(baseVertexShader, divergenceShader);
const curlProgram = new GLProgram(baseVertexShader, curlShader);
const vorticityProgram = new GLProgram(baseVertexShader, vorticityShader);
const pressureProgram = new GLProgram(baseVertexShader, pressureShader);
const gradienSubtractProgram = new GLProgram(baseVertexShader, gradientSubtractShader);


function initFramebuffers() {
    textureWidth = gl.drawingBufferWidth >> config.TEXTURE_DOWNSAMPLE;
    textureHeight = gl.drawingBufferHeight >> config.TEXTURE_DOWNSAMPLE;

    const iFormat = ext.internalFormat;
    const iFormatRG = ext.internalFormatRG;
    const formatRG = ext.formatRG;
    const texType = ext.texType;
    const filtering = support_linear_float ? gl.LINEAR : gl.NEAREST;

    density = createDoubleFBO(0, textureWidth, textureHeight, iFormat, gl.RGBA, texType, filtering);
    velocity = createDoubleFBO(2, textureWidth, textureHeight, iFormatRG, formatRG, texType, filtering);
    divergence = createFBO(4, textureWidth, textureHeight, iFormatRG, formatRG, texType, gl.NEAREST);
    curl = createFBO(5, textureWidth, textureHeight, iFormatRG, formatRG, texType, gl.NEAREST);
    pressure = createDoubleFBO(6, textureWidth, textureHeight, iFormatRG, formatRG, texType, gl.NEAREST);
}

function createFBO(texId, w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0 + texId);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return [texture, fbo, texId];
}

function createDoubleFBO(texId, w, h, internalFormat, format, type, param) {
    let fbo1 = createFBO(texId, w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(texId + 1, w, h, internalFormat, format, type, param);
    return {
        get first() { return fbo1; },
        get second() { return fbo2; },
        swap() {
            let temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;
        }
    };
}

const blit = (() => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    return (destination) => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    };
})();


// --- NEW: Real-time Control Panel using dat.gui ---
const gui = new dat.GUI({ width: 300 });
gui.add(config, 'DENSITY_DISSIPATION', 0.9, 1.0).name('Density Decay');
gui.add(config, 'VELOCITY_DISSIPATION', 0.9, 1.0).name('Velocity Decay');
gui.add(config, 'PRESSURE_DISSIPATION', 0.0, 1.0).name('Pressure Decay');
gui.add(config, 'PRESSURE_ITERATIONS', 1, 60).step(1).name('Pressure Iterations');
gui.add(config, 'CURL', 0, 50).name('Vorticity (Curl)');
gui.add(config, 'SPLAT_RADIUS', 0.01, 1.0).name('Splat Radius');


let lastTime = Date.now();
update(); // Start the animation loop

function update() {
    const dt = Math.min((Date.now() - lastTime) / 1000, 0.0166);
    lastTime = Date.now();

    resizeCanvas();
    gl.viewport(0, 0, textureWidth, textureHeight);

    // Advection step
    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.first[2]);
    gl.uniform1i(advectionProgram.uniforms.uSource, velocity.first[2]);
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, config.VELOCITY_DISSIPATION);
    blit(velocity.second[1]);
    velocity.swap();

    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.first[2]);
    gl.uniform1i(advectionProgram.uniforms.uSource, density.first[2]);
    gl.uniform1f(advectionProgram.uniforms.dissipation, config.DENSITY_DISSIPATION);
    blit(density.second[1]);
    density.swap();

    // Splatting user input
    for (let i = pointers.length - 1; i >= 0; i--) {
        const pointer = pointers[i];
        if (pointer.moved) {
            splat(pointer.x, pointer.y, pointer.dx, pointer.dy, pointer.color);
            pointer.moved = false;
        }
    }

    // Curl and Vorticity
    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.first[2]);
    blit(curl[1]);

    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.first[2]);
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl[2]);
    gl.uniform1f(vorticityProgram.uniforms.curl, config.CURL);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.second[1]);
    velocity.swap();

    // Divergence and Pressure solve
    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.first[2]);
    blit(divergence[1]);

    clearProgram.bind();
    gl.uniform1i(clearProgram.uniforms.uTexture, pressure.first[2]);
    gl.uniform1f(clearProgram.uniforms.value, config.PRESSURE_DISSIPATION);
    blit(pressure.second[1]);
    pressure.swap();

    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence[2]);
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
        gl.uniform1i(pressureProgram.uniforms.uPressure, pressure.first[2]);
        blit(pressure.second[1]);
        pressure.swap();
    }

    gradienSubtractProgram.bind();
    gl.uniform2f(gradienSubtractProgram.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
    gl.uniform1i(gradienSubtractProgram.uniforms.uPressure, pressure.first[2]);
    gl.uniform1i(gradienSubtractProgram.uniforms.uVelocity, velocity.first[2]);
    blit(velocity.second[1]);
    velocity.swap();

    // Final render to screen
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    displayProgram.bind();
    gl.uniform1i(displayProgram.uniforms.uTexture, density.first[2]);
    blit(null);

    requestAnimationFrame(update);
}

function splat(x, y, dx, dy, color) {
    const splatRadius = config.SPLAT_RADIUS / 100.0;
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.first[2]);
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x / canvas.width, 1.0 - y / canvas.height);
    gl.uniform3f(splatProgram.uniforms.color, dx, -dy, 1.0);
    gl.uniform1f(splatProgram.uniforms.radius, splatRadius);
    blit(velocity.second[1]);
    velocity.swap();

    gl.uniform1i(splatProgram.uniforms.uTarget, density.first[2]);
    gl.uniform3f(splatProgram.uniforms.color, color[0], color[1], color[2]);
    blit(density.second[1]);
    density.swap();
}

function resizeCanvas() {
    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        initFramebuffers();
    }
}

// --- NEW: Ambient Drifting Effect ---
// Periodically adds a random, gentle splat to make the simulation feel alive.
// setInterval(() => {
//     const color = hslToRgb(Math.random(), 1, 0.5);
//     const x = canvas.width * Math.random();
//     const y = canvas.height * Math.random();
//     const dx = 1000 * (Math.random() - 0.5);
//     const dy = 1000 * (Math.random() - 0.5);
//     splat(x, y, dx, dy, color.map(c => c * 0.1)); // Make ambient splats more subtle
// }, 2000); // every 2 seconds


// this code use for mouse listener based on click
//--- UPDATED: Mouse and Touch Listeners ---
// canvas.addEventListener('mousemove', e => {
//     // NEW: Smoothly cycle hue based on mouse movement
//     hue += 0.01;
//     if(hue > 1) hue = 0;
//     const color = hslToRgb(hue, 1.0, 0.5);

//     pointers[0].moved = pointers[0].down;
//     pointers[0].dx = (e.offsetX - pointers[0].x) * 5.0;
//     pointers[0].dy = (e.offsetY - pointers[0].y) * 5.0;
//     pointers[0].x = e.offsetX;
//     pointers[0].y = e.offsetY;
//     pointers[0].color = color;
// });

// canvas.addEventListener('mousedown', () => {
//     pointers[0].down = true;
// });

// window.addEventListener('mouseup', () => {
//     pointers[0].down = false;
// });


// --- UPDATED: Mouse and Touch Listeners ---
canvas.addEventListener('mousemove', e => {
    // Smoothly cycle hue based on mouse movement
    hue += 0.01;
    if(hue > 1) hue = 0;
    const color = hslToRgb(hue, 1.0, 0.5);

    // This now tells the animation to draw a splat on every mouse move
    pointers[0].moved = true; 
    
    pointers[0].dx = (e.offsetX - pointers[0].x) * 5.0;
    pointers[0].dy = (e.offsetY - pointers[0].y) * 5.0;
    pointers[0].x = e.offsetX;
    pointers[0].y = e.offsetY;
    pointers[0].color = color;
});


// --- Text animation and Audio control ---
const title = document.getElementById("main-title");
const audio = document.getElementById("background-music");

function animateTitle(elementId) {
    const el = document.getElementById(elementId);
    const text = el.innerText;
    el.innerHTML = '';
    for (let i = 0; i < text.length; i++) {
        const letter = text[i] === ' ' ? '&nbsp;' : text[i];
        const span = document.createElement('span');
        span.innerHTML = letter;
        span.style.transitionDelay = `${i * 0.05}s`;
        el.appendChild(span);
    }
    setTimeout(() => {
        el.classList.add('transition-in');
    }, 200);
}

window.onload = function() {
    animateTitle("main-title");
};

title.addEventListener('click', () => {
    if (audio.paused) {
        audio.play();
    } else {
        audio.pause();
    }
});


// --- NEW: HSL to RGB Color Conversion Utility ---
// This is how we get the smooth rainbow effect.
function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l; // achromatic
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }
    return [r * 0.5, g * 0.5, b * 0.5]; // Return a slightly darker color for better visibility
}