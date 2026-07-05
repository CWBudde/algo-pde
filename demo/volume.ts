/// <reference lib="dom" />

// WebGL2 volumetric renderer for the 3D acoustic box.
//
// The 3D solve already hands back the whole volume as stacked RGBA planes
// (blue→white→red diverging map of the pressure field, normalized once over the
// box). That colour map is invertible, so this renderer needs no extra data: a
// voxel's amplitude is `1 − min(r,g,b)` (white = node = 0, saturated =
// antinode = 1) and its sign is the red/blue side.
//
// We transcode that to a single signed scalar per voxel (−1 = full blue, 0 =
// node, +1 = full red) and upload it as a one-channel `R8` `TEXTURE_3D`, then
// re-apply the diverging colour map in the shader. The transcode matters: with
// hardware `LINEAR` filtering on the *encoded* RGBA, a sample between a red (+)
// and a blue (−) lobe blends to dark/purple, which `1 − min(r,g,b)` would read
// as opaque — so nodal zero-crossings would glow instead of vanish. Filtering a
// signed scalar instead passes cleanly through zero (white, transparent) at a
// node, keeping LINEAR's smoothness without the artefact.
//
// The volume is ray-marched: colour from the re-applied map, opacity from the
// amplitude. Antinodes glow, nodes stay transparent, and the whole standing-
// wave shape reads as one translucent cloud you can orbit.
//
// Compositing is front-to-back with premultiplied alpha, so the WebGL canvas
// composites correctly over the black page. A faint wireframe box is drawn first
// (behind the cloud) for spatial reference.

// ---- Small column-major mat4 / vec3 helpers (no external deps) -------------

type Vec3 = [number, number, number];
type Mat4 = Float32Array;

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(a: Vec3): Vec3 {
  const len = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / len, a[1] / len, a[2] / len];
}

// Right-handed lookAt, column-major (gluLookAt).
function lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const z = normalize(sub(eye, center)); // camera looks down -z toward center
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  // prettier-ignore
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

// Perspective projection, column-major.
function perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

// Column-major multiply: returns a·b.
function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

// Camera basis + position for a given orbit, shared by the ray-march (basis
// vectors) and the wireframe (view matrix), so the two passes stay aligned.
function orbitCamera(azimuthDeg: number, elevationDeg: number, distance: number) {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const eye: Vec3 = [distance * Math.cos(el) * Math.sin(az), distance * Math.sin(el), distance * Math.cos(el) * Math.cos(az)];
  const center: Vec3 = [0, 0, 0];
  const worldUp: Vec3 = [0, 1, 0];
  const z = normalize(sub(eye, center)); // points from center toward eye
  const x = normalize(cross(worldUp, z)); // screen right
  const y = cross(z, x); // screen up
  const forward: Vec3 = [-z[0], -z[1], -z[2]]; // toward center
  return { eye, center, worldUp, right: x, up: y, forward };
}

// ---- Shaders ----------------------------------------------------------------

const RAYMARCH_VS = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const RAYMARCH_FS = `#version 300 es
precision highp float;
precision highp sampler3D;

out vec4 fragColor;

uniform vec2 uResolution;
uniform vec3 uCamPos;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uForward;
uniform float uTanHalfFov;
uniform float uAspect;
uniform vec3 uBoxHalf;   // half-extents of the box in world units
uniform sampler3D uVolume;
uniform float uDensity;  // user opacity multiplier
uniform float uGamma;    // amplitude gamma (lifts faint lobes)

const int STEPS = 192;
const float ABSORB = 9.0; // base absorption so density ~1 reads well

void main() {
  vec2 ndc = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
  vec3 dir = normalize(
    uForward
    + (ndc.x * uAspect * uTanHalfFov) * uRight
    + (ndc.y * uTanHalfFov) * uUp);
  vec3 ro = uCamPos;

  // Slab intersection with the axis-aligned box centred at the origin.
  vec3 invD = 1.0 / dir;
  vec3 t0 = (-uBoxHalf - ro) * invD;
  vec3 t1 = ( uBoxHalf - ro) * invD;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tNear = max(max(tmin.x, tmin.y), tmin.z);
  float tFar  = min(min(tmax.x, tmax.y), tmax.z);
  if (tFar <= max(tNear, 0.0)) { fragColor = vec4(0.0); return; }
  tNear = max(tNear, 0.0);

  float dt = (tFar - tNear) / float(STEPS);
  vec4 acc = vec4(0.0); // premultiplied colour + accumulated alpha
  for (int i = 0; i < STEPS; i++) {
    float t = tNear + (float(i) + 0.5) * dt;
    if (t > tFar) break;
    vec3 p = ro + t * dir;
    vec3 uvw = (p + uBoxHalf) / (2.0 * uBoxHalf);
    uvw.y = 1.0 - uvw.y; // match the top-down orientation of the slice viewer
    // Signed scalar in [0,1]: 0.5 is a node, 0/1 are the blue/red extremes.
    float sv = texture(uVolume, uvw).r * 2.0 - 1.0; // → [-1, 1]
    float amp = abs(sv);
    // Re-apply the blue↔white↔red diverging map after interpolation.
    vec3 col = sv >= 0.0
      ? mix(vec3(1.0), vec3(1.0, 0.0, 0.0), amp)
      : mix(vec3(1.0), vec3(0.0, 0.0, 1.0), amp);
    float a = clamp(pow(amp, uGamma) * uDensity * dt * ABSORB, 0.0, 1.0);
    acc.rgb += (1.0 - acc.a) * a * col;
    acc.a   += (1.0 - acc.a) * a;
    if (acc.a > 0.995) break;
  }
  fragColor = acc; // premultiplied
}`;

const WIRE_VS = `#version 300 es
in vec3 aPos;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * vec4(aPos, 1.0); }`;

const WIRE_FS = `#version 300 es
precision highp float;
out vec4 fragColor;
uniform vec4 uColor; // premultiplied
void main() { fragColor = uColor; }`;

// ---- Renderer ---------------------------------------------------------------

export interface RenderParams {
  azimuthDeg: number;
  elevationDeg: number;
  distance: number;
  density: number;
  gamma: number;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile failed: ${log}`);
  }
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  const v = compile(gl, gl.VERTEX_SHADER, vs);
  const f = compile(gl, gl.FRAGMENT_SHADER, fs);
  gl.attachShader(p, v);
  gl.attachShader(p, f);
  gl.linkProgram(p);
  // Once linked, the program keeps its own copy — detach and delete the shader
  // objects so they don't leak if the renderer is re-created (reload / HMR).
  gl.detachShader(p, v);
  gl.detachShader(p, f);
  gl.deleteShader(v);
  gl.deleteShader(f);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(`program link failed: ${log}`);
  }
  return p;
}

export class VolumeRenderer {
  private gl: WebGL2RenderingContext;
  private rayProg: WebGLProgram;
  private wireProg: WebGLProgram;
  private quadVao: WebGLVertexArrayObject;
  private wireVao: WebGLVertexArrayObject;
  private wireBuf: WebGLBuffer;
  private wireCount: number;
  private tex: WebGLTexture;
  private hasVolume = false;
  private boxHalf: Vec3 = [0.5, 0.5, 0.5];
  private aspect = 1;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: true,
      antialias: true,
      depth: false,
    });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;

    this.rayProg = link(gl, RAYMARCH_VS, RAYMARCH_FS);
    this.wireProg = link(gl, WIRE_VS, WIRE_FS);

    // Full-screen triangle for the ray-march pass.
    this.quadVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.quadVao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const aPosRay = gl.getAttribLocation(this.rayProg, 'aPos');
    gl.enableVertexAttribArray(aPosRay);
    gl.vertexAttribPointer(aPosRay, 2, gl.FLOAT, false, 0, 0);

    // Box wireframe (12 edges). The VAO and its VBO are created once here;
    // setVolume only refills the buffer's contents (bufferData) when the box
    // extents change, so no GL buffer is allocated per solve.
    this.wireVao = gl.createVertexArray()!;
    this.wireBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.wireVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.wireBuf);
    const aPosWire = gl.getAttribLocation(this.wireProg, 'aPos');
    gl.enableVertexAttribArray(aPosWire);
    gl.vertexAttribPointer(aPosWire, 3, gl.FLOAT, false, 0, 0);
    this.wireCount = 0;
    this.tex = gl.createTexture()!;

    gl.bindVertexArray(null);
  }

  // Upload a fresh RGBA volume (nx×ny×nz, x contiguous, z slowest — the layout
  // the 3D solve returns) and set the physical box extents so proportions are
  // correct. Extents are normalized so the longest axis spans 1 world unit.
  setVolume(data: Uint8ClampedArray, nx: number, ny: number, nz: number, ex: number, ey: number, ez: number) {
    const gl = this.gl;

    // Transcode the encoded RGBA to a signed scalar per voxel so the GPU
    // interpolates the field itself (through zero at a node), not its colour.
    //   amplitude = 1 − min(r,g,b)/255   sign = red side (+) vs blue side (−)
    // packed into a byte as 128 + 127·signed (128 = node).
    const voxels = nx * ny * nz;
    const scalar = new Uint8Array(voxels);
    for (let i = 0; i < voxels; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const amp = 1 - Math.min(r, g, b) / 255; // |field|, gamma-encoded, in [0,1]
      const signed = r >= b ? amp : -amp; // red side positive, blue side negative
      scalar[i] = Math.max(0, Math.min(255, Math.round(128 + 127 * signed)));
    }

    gl.bindTexture(gl.TEXTURE_3D, this.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, scalar);

    const maxExtent = Math.max(ex, ey, ez) || 1;
    this.boxHalf = [(0.5 * ex) / maxExtent, (0.5 * ey) / maxExtent, (0.5 * ez) / maxExtent];
    this.buildWireframe();
    this.hasVolume = true;
  }

  private buildWireframe() {
    const gl = this.gl;
    const [hx, hy, hz] = this.boxHalf;
    // 8 corners.
    const c: Vec3[] = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0], // back face
      [4, 5], [5, 6], [6, 7], [7, 4], // front face
      [0, 4], [1, 5], [2, 6], [3, 7], // connectors
    ];
    const verts: number[] = [];
    for (const [a, b] of edges) verts.push(...c[a], ...c[b]);
    this.wireCount = edges.length * 2;

    // Refill the persistent VBO (created once in the constructor); the VAO's
    // attribute already points at it, so no new buffer or attrib setup here.
    gl.bindBuffer(gl.ARRAY_BUFFER, this.wireBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
  }

  // Match the drawing-buffer viewport after the caller resizes the canvas.
  resize(width: number, height: number) {
    this.gl.viewport(0, 0, width, height);
    this.aspect = height > 0 ? width / height : 1;
  }

  render(params: RenderParams) {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.hasVolume) return;

    // Both passes use premultiplied "over" compositing.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const cam = orbitCamera(params.azimuthDeg, params.elevationDeg, params.distance);
    const fovy = (45 * Math.PI) / 180;

    // Wireframe first, so the cloud composites over it.
    const view = lookAt(cam.eye, cam.center, cam.worldUp);
    const proj = perspective(fovy, this.aspect, 0.01, 100);
    const viewProj = multiply(proj, view);
    gl.useProgram(this.wireProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.wireProg, 'uViewProj'), false, viewProj);
    // Faint white, premultiplied (rgb already ×alpha).
    gl.uniform4f(gl.getUniformLocation(this.wireProg, 'uColor'), 0.09, 0.09, 0.09, 0.35);
    gl.bindVertexArray(this.wireVao);
    gl.drawArrays(gl.LINES, 0, this.wireCount);

    // Volume cloud.
    gl.useProgram(this.rayProg);
    const u = (n: string) => gl.getUniformLocation(this.rayProg, n);
    gl.uniform2f(u('uResolution'), gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.uniform3f(u('uCamPos'), cam.eye[0], cam.eye[1], cam.eye[2]);
    gl.uniform3f(u('uRight'), cam.right[0], cam.right[1], cam.right[2]);
    gl.uniform3f(u('uUp'), cam.up[0], cam.up[1], cam.up[2]);
    gl.uniform3f(u('uForward'), cam.forward[0], cam.forward[1], cam.forward[2]);
    gl.uniform1f(u('uTanHalfFov'), Math.tan(fovy / 2));
    gl.uniform1f(u('uAspect'), this.aspect);
    gl.uniform3f(u('uBoxHalf'), this.boxHalf[0], this.boxHalf[1], this.boxHalf[2]);
    gl.uniform1f(u('uDensity'), params.density);
    gl.uniform1f(u('uGamma'), params.gamma);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.tex);
    gl.uniform1i(u('uVolume'), 0);
    gl.bindVertexArray(this.quadVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindVertexArray(null);
  }
}
