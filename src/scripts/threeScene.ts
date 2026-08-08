// Dependency-free WebGL2 port of the hero "agent network" scene.
// Replicates the previous three.js rendering: fibonacci-sphere node cloud,
// near-neighbour connection lines and a translucent icosahedron core, with
// identical projection, Euler-XYZ rotation, blending and point-size math.

type V3 = { x: number; y: number; z: number };

function mat4Multiply(a: number[], b: number[]): number[] {
  const out = new Array<number>(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

function makePerspective(fovDeg: number, aspect: number, near: number, far: number): number[] {
  const f = 1 / Math.tan((fovDeg * Math.PI) / 360);
  const nf = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ];
}

// three.js Euler order 'XYZ': R = Rx(rx) * Ry(ry) * Rz(rz) applied as v' = R*v.
function applyEulerXYZ(src: Float32Array, out: Float32Array, rx: number, ry: number, rz: number): void {
  const c1 = Math.cos(rx), s1 = Math.sin(rx);
  const c2 = Math.cos(ry), s2 = Math.sin(ry);
  const c3 = Math.cos(rz), s3 = Math.sin(rz);
  const m00 = c2 * c3;
  const m01 = -c2 * s3;
  const m02 = s2;
  const m10 = c1 * s3 + c3 * s1 * s2;
  const m11 = c1 * c3 - s1 * s2 * s3;
  const m12 = -c2 * s1;
  const m20 = s1 * s3 - c1 * c3 * s2;
  const m21 = c3 * s1 + c1 * s2 * s3;
  const m22 = c1 * c2;
  const n = src.length;
  for (let i = 0; i < n; i += 3) {
    const x = src[i], y = src[i + 1], z = src[i + 2];
    out[i] = m00 * x + m01 * y + m02 * z;
    out[i + 1] = m10 * x + m11 * y + m12 * z;
    out[i + 2] = m20 * x + m21 * y + m22 * z;
  }
}

function buildIcosahedron(radius: number, detail: number): { positions: Float32Array; indices: Uint16Array } {
  const t = (1 + Math.sqrt(5)) / 2;
  const positions = [
    -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0,
    0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t,
    t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1,
  ];
  let current = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
    1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
    4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];

  const midCache = new Map<number, number>();
  for (let d = 0; d < detail; d++) {
    midCache.clear();
    const next: number[] = [];
    const mid = (a: number, b: number): number => {
      if (a > b) [a, b] = [b, a];
      const key = a * 1000 + b;
      const cached = midCache.get(key);
      if (cached !== undefined) return cached;
      const v = positions.length / 3;
      positions.push(
        (positions[a * 3] + positions[b * 3]) / 2,
        (positions[a * 3 + 1] + positions[b * 3 + 1]) / 2,
        (positions[a * 3 + 2] + positions[b * 3 + 2]) / 2
      );
      midCache.set(key, v);
      return v;
    };
    for (let i = 0; i < current.length; i += 3) {
      const a = current[i], b = current[i + 1], c = current[i + 2];
      const ab = mid(a, b);
      const bc = mid(b, c);
      const ca = mid(c, a);
      next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    current = next;
  }

  for (let i = 0; i < positions.length; i += 3) {
    const inv = radius / Math.hypot(positions[i], positions[i + 1], positions[i + 2]);
    positions[i] *= inv;
    positions[i + 1] *= inv;
    positions[i + 2] *= inv;
  }

  return { positions: new Float32Array(positions), indices: new Uint16Array(current) };
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('shader compile error: ' + log);
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vs: string, fs: string): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error('program link error: ' + log);
  }
  return program;
}

const VS_POINTS = `
  attribute vec3 aPosition;
  attribute vec3 aColor;
  uniform mat4 uProjView;
  uniform float uSize;
  uniform float uScale;
  varying vec3 vColor;
  void main() {
    vec4 mv = uProjView * vec4(aPosition, 1.0);
    gl_Position = mv;
    gl_PointSize = uSize * uScale / max(mv.w, 0.001);
    vColor = aColor;
  }`;

const VS_SOLID = `
  attribute vec3 aPosition;
  uniform mat4 uProjView;
  void main() { gl_Position = uProjView * vec4(aPosition, 1.0); }`;

const FS_COLOR = `
  precision mediump float;
  varying vec3 vColor;
  uniform float uOpacity;
  void main() { gl_FragColor = vec4(vColor, uOpacity); }`;

const FS_SOLID = `
  precision mediump float;
  uniform vec4 uColor;
  void main() { gl_FragColor = uColor; }`;

export function initThreeScene(root: HTMLElement): () => void {
  const canvas = root.querySelector<HTMLCanvasElement>('[data-three-canvas]');
  const fallback = root.querySelector<HTMLElement>('[data-three-fallback]');
  if (!canvas || !fallback) return () => {};

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const hasWebGL = (() => {
    try {
      const c = document.createElement('canvas');
      return !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('webgl2')));
    } catch {
      return false;
    }
  })();

  if (!hasWebGL || reduced) {
    root.classList.add('is-fallback');
    return () => {};
  }

  const gl = (canvas.getContext('webgl2', { alpha: true, antialias: true, premultipliedAlpha: true }) ||
    canvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: true })) as WebGLRenderingContext | null;
  if (!gl) {
    root.classList.add('is-fallback');
    return () => {};
  }

  // ——— geometry (identical to the previous three.js scene) ———
  const COUNT = 220;
  const RADIUS = 2.5;
  const posA = new Float32Array(COUNT * 3);
  const colA = new Float32Array(COUNT * 3);
  const pts: V3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  const cols = [
    [0.94, 0.66, 0.72], // #f0a8b0 rose-pink
    [0.88, 0.5, 0.63], // #e080a0 deeper pink
    [0.63, 0.56, 0.75], // #a090c0 lavender
  ];
  for (let i = 0; i < COUNT; i++) {
    const y = 1 - (i / (COUNT - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = golden * i;
    const x = Math.cos(theta) * r;
    const z = Math.sin(theta) * r;
    posA[i * 3] = x * RADIUS;
    posA[i * 3 + 1] = y * RADIUS;
    posA[i * 3 + 2] = z * RADIUS;
    pts.push({ x: x * RADIUS, y: y * RADIUS, z: z * RADIUS });

    const pick = Math.random();
    const c = pick < 0.12 ? cols[2] : pick < 0.3 ? cols[1] : cols[0];
    colA[i * 3] = c[0];
    colA[i * 3 + 1] = c[1];
    colA[i * 3 + 2] = c[2];
  }

  const lineSrc: number[] = [];
  const THRESH = 1.25;
  for (let i = 0; i < COUNT; i++) {
    for (let j = i + 1; j < COUNT; j++) {
      const dx = pts[i].x - pts[j].x;
      const dy = pts[i].y - pts[j].y;
      const dz = pts[i].z - pts[j].z;
      if (dx * dx + dy * dy + dz * dz < THRESH * THRESH) {
        lineSrc.push(pts[i].x, pts[i].y, pts[i].z, pts[j].x, pts[j].y, pts[j].z);
      }
    }
  }
  const lineBase = new Float32Array(lineSrc);
  const ico = buildIcosahedron(1.05, 2);

  // ——— programs ———
  const pointsProg = createProgram(gl, VS_POINTS, FS_COLOR);
  const solidProg = createProgram(gl, VS_SOLID, FS_SOLID);
  const uPointsProj = gl.getUniformLocation(pointsProg, 'uProjView');
  const uPointsSize = gl.getUniformLocation(pointsProg, 'uSize');
  const uPointsScale = gl.getUniformLocation(pointsProg, 'uScale');
  const uPointsOp = gl.getUniformLocation(pointsProg, 'uOpacity');
  const aPointsPos = gl.getAttribLocation(pointsProg, 'aPosition');
  const aPointsCol = gl.getAttribLocation(pointsProg, 'aColor');
  const uSolidProj = gl.getUniformLocation(solidProg, 'uProjView');
  const uSolidColor = gl.getUniformLocation(solidProg, 'uColor');

  // ——— buffers ———
  const pointsPosBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, pointsPosBuf);
  gl.bufferData(gl.ARRAY_BUFFER, posA, gl.DYNAMIC_DRAW);
  const pointsColBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, pointsColBuf);
  gl.bufferData(gl.ARRAY_BUFFER, colA, gl.STATIC_DRAW);
  const lineBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ARRAY_BUFFER, lineBase, gl.DYNAMIC_DRAW);
  const icoBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, icoBuf);
  gl.bufferData(gl.ARRAY_BUFFER, ico.positions, gl.DYNAMIC_DRAW);
  const icoIdx = gl.createBuffer()!;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, icoIdx);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ico.indices, gl.STATIC_DRAW);

  // scratch buffers for per-frame rotation
  const pointsRot = new Float32Array(posA.length);
  const linesRot = new Float32Array(lineBase.length);
  const icoGroupRot = new Float32Array(ico.positions.length);
  const icoRot = new Float32Array(ico.positions.length);

  // ——— renderer state ———
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  let projView = makePerspective(60, 1, 0.1, 100);

  const resize = () => {
    const rect = canvas.parentElement?.getBoundingClientRect();
    const w = rect?.width || 320;
    const h = rect?.height || 320;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    gl.viewport(0, 0, canvas.width, canvas.height);
    projView = mat4Multiply(
      makePerspective(60, w / h, 0.1, 100),
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -6.4, 1]
    );
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(canvas.parentElement as Element);

  // ——— cursor parallax ———
  let mx = 0, my = 0, cx = 0, cy = 0;
  const onMove = (e: PointerEvent) => {
    mx = (e.clientX / window.innerWidth) * 2 - 1;
    my = -((e.clientY / window.innerHeight) * 2 - 1);
  };
  window.addEventListener('pointermove', onMove, { passive: true });

  // ——— render loop ———
  let raf = 0;
  let coreRx = 0, coreRy = 0;
  const start = performance.now();
  const tick = () => {
    const t = (performance.now() - start) / 1000;
    cx += (mx - cx) * 0.045;
    cy += (my - cy) * 0.045;
    const ry = 0.0016 + cx * 0.0008 + Math.sin(t * 0.1) * 0.012;
    const rx = cy * 0.32;
    coreRx += 0.0008;
    coreRy += 0.0011;

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // points
    applyEulerXYZ(posA, pointsRot, rx, ry, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, pointsPosBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, pointsRot);
    gl.useProgram(pointsProg);
    gl.uniformMatrix4fv(uPointsProj, false, projView);
    gl.uniform1f(uPointsSize, 0.085);
    gl.uniform1f(uPointsScale, canvas.height * 0.5);
    gl.uniform1f(uPointsOp, 0.9);
    gl.enableVertexAttribArray(aPointsPos);
    gl.bindBuffer(gl.ARRAY_BUFFER, pointsPosBuf);
    gl.vertexAttribPointer(aPointsPos, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aPointsCol);
    gl.bindBuffer(gl.ARRAY_BUFFER, pointsColBuf);
    gl.vertexAttribPointer(aPointsCol, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, COUNT);

    // lines
    applyEulerXYZ(lineBase, linesRot, rx, ry, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, linesRot);
    gl.useProgram(solidProg);
    gl.uniformMatrix4fv(uSolidProj, false, projView);
    gl.uniform4f(uSolidColor, 0.941, 0.659, 0.706, 0.16);
    gl.enableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, lineBase.length / 3);

    // core (group rotation, then its own spin)
    applyEulerXYZ(ico.positions, icoGroupRot, rx, ry, 0);
    applyEulerXYZ(icoGroupRot, icoRot, coreRx, coreRy, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, icoBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, icoRot);
    gl.useProgram(solidProg);
    gl.uniformMatrix4fv(uSolidProj, false, projView);
    gl.uniform4f(uSolidColor, 0.941, 0.659, 0.706, 0.08);
    gl.enableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, icoBuf);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, icoIdx);
    gl.enable(gl.CULL_FACE);
    gl.drawElements(gl.TRIANGLES, ico.indices.length, gl.UNSIGNED_SHORT, 0);
    gl.disable(gl.CULL_FACE);

    raf = requestAnimationFrame(tick);
  };
  tick();

  // pause when offscreen (perf)
  let visible = true;
  const io = new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
    if (!visible) cancelAnimationFrame(raf);
    else raf = requestAnimationFrame(tick);
  });
  io.observe(root);

  return () => {
    window.removeEventListener('pointermove', onMove);
    ro.disconnect();
    io.disconnect();
    cancelAnimationFrame(raf);
    gl.deleteBuffer(pointsPosBuf);
    gl.deleteBuffer(pointsColBuf);
    gl.deleteBuffer(lineBuf);
    gl.deleteBuffer(icoBuf);
    gl.deleteBuffer(icoIdx);
    gl.deleteProgram(pointsProg);
    gl.deleteProgram(solidProg);
  };
}
