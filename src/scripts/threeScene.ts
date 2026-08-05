import * as THREE from 'three';

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

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.set(0, 0, 6.4);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  // ——— nodes on a fibonacci sphere ———
  const COUNT = 220;
  const RADIUS = 2.5;
  const positions = new Float32Array(COUNT * 3);
  const colors = new Float32Array(COUNT * 3);
  const vA = new THREE.Color('#e04a10'); // primary orange
  const vB = new THREE.Color('#c2410c'); // burnt orange
  const vC = new THREE.Color('#d97706'); // amber

  const pts: { x: number; y: number; z: number }[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < COUNT; i++) {
    const y = 1 - (i / (COUNT - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = golden * i;
    const x = Math.cos(theta) * r;
    const z = Math.sin(theta) * r;
    positions[i * 3] = x * RADIUS;
    positions[i * 3 + 1] = y * RADIUS;
    positions[i * 3 + 2] = z * RADIUS;
    pts.push({ x: x * RADIUS, y: y * RADIUS, z: z * RADIUS });

    const t = Math.random();
    const c = t < 0.12 ? vC : t < 0.3 ? vB : vA;
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.085,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    blending: THREE.NormalBlending,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(g, mat);

  // ——— connection lines (near-neighbour web) ———
  const linePositions: number[] = [];
  const THRESH = 1.25;
  for (let i = 0; i < COUNT; i++) {
    for (let j = i + 1; j < COUNT; j++) {
      const dx = pts[i].x - pts[j].x;
      const dy = pts[i].y - pts[j].y;
      const dz = pts[i].z - pts[j].z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < THRESH) {
        linePositions.push(pts[i].x, pts[i].y, pts[i].z, pts[j].x, pts[j].y, pts[j].z);
      }
    }
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(linePositions), 3));
  const lmat = new THREE.LineBasicMaterial({
    color: new THREE.Color('#f05010'),
    transparent: true,
    opacity: 0.16,
    blending: THREE.NormalBlending,
  });
  const lines = new THREE.LineSegments(lg, lmat);

  // inner glow core
  const coreMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color('#f05010'),
    transparent: true,
    opacity: 0.08,
    blending: THREE.NormalBlending,
  });
  const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.05, 2), coreMat);

  const group = new THREE.Group();
  group.add(points, lines, core);
  scene.add(group);

  // ——— cursor parallax ———
  let mx = 0;
  let my = 0;
  let cx = 0;
  let cy = 0;
  const onMove = (e: PointerEvent) => {
    mx = (e.clientX / window.innerWidth) * 2 - 1;
    my = -((e.clientY / window.innerHeight) * 2 - 1);
  };
  window.addEventListener('pointermove', onMove, { passive: true });

  const resize = () => {
    const rect = canvas.parentElement?.getBoundingClientRect();
    const w = rect?.width || 320;
    const h = rect?.height || 320;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(canvas.parentElement as Element);

  // ——— render loop ———
  const clock = new THREE.Clock();
  let raf = 0;
  const tick = () => {
    const t = clock.getElapsedTime();
    cx += (mx - cx) * 0.045;
    cy += (my - cy) * 0.045;
    group.rotation.y += 0.0016;
    group.rotation.x = cy * 0.32;
    group.rotation.y += cx * 0.0008 + Math.sin(t * 0.1) * 0.012;
    core.rotation.x += 0.0008;
    core.rotation.y += 0.0011;
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  };
  tick();

  // cleanup when the section leaves the viewport (perf) — pause instead
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
    g.dispose();
    lg.dispose();
    mat.dispose();
    lmat.dispose();
    coreMat.dispose();
    renderer.dispose();
  };
}
