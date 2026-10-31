// Physical topology: 4 HGX-style nodes x 8 GPUs, one scale-out NIC per GPU,
// and a two-tier leaf-spine fabric wired either rail-optimized or ToR-style.
// Both wirings are 1:1 non-blocking with identical NIC, uplink and spine port
// counts (ADR §18); only the NIC-to-leaf wiring differs.

export const NODES = 4;
export const GPUS_PER_NODE = 8;
export const SPINES = 4;
export const GPU_COUNT = NODES * GPUS_PER_NODE;

export type Wiring = 'rail' | 'tor';

export interface Gpu {
  id: string;
  node: number;
  /** Local index inside the node; equals the rail number. */
  local: number;
  /** Global index 0..31. */
  index: number;
  nicId: string;
}

export type SwitchKind = 'nvswitch' | 'leaf' | 'spine';

export interface Switch {
  id: string;
  kind: SwitchKind;
  index: number;
  /** Rail served by a leaf in rail-optimized wiring; node served by a ToR leaf. */
  rail?: number;
  node?: number;
}

/**
 * - nvlink: GPU <-> scale-up switch inside a node (vendor-neutral name for NVSwitch / UALink / SUE-class switches)
 * - access: GPU NIC <-> leaf (scale-out)
 * - fabric: leaf <-> spine (scale-out); ToR wiring has two parallel links per pair
 */
export type LinkKind = 'nvlink' | 'access' | 'fabric';

export interface Link {
  id: string;
  kind: LinkKind;
  a: string;
  b: string;
  /** Index among parallel links between the same pair (fabric links only). */
  parallel: number;
  /** Relative bandwidth; every link is 1 unit (ADR §18). */
  capacity: number;
}

export interface Topology {
  wiring: Wiring;
  gpus: Gpu[];
  switches: Switch[];
  links: Link[];
  /** Leaf id each GPU's NIC is cabled to. */
  leafOfGpu: Map<string, string>;
  /** Fabric links grouped by leaf id, then by spine id. */
  fabric: Map<string, Map<string, Link[]>>;
  /** Access link for each GPU id. */
  accessOfGpu: Map<string, Link>;
  /** NVLink for each GPU id. */
  nvlinkOfGpu: Map<string, Link>;
  gpuMap: Map<string, Gpu>;
  linkMap: Map<string, Link>;
}

export function gpuId(node: number, local: number): string {
  return `g${node}.${local}`;
}

export function nvswitchId(node: number): string {
  return `nvs${node}`;
}

export function buildTopology(wiring: Wiring): Topology {
  const gpus: Gpu[] = [];
  const switches: Switch[] = [];
  const links: Link[] = [];
  const leafOfGpu = new Map<string, string>();
  const fabric = new Map<string, Map<string, Link[]>>();
  const accessOfGpu = new Map<string, Link>();
  const nvlinkOfGpu = new Map<string, Link>();

  for (let n = 0; n < NODES; n++) {
    switches.push({ id: nvswitchId(n), kind: 'nvswitch', index: n, node: n });
    for (let l = 0; l < GPUS_PER_NODE; l++) {
      const id = gpuId(n, l);
      gpus.push({ id, node: n, local: l, index: n * GPUS_PER_NODE + l, nicId: `nic${n}.${l}` });
      const nv: Link = { id: `nv:${id}`, kind: 'nvlink', a: id, b: nvswitchId(n), parallel: 0, capacity: 1 };
      links.push(nv);
      nvlinkOfGpu.set(id, nv);
    }
  }

  const leafCount = wiring === 'rail' ? GPUS_PER_NODE : NODES;
  for (let i = 0; i < leafCount; i++) {
    switches.push(
      wiring === 'rail'
        ? { id: `leaf${i}`, kind: 'leaf', index: i, rail: i }
        : { id: `leaf${i}`, kind: 'leaf', index: i, node: i },
    );
  }
  for (let s = 0; s < SPINES; s++) {
    switches.push({ id: `spine${s}`, kind: 'spine', index: s });
  }

  for (const g of gpus) {
    const leaf = wiring === 'rail' ? `leaf${g.local}` : `leaf${g.node}`;
    leafOfGpu.set(g.id, leaf);
    const acc: Link = { id: `acc:${g.nicId}`, kind: 'access', a: g.id, b: leaf, parallel: 0, capacity: 1 };
    links.push(acc);
    accessOfGpu.set(g.id, acc);
  }

  // Uplinks per leaf equal its downlinks (1:1). Spread them evenly over the spines.
  const downlinksPerLeaf = GPU_COUNT / leafCount;
  const parallelPerSpine = downlinksPerLeaf / SPINES;
  for (let i = 0; i < leafCount; i++) {
    const leaf = `leaf${i}`;
    const bySpine = new Map<string, Link[]>();
    for (let s = 0; s < SPINES; s++) {
      const spine = `spine${s}`;
      const group: Link[] = [];
      for (let p = 0; p < parallelPerSpine; p++) {
        const l: Link = { id: `fab:${leaf}-${spine}#${p}`, kind: 'fabric', a: leaf, b: spine, parallel: p, capacity: 1 };
        links.push(l);
        group.push(l);
      }
      bySpine.set(spine, group);
    }
    fabric.set(leaf, bySpine);
  }

  const gpuMap = new Map(gpus.map((g) => [g.id, g]));
  const linkMap = new Map(links.map((l) => [l.id, l]));
  return { wiring, gpus, switches, links, leafOfGpu, fabric, accessOfGpu, nvlinkOfGpu, gpuMap, linkMap };
}

export function gpuById(topo: Topology, id: string): Gpu {
  const g = topo.gpuMap.get(id);
  if (!g) throw new Error(`unknown GPU ${id}`);
  return g;
}
