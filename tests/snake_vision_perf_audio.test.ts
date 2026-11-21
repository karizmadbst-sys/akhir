import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { WebSocket } from 'ws';
import { createAppServer } from '../server.ts';
import { getLevelById } from '../src/game/levels.ts';
import {
  computeSerpentLookAwayYaw,
  getSerpentWatchStandoffDist,
  isSerpentLookingAtTarget,
  SERPENT_LOOK_AWAY_DURATION_SEC,
  SERPENT_WATCH_CONE_RAD,
  SERPENT_WATCH_DURATION_SEC,
  shortestAngleDiffNav,
} from '../src/game/serpentNav.ts';
import {
  createMonsterRig,
  createPlayerRig,
  disposeEntityResources,
  setEntityQualityTier,
} from '../src/game/entities';
import type { TextureBundle } from '../src/game/textures';
import { FEMALE_FEAR_SCREAM_URL, MALE_FEAR_SCREAM_URL } from '../src/game/audio.ts';
import { GAME_VERSION, type WorldSyncPayload } from '../src/multiplayer/types.ts';

interface WsMsg {
  event: string;
  payload: any;
}

class TestClient {
  public token = '';
  public userId = '';
  public username = '';
  public ws: WebSocket | null = null;
  public inbox: WsMsg[] = [];
  private waiters: Array<{
    predicate: (msg: WsMsg) => boolean;
    resolve: (msg: WsMsg) => void;
    reject: (err: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }> = [];

  constructor(private baseUrl: string) {}

  async register(username: string, password: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = (await res.json()) as any;
    assert.equal(res.status, 201, `register failed: ${JSON.stringify(data)}`);
    this.token = data.token;
    this.userId = data.user.id;
    this.username = data.user.username;
  }

  async connectWs(): Promise<void> {
    const wsUrl = this.baseUrl.replace('http://', 'ws://') + '/ws';
    await new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      this.ws = ws;
      ws.on('open', () => {
        ws.send(JSON.stringify({ event: 'auth', payload: { token: this.token, gameVersion: GAME_VERSION } }));
        resolve();
      });
      ws.on('error', reject);
      ws.on('message', (raw) => {
        const msg = JSON.parse(raw.toString()) as WsMsg;
        let matched = false;
        for (let i = 0; i < this.waiters.length; i++) {
          const w = this.waiters[i];
          if (w.predicate(msg)) {
            this.waiters.splice(i, 1);
            clearTimeout(w.timer);
            w.resolve(msg);
            matched = true;
            break;
          }
        }
        if (!matched) this.inbox.push(msg);
      });
    });
    await this.waitFor((m) => m.event === 'auth:ok');
  }

  send(event: string, payload: unknown = {}): void {
    assert.ok(this.ws && this.ws.readyState === WebSocket.OPEN, 'WS must be open');
    this.ws.send(JSON.stringify({ event, payload }));
  }

  clearInbox(): void {
    this.inbox.length = 0;
  }

  waitFor(predicate: (msg: WsMsg) => boolean, timeoutMs = 8000): Promise<WsMsg> {
    const idx = this.inbox.findIndex(predicate);
    if (idx >= 0) {
      return Promise.resolve(this.inbox.splice(idx, 1)[0]);
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const wi = this.waiters.findIndex((w) => w.timer === timer);
        if (wi >= 0) this.waiters.splice(wi, 1);
        reject(new Error(`Timed out waiting for WS event (${timeoutMs}ms)`));
      }, timeoutMs);
      this.waiters.push({ predicate, resolve, reject, timer });
    });
  }

  close(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function testSerpentVisionMathHelpers(): void {
  assert.ok(SERPENT_WATCH_CONE_RAD > 0.3 && SERPENT_WATCH_CONE_RAD < 0.8);
  assert.equal(SERPENT_WATCH_DURATION_SEC, 5.0, 'Platform watch duration must be fixed at 5.0 seconds');
  assert.ok(getSerpentWatchStandoffDist(1.0) >= 3.5);

  // Looking straight along +Z toward target at (0, 6)
  assert.equal(isSerpentLookingAtTarget(0, 0, 0, 0, 6, 12), true);
  // Outside range
  assert.equal(isSerpentLookingAtTarget(0, 0, 0, 0, 15, 12), false);
  // Turned away (~90 deg right)
  assert.equal(isSerpentLookingAtTarget(0, 0, Math.PI * 0.5, 0, 6, 12), false);
  // Wall occlusion blocks vision
  assert.equal(
    isSerpentLookingAtTarget(0, 0, 0, 0, 6, 12, () => true),
    false
  );

  const toTarget = Math.atan2(0, 6);
  const awayPos = computeSerpentLookAwayYaw(0, 0, 0, 6, 1);
  const awayNeg = computeSerpentLookAwayYaw(0, 0, 0, 6, -1);
  assert.ok(Math.abs(shortestAngleDiffNav(toTarget, awayPos)) > 1.8, 'Look-away yaw (+1) must face well away from player');
  assert.ok(Math.abs(shortestAngleDiffNav(toTarget, awayNeg)) > 1.8, 'Look-away yaw (-1) must face well away from player');
  console.log('1. PASS: Serpent gaze cone, fixed 5s watch duration, range, wall occlusion, and look-away yaw helpers verified.');
}

async function testNodeMultiplayerVisionFreezeAndAttack(): Promise<void> {
  const srv = await createAppServer({ dbPath: ':memory:', attachVite: false, disconnectGraceMs: 800 });
  await new Promise<void>((resolve) => srv.server.listen(0, '127.0.0.1', () => resolve()));
  const addr = srv.server.address() as { port: number };
  const baseUrl = `http://127.0.0.1:${addr.port}`;

  const alice = new TestClient(baseUrl);
  const bob = new TestClient(baseUrl);

  try {
    await alice.register('VisionAlice', 'SecretPass1!');
    await bob.register('VisionBob', 'SecretPass1!');
    await alice.connectWs();
    await bob.connectWs();

    alice.send('room:create', { levelId: 1 });
    const created = await alice.waitFor((m) => m.event === 'room:state' && m.payload?.code);
    const code = created.payload.code as string;
    bob.send('room:join_code', { code });
    await bob.waitFor((m) => m.event === 'room:state' && m.payload?.players?.length === 2);

    alice.send('room:start', {});
    await Promise.all([
      alice.waitFor((m) => m.event === 'game:started'),
      bob.waitFor((m) => m.event === 'game:started'),
    ]);

    const level1 = getLevelById(1);
    const startPlat = level1.platforms[0];
    const deckY = startPlat.position[1] + startPlat.size[1] * 0.5;
    const waterY = level1.bounds.waterY;
    const swimY = waterY - 0.14;

    // 1. Out of water initially on platform: moves freely, never watched
    alice.send('game:input_state', {
      x: startPlat.position[0] + 1.0,
      y: deckY,
      z: startPlat.position[2] - 0.5,
      yaw: Math.PI,
      animState: 'run',
      horizSpeed: 7.5,
      velX: 0,
      velY: 0,
      velZ: -7.5,
      inWater: false,
      checkpointIndex: 0,
      escaped: false,
    });
    const landWorld = await alice.waitFor((m) => m.event === 'game:world');
    const landMonsters = (landWorld.payload as WorldSyncPayload).monsters;
    assert.ok(landMonsters.every((m) => !m.watchingUserId && !m.intimPhase), 'Player on platform without water exit must not be watched');

    // 2. Alice enters water and stays COMPLETELY STILL -> immediate attack & devour (single attack, no duplicate)
    const waterX = startPlat.position[0] + startPlat.size[0] * 0.5 + 1.5;
    const waterZ = startPlat.position[2] - 3.5;
    alice.clearInbox();
    bob.clearInbox();

    let aliceEatingCount = 0;
    let aliceCaughtCount = 0;
    let aliceWatchedInWater = false;
    const tWaterStart = Date.now();
    while (Date.now() - tWaterStart < 7000 && aliceCaughtCount === 0) {
      alice.send('game:input_state', {
        x: waterX,
        y: swimY,
        z: waterZ,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 0,
        velX: 0,
        velY: 0,
        velZ: 0,
        inWater: true,
        moveAttemptWhileWatched: false,
        checkpointIndex: 0,
        escaped: false,
      });
      while (alice.inbox.length > 0) {
        const msg = alice.inbox.shift()!;
        if (msg.event === 'game:world') {
          for (const m of (msg.payload as WorldSyncPayload).monsters) {
            if (m.watchingUserId === alice.userId) aliceWatchedInWater = true;
          }
        } else if (msg.event === 'game:player_event' && msg.payload?.userId === alice.userId) {
          if (msg.payload.kind === 'eating_started') aliceEatingCount++;
          if (msg.payload.kind === 'caught') aliceCaughtCount++;
        }
      }
      await sleep(45);
    }
    assert.equal(aliceWatchedInWater, false, 'Water contact must NOT trigger watch freeze');
    assert.equal(aliceEatingCount, 1, 'Stationary player touching water must be attacked once immediately');
    assert.equal(aliceCaughtCount, 1, 'Stationary player touching water must be devoured once');

    // Wait for Alice to respawn on startPlat before multi-player independence check
    await alice.waitFor((m) => m.event === 'game:player_event' && m.payload?.kind === 'respawned' && m.payload?.userId === alice.userId, 6000);

    // 3. Bob enters water briefly, climbs onto startPlat, and stays still ->
    //    serpent arrives, rises (NOT watching/frozen yet), then stares for 5.0s, then lowers & retreats.
    //    Meanwhile Alice runs freely on the same platform without being affected!
    const bobWaterX = startPlat.position[0] - startPlat.size[0] * 0.5 - 1.4;
    const bobWaterZ = startPlat.position[2] - 2.0;
    bob.send('game:input_state', {
      x: bobWaterX,
      y: swimY,
      z: bobWaterZ,
      yaw: 0,
      animState: 'swim',
      horizSpeed: 0,
      velX: 0,
      velY: 0,
      velZ: 0,
      inWater: true,
      checkpointIndex: 0,
      escaped: false,
    });
    await sleep(90);

    bob.clearInbox();
    alice.clearInbox();
    const bobPlatX = startPlat.position[0] - 0.8;
    const bobPlatZ = startPlat.position[2];
    let sawApproachingOrRisingNotFrozen = false;
    let stareStartMs = 0;
    let stareEndMs = 0;
    let bobDiedDuringSafeWatch = false;
    let aliceAffectedByBobsWatch = false;
    let stepTick = 0;

    const tPlatStart = Date.now();
    while (Date.now() - tPlatStart < 12000 && !(stareEndMs > 0 && stareEndMs > stareStartMs)) {
      stepTick++;
      bob.send('game:input_state', {
        x: bobPlatX,
        y: deckY,
        z: bobPlatZ,
        yaw: Math.PI,
        animState: 'idle',
        horizSpeed: 0,
        velX: 0,
        velY: 0,
        velZ: 0,
        inWater: false,
        moveAttemptWhileWatched: false,
        checkpointIndex: 0,
        escaped: false,
      });
      // Alice moves around freely on the same platform throughout Bob's watch
      alice.send('game:input_state', {
        x: startPlat.position[0] + 1.2 + Math.sin(stepTick * 0.25) * 0.6,
        y: deckY,
        z: startPlat.position[2] + 0.5,
        yaw: 0,
        animState: 'run',
        horizSpeed: 6.0,
        velX: 2.0,
        velY: 0,
        velZ: 0,
        inWater: false,
        checkpointIndex: 0,
        escaped: false,
      });

      while (bob.inbox.length > 0) {
        const msg = bob.inbox.shift()!;
        if (msg.event === 'game:world') {
          const w = msg.payload as WorldSyncPayload;
          const watchers = w.monsters.filter((m) => m.watchingUserId === bob.userId);
          assert.ok(watchers.length <= 1, 'Must never assign multiple watchers to the same player');
          for (const m of w.monsters) {
            if (m.watchingUserId === alice.userId) aliceAffectedByBobsWatch = true;
            if ((m.intimPhase === 'approaching' || m.intimPhase === 'rising') && !m.watchingUserId) {
              sawApproachingOrRisingNotFrozen = true;
            }
            if (m.intimPhase === 'staring' && m.watchingUserId === bob.userId) {
              if (stareStartMs === 0) stareStartMs = Date.now();
            } else if (stareStartMs > 0 && stareEndMs === 0 && (m.intimPhase === 'lowering' || m.intimPhase === 'retreating')) {
              stareEndMs = Date.now();
            }
          }
        } else if (msg.event === 'game:player_event') {
          if ((msg.payload?.kind === 'eating_started' || msg.payload?.kind === 'caught') && msg.payload?.userId === bob.userId) {
            bobDiedDuringSafeWatch = true;
          }
          if ((msg.payload?.kind === 'eating_started' || msg.payload?.kind === 'caught') && msg.payload?.userId === alice.userId) {
            aliceAffectedByBobsWatch = true;
          }
        }
      }
      await sleep(45);
    }

    assert.ok(sawApproachingOrRisingNotFrozen, 'Watch freeze & 5s countdown must NOT start before serpent arrives & finishes rising');
    assert.ok(stareStartMs > 0 && stareEndMs > stareStartMs, 'Serpent must complete 5.0s platform stare');
    const measuredStareMs = stareEndMs - stareStartMs;
    assert.ok(measuredStareMs >= 4750 && measuredStareMs <= 5600, `Expected ~5000ms platform stare, got ${measuredStareMs}ms`);
    assert.equal(bobDiedDuringSafeWatch, false, 'Still player on platform must NOT be attacked during or after 5s watch');
    assert.equal(aliceAffectedByBobsWatch, false, 'Watching Bob MUST NOT affect or freeze Alice in Multiplayer');

    // 4. Alice enters water briefly -> climbs onto startPlat -> waits for staring -> moves -> immediate strike & single devour
    alice.send('game:input_state', {
      x: waterX,
      y: swimY,
      z: startPlat.position[2] - 2.0,
      yaw: 0,
      animState: 'swim',
      horizSpeed: 0,
      velX: 0,
      velY: 0,
      velZ: 0,
      inWater: true,
      checkpointIndex: 0,
      escaped: false,
    });
    await sleep(90);

    alice.clearInbox();
    let aliceWatchedOnPlat = false;
    let aliceStrikeEat = 0;
    let aliceStrikeCaught = 0;
    const tStrikeStart = Date.now();
    while (Date.now() - tStrikeStart < 9500 && aliceStrikeCaught === 0) {
      alice.send('game:input_state', {
        x: startPlat.position[0] + (aliceWatchedOnPlat ? 1.4 : 0.6),
        y: deckY,
        z: startPlat.position[2],
        yaw: 0,
        animState: aliceWatchedOnPlat ? 'run' : 'idle',
        horizSpeed: aliceWatchedOnPlat ? 4.5 : 0,
        velX: aliceWatchedOnPlat ? 4.5 : 0,
        velY: 0,
        velZ: 0,
        inWater: false,
        moveAttemptWhileWatched: aliceWatchedOnPlat,
        checkpointIndex: 0,
        escaped: false,
      });
      while (alice.inbox.length > 0) {
        const msg = alice.inbox.shift()!;
        if (msg.event === 'game:world') {
          for (const m of (msg.payload as WorldSyncPayload).monsters) {
            if (m.intimPhase === 'staring' && m.watchingUserId === alice.userId) {
              aliceWatchedOnPlat = true;
            }
          }
        } else if (msg.event === 'game:player_event' && msg.payload?.userId === alice.userId) {
          if (msg.payload.kind === 'eating_started') aliceStrikeEat++;
          if (msg.payload.kind === 'caught') aliceStrikeCaught++;
        }
      }
      await sleep(45);
    }
    assert.ok(aliceWatchedOnPlat, 'Serpent must stare at Alice after she climbs onto platform');
    assert.equal(aliceStrikeEat, 1, 'Moving during 5s platform stare must trigger single immediate strike');
    assert.equal(aliceStrikeCaught, 1, 'Moving during 5s platform stare must devour player');

    // 5. Bob touches water a SECOND time while completely still -> immediately attacked & devoured again
    bob.clearInbox();
    let bobReEntryEat = 0;
    let bobReEntryCaught = 0;
    const tReEntry = Date.now();
    while (Date.now() - tReEntry < 7000 && bobReEntryCaught === 0) {
      bob.send('game:input_state', {
        x: bobWaterX,
        y: swimY,
        z: bobWaterZ,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 0,
        velX: 0,
        velY: 0,
        velZ: 0,
        inWater: true,
        moveAttemptWhileWatched: false,
        checkpointIndex: 0,
        escaped: false,
      });
      while (bob.inbox.length > 0) {
        const msg = bob.inbox.shift()!;
        if (msg.event === 'game:player_event' && msg.payload?.userId === bob.userId) {
          if (msg.payload.kind === 'eating_started') bobReEntryEat++;
          if (msg.payload.kind === 'caught') bobReEntryCaught++;
        }
      }
      await sleep(45);
    }
    assert.equal(bobReEntryEat, 1, 'Second stationary water contact must immediately attack again');
    assert.equal(bobReEntryCaught, 1, 'Second stationary water contact must devour player');

    console.log('2. PASS: Node Multiplayer water-devour, 5s platform watch/freeze, move strike, MP isolation, and water re-entry verified.');
  } finally {
    alice.close();
    bob.close();
    await srv.close();
  }
}

async function testCloudflareDOVisionParity(): Promise<void> {
  const roomModPath = '../cloudflare/src/' + 'roomDO.ts';
  const { GameRoomDO } = await import(/* @vite-ignore */ roomModPath);
  const deliveredBatches: any[] = [];
  const fakeHubStub = {
    fetch: async (_url: string, init?: RequestInit) => {
      if (init?.body) deliveredBatches.push(JSON.parse(String(init.body)));
      return Response.json({ ok: true });
    },
  };
  const fakeCtx: any = {
    storage: { setAlarm: async () => {}, deleteAlarm: async () => {} },
    waitUntil: (p: Promise<unknown>) => { void p.catch(() => {}); },
  };
  const fakeEnv = {
    PRESENCE_HUB: { idFromName: () => 'hub', get: () => fakeHubStub },
    DISCONNECT_GRACE_MS: '8000',
  } as any;

  const roomDO = new GameRoomDO(fakeCtx, fakeEnv);
  const callRoom = async (payload: Record<string, unknown>) => {
    const res = await roomDO.fetch(
      new Request('https://room/internal/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    );
    return (await res.json()) as { room?: any; error?: string; batch?: any };
  };

  const userA = { id: 'v-cf-1', username: 'VCFAlice', avatarColor: '#52e0b6', avatarIcon: 'diver' as const, status: 'online' as const };
  const userB = { id: 'v-cf-2', username: 'VCFBob', avatarColor: '#f59e0b', avatarIcon: 'diver' as const, status: 'online' as const };
  await callRoom({ action: 'create', roomId: 'room-vcf', code: 'VCF123', user: userA, levelId: 1, gameVersion: GAME_VERSION });
  await callRoom({ action: 'join', user: userB, gameVersion: GAME_VERSION });
  await callRoom({ action: 'start', userId: userA.id });

  const level1 = getLevelById(1);
  const startPlat = level1.platforms[0];
  const deckY = startPlat.position[1] + startPlat.size[1] * 0.5;
  const waterY = level1.bounds.waterY;
  const swimX = startPlat.position[0] + startPlat.size[0] * 0.5 + 1.2;
  const swimZ = startPlat.position[2] - 2.5;

  const drainDeliveries = (resBatch: any) => {
    const out: any[] = [...(resBatch?.deliveries || [])];
    while (deliveredBatches.length > 0) {
      const b = deliveredBatches.shift()!;
      if (b?.deliveries) out.push(...b.deliveries);
    }
    return out;
  };

  // 1. Alice stationary in water -> immediate attack & devour without watching
  let aEat = 0;
  let aCaught = 0;
  let aWatchedInWater = false;
  const waterDeadline = Date.now() + 7000;
  while (Date.now() < waterDeadline && aCaught === 0) {
    const res = await callRoom({
      action: 'input_state',
      userId: userA.id,
      packet: {
        x: swimX,
        y: waterY - 0.14,
        z: swimZ,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 0,
        velX: 0,
        velY: 0,
        velZ: 0,
        inWater: true,
        moveAttemptWhileWatched: false,
        checkpointIndex: 0,
        escaped: false,
      },
    });
    for (const d of drainDeliveries(res.batch)) {
      if (d.event === 'game:world' && d.targetUserId === userA.id) {
        for (const m of (d.payload as WorldSyncPayload).monsters) {
          if (m.watchingUserId === userA.id) aWatchedInWater = true;
        }
      } else if (d.event === 'game:player_event' && d.payload?.userId === userA.id) {
        if (d.payload.kind === 'eating_started') aEat++;
        if (d.payload.kind === 'caught') aCaught++;
      }
    }
    await sleep(50);
  }
  assert.equal(aWatchedInWater, false, 'CF DO must not freeze stationary swimmer in water');
  assert.equal(aEat, 1, 'CF DO must immediately attack stationary swimmer once');
  assert.equal(aCaught, 1, 'CF DO must devour stationary swimmer');

  // 2. Bob briefly touches water then climbs onto platform -> arrives -> rises -> 5s staring -> retreats cleanly
  await callRoom({
    action: 'input_state',
    userId: userB.id,
    packet: {
      x: startPlat.position[0] - startPlat.size[0] * 0.5 - 1.2,
      y: waterY - 0.14,
      z: startPlat.position[2] - 2.0,
      yaw: 0,
      animState: 'swim',
      horizSpeed: 0,
      velX: 0,
      velY: 0,
      velZ: 0,
      inWater: true,
      checkpointIndex: 0,
      escaped: false,
    },
  });
  await sleep(80);

  let cfStareStart = 0;
  let cfStareEnd = 0;
  let cfNotFrozenEarly = false;
  let bobDiedDuringSafe = false;
  const platDeadline = Date.now() + 12000;
  while (Date.now() < platDeadline && !(cfStareEnd > 0 && cfStareEnd > cfStareStart)) {
    const res = await callRoom({
      action: 'input_state',
      userId: userB.id,
      packet: {
        x: startPlat.position[0] - 0.5,
        y: deckY,
        z: startPlat.position[2],
        yaw: Math.PI,
        animState: 'idle',
        horizSpeed: 0,
        velX: 0,
        velY: 0,
        velZ: 0,
        inWater: false,
        moveAttemptWhileWatched: false,
        checkpointIndex: 0,
        escaped: false,
      },
    });
    for (const d of drainDeliveries(res.batch)) {
      if (d.event === 'game:world' && d.targetUserId === userB.id) {
        for (const m of (d.payload as WorldSyncPayload).monsters) {
          if ((m.intimPhase === 'approaching' || m.intimPhase === 'rising') && !m.watchingUserId) {
            cfNotFrozenEarly = true;
          }
          if (m.intimPhase === 'staring' && m.watchingUserId === userB.id) {
            if (cfStareStart === 0) cfStareStart = Date.now();
          } else if (cfStareStart > 0 && cfStareEnd === 0 && (m.intimPhase === 'lowering' || m.intimPhase === 'retreating')) {
            cfStareEnd = Date.now();
          }
        }
      } else if (d.event === 'game:player_event' && d.payload?.userId === userB.id) {
        if (d.payload.kind === 'eating_started' || d.payload.kind === 'caught') bobDiedDuringSafe = true;
      }
    }
    await sleep(50);
  }
  assert.ok(cfNotFrozenEarly, 'CF DO must not start freeze/timer before serpent arrives & finishes rising');
  assert.ok(cfStareStart > 0 && cfStareEnd > cfStareStart, 'CF DO must complete 5s stare on platform');
  const cfDur = cfStareEnd - cfStareStart;
  assert.ok(cfDur >= 4750 && cfDur <= 5600, `Expected ~5000ms CF DO stare, got ${cfDur}ms`);
  assert.equal(bobDiedDuringSafe, false, 'CF DO must not attack player who stays still during 5s platform watch');

  await callRoom({ action: 'close', userId: userA.id });
  console.log('3. PASS: Cloudflare GameRoomDO water-devour & 5s platform-watch parity verified.');
}

function testShaderRecompileEliminationAndAudioPreservation(): void {
  const mk = () => new THREE.DataTexture(new Uint8Array([180, 180, 180, 255]), 1, 1) as unknown as THREE.CanvasTexture;
  const all = Array.from({ length: 23 }, () => mk());
  const tex: TextureBundle = {
    compact: false,
    tileAlbedo: all[0],
    tileBump: all[1],
    tileRoughness: all[2],
    wallAlbedo: all[3],
    wallBump: all[4],
    concreteAlbedo: all[5],
    concreteBump: all[6],
    metalAlbedo: all[7],
    metalBump: all[8],
    cautionAlbedo: all[9],
    woodAlbedo: all[10],
    woodBump: all[11],
    woodRoughness: all[12],
    ceilingAlbedo: all[13],
    waterNormal: all[14],
    monsterSkin: all[15],
    monsterBump: all[16],
    monsterNormal: all[17],
    monsterRoughness: all[18],
    monsterEmissive: all[19],
    monsterFinSkin: all[20],
    monsterFinEmissive: all[21],
    monsterEye: all[22],
    setAnisotropy: () => {},
    markAllDirty: () => {},
    dispose: () => {
      for (const t of all) t.dispose();
    },
  };

  const monster = createMonsterRig(tex, 1.0);
  const skinMat = monster.bodyMesh.material as THREE.MeshStandardMaterial;
  const initialNormalMap = skinMat.normalMap;
  assert.ok(initialNormalMap, 'Monster skin must bind normalMap initially');

  setEntityQualityTier(tex, 'smooth');
  assert.equal(skinMat.normalMap, initialNormalMap, 'Monster normalMap must stay bound in smooth tier to avoid shader stalls');
  assert.ok(skinMat.normalScale.x > 0 && skinMat.normalScale.x < 1.0, 'Smooth tier modulates normalScale without dropping map');

  setEntityQualityTier(tex, 'ultra');
  assert.equal(skinMat.normalMap, initialNormalMap);
  assert.ok(skinMat.normalScale.x >= 1.85);

  const rig = createPlayerRig(tex, 0x52e0b6, false, 'female');
  setEntityQualityTier(tex, 'smooth');
  let foundSkinMat: THREE.MeshStandardMaterial | null = null;
  rig.root.traverse((o) => {
    if (o instanceof THREE.Mesh && o.material instanceof THREE.MeshStandardMaterial && o.material.bumpMap) {
      foundSkinMat = o.material;
    }
  });
  assert.ok(foundSkinMat, 'Character materials keep bumpMap permanently bound in smooth tier');

  rig.dispose();
  monster.dispose();
  disposeEntityResources(tex);
  tex.dispose();

  // Verify original scream MP3s, trimming & timing are untouched, while peak gains are moderately boosted
  const audioSrc = fs.readFileSync(path.join(process.cwd(), 'src/game/audio.ts'), 'utf8');
  assert.equal(FEMALE_FEAR_SCREAM_URL, '/assets/aistudio/1795113025845_20261119_183259_dda4ed84.mp3');
  assert.equal(MALE_FEAR_SCREAM_URL, '/assets/aistudio/1795113054247_20261119_183222_15cf3533.mp3');
  assert.ok(audioSrc.includes('leadTrimSec: 0,') && audioSrc.includes('maxDurationSec: 2.25,'), 'Female scream timing preserved');
  assert.ok(audioSrc.includes('leadTrimSec: 0.09,') && audioSrc.includes('maxDurationSec: 2.28,'), 'Male scream timing preserved');
  assert.ok(audioSrc.includes('localPeakGain: 1.72') && audioSrc.includes('localPeakGain: 1.12'), 'Scream volume slightly increased');
  console.log('4. PASS: Zero shader-recompile quality tiering & original scream MP3/timing preservation verified.');
}

(async () => {
  testSerpentVisionMathHelpers();
  await testNodeMultiplayerVisionFreezeAndAttack();
  await testCloudflareDOVisionParity();
  testShaderRecompileEliminationAndAudioPreservation();
  console.log('\nALL SNAKE VISION, PERFORMANCE & AUDIO VERIFICATION TESTS PASSED!');
  process.exit(0);
})().catch((err) => {
  console.error('\nTEST FAILED:', err);
  process.exit(1);
});
