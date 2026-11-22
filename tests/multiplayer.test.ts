import assert from 'node:assert/strict';
import type { AddressInfo } from 'node:net';
import { WebSocket } from 'ws';
import { createAppServer } from '../server.ts';
import {
  LEVELS,
  TOTAL_LEVELS,
  getLevelById,
  getLevelSummary,
  isValidLevelId,
} from '../src/game/levels.ts';
import {
  EAT_DURATION_MS,
  GAME_VERSION,
  MAX_ROOM_PLAYERS,
  MIN_ROOM_PLAYERS,
  PUSH_COOLDOWN_MS,
  PUSH_FORCE_HORIZ,
  PUSH_FORCE_LIFT_Y,
  PUSH_RECOVERY_MS,
  RESPAWN_DELAY_MS,
  type GameLevelChangedPayload,
  type GamePlayerEvent,
  type PublicUser,
  type RoomInviteInfo,
  type RoomState,
  type WorldSyncPayload,
} from '../src/multiplayer/types.ts';
import { isPushPathObstructed, selectPushTarget } from '../src/multiplayer/pushPhysics.ts';

interface TestPacket {
  event: string;
  payload: unknown;
}

class TestClient {
  public user!: PublicUser;
  public token!: string;
  public ws!: WebSocket;
  public currentRoom: RoomState | null = null;
  public invites: RoomInviteInfo[] = [];
  public packets: TestPacket[] = [];
  private baseUrl: string;
  private wsUrl: string;
  private waiters: Array<(pkt: TestPacket) => boolean> = [];

  constructor(baseUrl: string, wsUrl: string) {
    this.baseUrl = baseUrl;
    this.wsUrl = wsUrl;
  }

  public async register(username: string, password = 'password123'): Promise<void> {
    const res = await fetch(`${this.baseUrl}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = (await res.json()) as { user?: PublicUser; token?: string; error?: string };
    if (!res.ok || !data.token || !data.user) {
      throw new Error(data.error || `Register failed for ${username}`);
    }
    this.user = data.user;
    this.token = data.token;
    await this.connectWs();
  }

  public async connectWs(): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      this.ws = ws;

      const timer = setTimeout(() => {
        reject(new Error(`WS auth timeout for ${this.user?.username}`));
      }, 4000);

      ws.on('open', () => {
        ws.send(JSON.stringify({ event: 'auth', payload: { token: this.token, gameVersion: GAME_VERSION } }));
      });

      ws.on('message', (raw) => {
        const pkt = JSON.parse(raw.toString('utf8')) as TestPacket;
        this.packets.push(pkt);

        if (pkt.event === 'auth:ok') {
          const p = pkt.payload as {
            user: PublicUser;
            pendingInvites: RoomInviteInfo[];
            activeRoom: RoomState | null;
          };
          this.user = p.user;
          this.invites = p.pendingInvites;
          this.currentRoom = p.activeRoom;
          clearTimeout(timer);
          resolve();
        } else if (pkt.event === 'room:state') {
          this.currentRoom = pkt.payload as RoomState | null;
        } else if (pkt.event === 'room:closed' || pkt.event === 'room:kicked') {
          this.currentRoom = null;
        } else if (pkt.event === 'invite:received') {
          this.invites.push(pkt.payload as RoomInviteInfo);
        } else if (pkt.event === 'invite:removed') {
          const rm = pkt.payload as { inviteId: string };
          this.invites = this.invites.filter((i) => i.id !== rm.inviteId);
        }

        this.waiters = this.waiters.filter((fn) => !fn(pkt));
      });

      ws.on('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  public send(event: string, payload: Record<string, unknown> = {}): void {
    this.ws.send(JSON.stringify({ event, payload }));
  }

  public waitFor(
    predicate: (pkt: TestPacket) => boolean,
    label: string,
    timeoutMs = 3500
  ): Promise<TestPacket> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Timed out waiting for ${label} (${this.user?.username})`));
      }, timeoutMs);

      const matcher = (pkt: TestPacket): boolean => {
        if (predicate(pkt)) {
          clearTimeout(timer);
          resolve(pkt);
          return true;
        }
        return false;
      };
      this.waiters.push(matcher);
    });
  }

  public async http(path: string, method = 'GET', body?: unknown): Promise<{ status: number; data: Record<string, unknown> }> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.token}`,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const data = (await res.json()) as Record<string, unknown>;
    return { status: res.status, data };
  }

  public close(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      this.ws.close();
    }
  }
}

async function runTests() {
  const srv = await createAppServer({
    dbPath: ':memory:',
    attachVite: false,
    disconnectGraceMs: 60,
  });

  await new Promise<void>((resolve) => srv.server.listen(0, '127.0.0.1', () => resolve()));
  const port = (srv.server.address() as AddressInfo).port;
  const baseUrl = `http://127.0.0.1:${port}`;
  const wsUrl = `ws://127.0.0.1:${port}/ws`;

  const clients: TestClient[] = [];
  const makeClient = async (name: string) => {
    const c = new TestClient(baseUrl, wsUrl);
    await c.register(name);
    clients.push(c);
    return c;
  };

  try {
    console.log('1. Testing user accounts, search, and friend requests...');
    const host = await makeClient('HostPlayer');
    const friend = await makeClient('FriendPlayer');
    const stranger = await makeClient('StrangerPlayer');

    // Search returns real users only
    const searchRes = await host.http('/api/users/search?q=Friend');
    assert.equal(searchRes.status, 200);
    const searchUsers = searchRes.data.users as Array<{ id: string; username: string; relationship: string }>;
    assert.equal(searchUsers.length, 1);
    assert.equal(searchUsers[0].username, 'FriendPlayer');
    assert.equal(searchUsers[0].relationship, 'none');

    // Send & prevent duplicate friend request
    const frRes = await host.http('/api/friends/request', 'POST', { targetUserId: friend.user.id });
    assert.equal(frRes.status, 201);
    const dupFrRes = await host.http('/api/friends/request', 'POST', { targetUserId: friend.user.id });
    assert.equal(dupFrRes.status, 400);

    // Reject flow works
    const rejectFr = await stranger.http('/api/friends/request', 'POST', { targetUserId: friend.user.id });
    assert.equal(rejectFr.status, 201);
    const friendListBefore = await friend.http('/api/friends');
    const incomingList = friendListBefore.data.incomingRequests as Array<{ id: string; fromUserId: string }>;
    const strangerReq = incomingList.find((r) => r.fromUserId === stranger.user.id)!;
    const hostReq = incomingList.find((r) => r.fromUserId === host.user.id)!;

    await friend.http('/api/friends/respond', 'POST', { requestId: strangerReq.id, action: 'reject' });
    await friend.http('/api/friends/respond', 'POST', { requestId: hostReq.id, action: 'accept' });

    const hostFriendsRes = await host.http('/api/friends');
    const hostFriends = hostFriendsRes.data.friends as Array<{ id: string; status: string }>;
    assert.equal(hostFriends.length, 1);
    assert.equal(hostFriends[0].id, friend.user.id);
    assert.equal(hostFriends[0].status, 'online');
    console.log('   PASS: Real accounts, search, friend request accept/reject & live presence verified.');

    console.log('2. Testing Scenario 1: 1 player cannot start match (min 2 players enforced)...');
    const roomCreatedWait = host.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'room:create');
    host.send('room:create', { levelId: 1 });
    await roomCreatedWait;
    assert.ok(host.currentRoom);
    assert.equal(host.currentRoom.players.length, 1);
    assert.equal(host.currentRoom.minPlayers, MIN_ROOM_PLAYERS);
    assert.equal(host.currentRoom.maxPlayers, MAX_ROOM_PLAYERS);

    const startBlockedWait = host.waitFor((p) => p.event === 'room:error', 'room:start blocked for 1 player');
    host.send('room:start');
    const blockedErr = await startBlockedWait;
    assert.match((blockedErr.payload as { message: string }).message, /At least 2 real players/i);
    console.log('   PASS: 1 player strictly prevented from starting.');

    console.log('3. Testing Invalid room code, private invites, and Scenario 2 (2 players start + shared serpent sync)...');
    const invalidJoin = await stranger.http('/api/rooms/join', 'POST', { code: 'ZZZZZZ' });
    assert.equal(invalidJoin.status, 400);

    // Send invite & test duplicate invite prevention
    const inviteReceivedWait = friend.waitFor((p) => p.event === 'invite:received', 'invite received');
    host.send('room:invite', { friendUserId: friend.user.id });
    const invPkt = await inviteReceivedWait;
    const inv = invPkt.payload as RoomInviteInfo;

    const dupInvWait = host.waitFor((p) => p.event === 'room:error', 'duplicate invite blocked');
    host.send('room:invite', { friendUserId: friend.user.id });
    await dupInvWait;

    // Non-host cannot start
    const acceptJoinWait = friend.waitFor(
      (p) => p.event === 'room:state' && (p.payload as RoomState | null)?.players.length === 2,
      'friend joined via invite'
    );
    friend.send('room:invite_respond', { inviteId: inv.id, accept: true });
    await acceptJoinWait;
    assert.equal(friend.currentRoom?.players.length, 2);
    assert.ok(friend.currentRoom?.players.every((p) => p.isBot === false));

    const nonHostStartWait = friend.waitFor((p) => p.event === 'room:error', 'non-host start blocked');
    friend.send('room:start');
    await nonHostStartWait;

    // 2 real players CAN start match
    const hostStartWait = host.waitFor((p) => p.event === 'game:started', 'host game:started');
    const friendStartWait = friend.waitFor((p) => p.event === 'game:started', 'friend game:started');
    host.send('room:start');
    await Promise.all([hostStartWait, friendStartWait]);

    // Verify shared sea serpent chases friend in water, triggers synchronized eating_started, allows host to play normally during swallow, and emits caught only after EAT_DURATION_MS
    const hostEatStartWait = host.waitFor(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'eating_started',
      'shared serpent eating_started on host'
    );
    const friendEatStartWait = friend.waitFor(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'eating_started',
      'shared serpent eating_started on friend'
    );
    const hostCatchEventWait = host.waitFor(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'caught',
      'shared serpent catch event on host',
      EAT_DURATION_MS + 3500
    );
    const friendCatchEventWait = friend.waitFor(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'caught',
      'shared serpent catch event on friend',
      EAT_DURATION_MS + 3500
    );

    // Place friend directly in water (-1.1 below surface) near serpent waypoint (0, -25) while host stays safe on spawn deck
    const friendInWaterInterval = setInterval(() => {
      friend.send('game:input_state', {
        x: 0,
        y: -1.1,
        z: -24,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 2,
        inWater: true,
        checkpointIndex: 0,
        escaped: false,
      });
    }, 45);

    const worldChaseWait = host.waitFor(
      (p) =>
        p.event === 'game:world' &&
        (p.payload as WorldSyncPayload).monsters.some((m) => m.alert === 'chasing' && m.targetUserId === friend.user.id),
      'shared serpent targets player in water'
    );

    await worldChaseWait;
    const [hostEatPkt, friendEatPkt] = await Promise.all([hostEatStartWait, friendEatStartWait]);
    const eatStartMs = Date.now();
    clearInterval(friendInWaterInterval);
    const hostEatEv = hostEatPkt.payload as GamePlayerEvent;
    const friendEatEv = friendEatPkt.payload as GamePlayerEvent;
    assert.equal(hostEatEv.userId, friend.user.id);
    assert.equal(friendEatEv.userId, friend.user.id);
    assert.ok(hostEatEv.monsterId, 'eating_started must identify monsterId');
    assert.equal(hostEatEv.x, friendEatEv.x);
    assert.equal(hostEatEv.z, friendEatEv.z);

    // Mid-swallow (~400ms): friend must STILL be alive=true, monster must broadcast eatingUserId + eatingProgress, and host must move normally
    await new Promise((r) => setTimeout(r, 400));
    const level1Spawn = LEVELS.find((l) => l.id === 1)!.spawn;
    const midSwallowHostX = level1Spawn[0] + 1.25;
    host.send('game:input_state', {
      x: midSwallowHostX,
      y: level1Spawn[1],
      z: level1Spawn[2],
      yaw: Math.PI,
      animState: 'walk',
      horizSpeed: 4,
      inWater: false,
      checkpointIndex: 0,
      escaped: false,
    });
    const midEatWorldPkt = await host.waitFor(
      (p) => {
        if (p.event !== 'game:world') return false;
        const w = p.payload as WorldSyncPayload;
        const hState = w.players.find((pl) => pl.userId === host.user.id);
        const fState = w.players.find((pl) => pl.userId === friend.user.id);
        const mState = w.monsters.find((m) => m.eatingUserId === friend.user.id);
        return Boolean(
          hState &&
            Math.abs(hState.x - midSwallowHostX) < 0.05 &&
            fState &&
            fState.alive === true &&
            mState &&
            typeof mState.eatingProgress === 'number' &&
            mState.eatingProgress > 0 &&
            mState.eatingProgress < 1
        );
      },
      'mid-swallow world sync shows victim alive, monster eatingProgress in (0,1), and host moving freely'
    );
    assert.ok(midEatWorldPkt);

    const [hostCatchPkt, friendCatchPkt] = await Promise.all([hostCatchEventWait, friendCatchEventWait]);
    const eatElapsedMs = Date.now() - eatStartMs;
    assert.ok(
      eatElapsedMs >= EAT_DURATION_MS - 180,
      `caught must not fire before eating finishes (elapsed=${eatElapsedMs}ms, expected>=${EAT_DURATION_MS - 180}ms)`
    );
    const hostCatchEv = hostCatchPkt.payload as GamePlayerEvent;
    const friendCatchEv = friendCatchPkt.payload as GamePlayerEvent;
    assert.equal(hostCatchEv.userId, friend.user.id);
    assert.equal(friendCatchEv.userId, friend.user.id);
    assert.ok(Number.isFinite(hostCatchEv.x) && Number.isFinite(hostCatchEv.y) && Number.isFinite(hostCatchEv.z), 'Caught event must include 3D coordinates');
    assert.equal(hostCatchEv.x, friendCatchEv.x);
    assert.equal(hostCatchEv.z, friendCatchEv.z);

    // Host returns room to lobby cleanly
    const backLobbyWait = friend.waitFor(
      (p) => p.event === 'room:state' && (p.payload as RoomState | null)?.status === 'lobby',
      'return to lobby'
    );
    host.send('room:return_lobby');
    await backLobbyWait;
    console.log('   PASS: 2 players start match, synchronized two-phase eating plays over EAT_DURATION_MS while teammate moves freely, then eliminates victim.');

    console.log('4. Testing Scenario 3 (Max 12 players), Scenario 4 (13th rejected), Scenario 5 (Leave opens slot)...');
    // Currently room has 2 players (host + friend). Add 10 more = 12 total!
    const roomCode = host.currentRoom!.code;
    const extraPlayers: TestClient[] = [];
    for (let i = 3; i <= 12; i++) {
      const p = await makeClient(`SquadMember_${i}`);
      extraPlayers.push(p);
      const joinRes = await p.http('/api/rooms/join', 'POST', { code: roomCode });
      assert.equal(joinRes.status, 200, `Player ${i} failed to join`);
    }

    // Wait briefly for room:state propagation
    await new Promise((r) => setTimeout(r, 40));
    const roomRes12 = await host.http('/api/auth/me');
    const active12 = roomRes12.data.activeRoom as RoomState;
    assert.equal(active12.players.length, 12, 'Room must hold exactly 12/12 players');

    // Scenario 4: 13th player must be rejected (HTTP + WS)
    const p13 = await makeClient('Player13');
    const reject13Http = await p13.http('/api/rooms/join', 'POST', { code: roomCode });
    assert.equal(reject13Http.status, 400);
    assert.match(String(reject13Http.data.error), /full \(12\/12/i);

    const reject13WsWait = p13.waitFor((p) => p.event === 'room:error', '13th WS join rejected');
    p13.send('room:join_code', { code: roomCode });
    const reject13Ws = await reject13WsWait;
    assert.match((reject13Ws.payload as { message: string }).message, /full \(12\/12/i);

    // Scenario 5: One player leaves -> slot opens -> Player 13 joins!
    const leaver = extraPlayers[0];
    const leaveWait = host.waitFor(
      (p) => p.event === 'room:state' && (p.payload as RoomState | null)?.players.length === 11,
      'slot opens to 11/12'
    );
    leaver.send('room:leave');
    await leaveWait;

    const join13AfterLeave = await p13.http('/api/rooms/join', 'POST', { code: roomCode });
    assert.equal(join13AfterLeave.status, 200);
    assert.equal((join13AfterLeave.data.room as RoomState).players.length, 12);
    console.log('   PASS: 12-player cap enforced, 13th rejected, leaving freed 1 slot for 13th player.');

    console.log('5. Testing Scenario 6: Host disconnect safely transfers ownership...');
    const oldHostId = host.user.id;
    const transferWait = friend.waitFor(
      (p) =>
        p.event === 'room:state' &&
        (p.payload as RoomState | null)?.hostId !== undefined &&
        (p.payload as RoomState | null)?.hostId !== oldHostId,
      'host ownership transfer'
    );
    host.close();
    const transferPkt = await transferWait;
    const afterTransfer = transferPkt.payload as RoomState;
    assert.equal(afterTransfer.players.length, 11);
    assert.notEqual(afterTransfer.hostId, oldHostId);
    const newHost = clients.find((c) => c.user.id === afterTransfer.hostId)!;
    assert.ok(newHost, 'New host must be one of the remaining connected players');
    assert.ok(afterTransfer.players.find((p) => p.userId === newHost.user.id)?.isHost === true);

    // Verify new host actually holds host privileges (can close room!)
    const roomClosedWait = friend.waitFor((p) => p.event === 'room:closed', 'new host closes room');
    newHost.send('room:close');
    await roomClosedWait;
    console.log('   PASS: Host disconnect immediately transferred ownership and privileges to connected player.');

    console.log('6. Testing Scenario 7: Simultaneous joins race condition never exceed 12...');
    const raceHost = await makeClient('RaceHost');
    const raceCreated = raceHost.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'race room created');
    raceHost.send('room:create', { levelId: 2 });
    await raceCreated;
    const raceCode = raceHost.currentRoom!.code;

    const racers: TestClient[] = [];
    for (let i = 1; i <= 20; i++) {
      racers.push(await makeClient(`Racer_${i}`));
    }

    const raceResults = await Promise.all(
      racers.map((r) => r.http('/api/rooms/join', 'POST', { code: raceCode }))
    );

    const successes = raceResults.filter((r) => r.status === 200);
    const failures = raceResults.filter((r) => r.status === 400);
    assert.equal(successes.length, 11, 'Only 11 racers may join a room that already has 1 host');
    assert.equal(failures.length, 9, 'Remaining 9 simultaneous joiners must be rejected');

    const finalRaceMe = await raceHost.http('/api/auth/me');
    const finalRaceRoom = finalRaceMe.data.activeRoom as RoomState;
    assert.equal(finalRaceRoom.players.length, 12, 'Room player count after concurrent burst must be exactly 12');
    console.log('   PASS: 20 simultaneous joins strictly capped at 12/12.');

    console.log('7. Testing Scenario 8: Huge Sector 55 with 12 players spawns wide apart & multi-serpent AI distributes targets...');
    const level55Wait = raceHost.waitFor(
      (p) => p.event === 'room:state' && (p.payload as RoomState | null)?.levelId === 55,
      'set sector 55'
    );
    raceHost.send('room:set_level', { levelId: 55 });
    await level55Wait;

    const hugeStartWait = raceHost.waitFor((p) => p.event === 'game:started', '12-player huge sector start');
    raceHost.send('room:start');
    const hugeStartPkt = await hugeStartWait;
    const hugePayload = hugeStartPkt.payload as {
      levelId: number;
      players: Array<{ userId: string; x: number; y: number; z: number }>;
      monsters: Array<{ id: string; x: number; z: number }>;
    };
    assert.equal(hugePayload.levelId, 55);
    assert.equal(hugePayload.players.length, 12);
    assert.ok(hugePayload.monsters.length >= 3, 'Huge Sector 55 must spawn multiple sea serpents');

    let minPairDist = Infinity;
    for (let i = 0; i < hugePayload.players.length; i++) {
      for (let j = i + 1; j < hugePayload.players.length; j++) {
        const a = hugePayload.players[i];
        const b = hugePayload.players[j];
        const d = Math.hypot(a.x - b.x, a.z - b.z);
        if (d < minPairDist) minPairDist = d;
      }
    }
    assert.ok(minPairDist >= 2.2, `12 players on huge start deck must be well spaced (minPairDist=${minPairDist.toFixed(2)})`);

    // Place 2 players in water near serpent 1 and serpent 2 waypoints -> confirm distinct target assignment
    const swimmer1 = raceHost;
    const swimmer2 = racers.find((r) => finalRaceRoom.players.some((p) => p.userId === r.user.id))!;
    const m1 = hugePayload.monsters[0];
    const m2 = hugePayload.monsters[1];
    const sendSwimmerStates = () => {
      swimmer1.send('game:input_state', {
        x: m1.x,
        y: -1.1,
        z: m1.z + 18,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 2,
        inWater: true,
        checkpointIndex: 0,
        escaped: false,
      });
      swimmer2.send('game:input_state', {
        x: m2.x,
        y: -1.1,
        z: m2.z + 18,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 2,
        inWater: true,
        checkpointIndex: 0,
        escaped: false,
      });
    };
    sendSwimmerStates();
    const swimTimer = setInterval(sendSwimmerStates, 45);

    const multiTargetSync = await raceHost.waitFor(
      (p) => {
        if (p.event !== 'game:world') return false;
        const w = p.payload as WorldSyncPayload;
        const targets = new Set(w.monsters.filter((m) => m.alert === 'chasing' && m.targetUserId).map((m) => m.targetUserId));
        return targets.size >= 2;
      },
      'multiple serpents target distinct swimmers'
    );
    clearInterval(swimTimer);
    assert.ok(multiTargetSync);
    console.log(`   PASS: 12 players spawned >= ${minPairDist.toFixed(2)}m apart and ${hugePayload.monsters.length} serpents hunted distinct swimmers.`);

    console.log('9. Testing Real-Time Voice Chat Backend (ICE config, room-scoped WebRTC signaling, voice state & privacy cleanup)...');
    // Unauthenticated ICE config must return 401
    const unauthIceRes = await fetch(`${baseUrl}/api/voice/ice-config`);
    assert.equal(unauthIceRes.status, 401, 'ICE config must require authentication');

    // Authenticated ICE config must return STUN servers
    const authIceRes = await fetch(`${baseUrl}/api/voice/ice-config`, {
      headers: { Authorization: `Bearer ${raceHost.token}` },
    });
    assert.equal(authIceRes.status, 200);
    const iceJson = (await authIceRes.json()) as { iceServers: Array<{ urls: string | string[] }> };
    assert.ok(Array.isArray(iceJson.iceServers) && iceJson.iceServers.length >= 1, 'ICE config must provide STUN servers');

    // Outsider outside the 12-player room cannot receive or inject signals into the room
    const outsider = await makeClient('VoiceOutsider');
    const peerWaitVoice = swimmer2.waitFor(
      (p) =>
        p.event === 'voice:peer_state' &&
        (p.payload as { userId: string; voice: { speaking: boolean; micEnabled: boolean } }).userId === raceHost.user.id &&
        (p.payload as { userId: string; voice: { speaking: boolean; micEnabled: boolean } }).voice.speaking === true,
      'teammate receives speaking voice state'
    );
    raceHost.send('voice:state', {
      voiceActive: true,
      micEnabled: true,
      selfMuted: false,
      speaking: true,
    });
    await peerWaitVoice;

    // Server invariant: micEnabled=false forces speaking=false even if client claims speaking=true
    const mutedInvariantWait = swimmer2.waitFor(
      (p) =>
        p.event === 'voice:peer_state' &&
        (p.payload as { userId: string; voice: { speaking: boolean; micEnabled: boolean } }).userId === raceHost.user.id &&
        (p.payload as { userId: string; voice: { speaking: boolean; micEnabled: boolean } }).voice.micEnabled === false &&
        (p.payload as { userId: string; voice: { speaking: boolean; micEnabled: boolean } }).voice.speaking === false,
      'server enforces speaking=false when mic is off'
    );
    raceHost.send('voice:state', {
      voiceActive: true,
      micEnabled: false,
      selfMuted: false,
      speaking: true,
    });
    await mutedInvariantWait;

    // WebRTC offer signaling relays between authorized peers in the same room
    const signalRelayWait = swimmer2.waitFor(
      (p) =>
        p.event === 'voice:signal' &&
        (p.payload as { fromUserId: string; kind: string }).fromUserId === raceHost.user.id &&
        (p.payload as { kind: string }).kind === 'offer',
      'WebRTC offer relayed inside room'
    );
    raceHost.send('voice:signal', {
      targetUserId: swimmer2.user.id,
      kind: 'offer',
      data: { type: 'offer', sdp: 'v=0\r\no=- 123 2 IN IP4 127.0.0.1\r\n' },
    });
    await signalRelayWait;

    // Cross-room signaling attempt from outsider is rejected cleanly
    const outsiderSignalErr = outsider.waitFor(
      (p) => p.event === 'room:error',
      'outsider voice signal rejected'
    );
    outsider.send('voice:signal', {
      targetUserId: raceHost.user.id,
      kind: 'offer',
      data: { type: 'offer', sdp: 'fake' },
    });
    await outsiderSignalErr;

    // Leaving room emits voice:peer_left to teammates
    const peerLeftWait = raceHost.waitFor(
      (p) =>
        p.event === 'voice:peer_left' &&
        (p.payload as { userId: string }).userId === swimmer2.user.id,
      'voice:peer_left emitted when teammate leaves room'
    );
    swimmer2.send('room:leave', {});
    await peerLeftWait;
    console.log('   PASS: Authenticated ICE config, 12-player voice state sync, same-room WebRTC signaling, outsider blocking & peer-left cleanup verified.');

    console.log('10. Testing mid-match WebSocket reconnect delivers auth:ok followed by game:started...');
    const reconnectTarget = racers.find(
      (r) => r.user.id !== swimmer2.user.id && finalRaceRoom.players.some((p) => p.userId === r.user.id)
    )!;
    const prevPacketCount = reconnectTarget.packets.length;
    reconnectTarget.close();
    await new Promise((r) => setTimeout(r, 25));
    const reconnectedStartWait = reconnectTarget.waitFor(
      (p) => p.event === 'game:started',
      'mid-match reconnect receives game:started'
    );
    await reconnectTarget.connectWs();
    await reconnectedStartWait;
    const reconnectSeq = reconnectTarget.packets.slice(prevPacketCount).map((p) => p.event);
    const authOkIdx = reconnectSeq.indexOf('auth:ok');
    const gameStartedIdx = reconnectSeq.indexOf('game:started');
    assert.ok(authOkIdx >= 0, 'Reconnect sequence must include auth:ok');
    assert.ok(gameStartedIdx > authOkIdx, 'game:started must arrive immediately after auth:ok on mid-match reconnect');
    const reconnectGameStarted = reconnectTarget.packets
      .slice(prevPacketCount)
      .find((p) => p.event === 'game:started')!.payload as { elapsedSec?: number };
    assert.equal(typeof reconnectGameStarted.elapsedSec, 'number', 'Reconnect game:started must include elapsedSec');
    assert.ok((reconnectGameStarted.elapsedSec ?? -1) >= 0, 'elapsedSec must be non-negative');
    console.log('   PASS: Mid-match WebSocket reconnect restored auth:ok -> game:started ordering with elapsedSec.');

    console.log('11. Testing Scenario 11: Rotating-platform multiplayer elapsedSec sync & accurate OBB water detection...');
    const rotHost = await makeClient('RotHost');
    const rotGuest = await makeClient('RotGuest');
    const rotRoomCreated = rotHost.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'rot room created');
    rotHost.send('room:create', { levelId: 3 });
    await rotRoomCreated;
    const rotJoinRes = await rotGuest.http('/api/rooms/join', 'POST', { code: rotHost.currentRoom!.code });
    assert.equal(rotJoinRes.status, 200);
    await new Promise((r) => setTimeout(r, 40));

    const rotStartWait = rotHost.waitFor((p) => p.event === 'game:started', 'rot game:started');
    rotHost.send('room:start');
    const rotStartPkt = await rotStartWait;
    assert.equal((rotStartPkt.payload as { elapsedSec?: number }).elapsedSec, 0, 'Initial game:started elapsedSec must be 0');

    // Sector 3 rotating bridge at [-2.37, 0.54, -4.45], size [2.99, 1.16, 6.78], axis 'y', waterY = 0
    // rotHost stands on center of rotating bridge (y = 0.54 + 0.58 = 1.12) -> must be inWater = false
    // rotGuest swims in water right next to the rotating bridge -> must be inWater = true (would falsely be false under old cylinder check!)
    const rotInputTimer = setInterval(() => {
      rotHost.send('game:input_state', {
        x: -2.37,
        y: 1.12,
        z: -4.45,
        yaw: 0,
        animState: 'idle',
        horizSpeed: 0,
        inWater: false,
        checkpointIndex: 0,
        escaped: false,
      });
      rotGuest.send('game:input_state', {
        x: -2.37 + 3.1,
        y: -0.45,
        z: -4.45,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 2,
        inWater: true,
        checkpointIndex: 0,
        escaped: false,
      });
    }, 40);

    const rotWorldSync = await rotHost.waitFor(
      (p) => {
        if (p.event !== 'game:world') return false;
        const w = p.payload as WorldSyncPayload;
        const hState = w.players.find((pl) => pl.userId === rotHost.user.id);
        const gState = w.players.find((pl) => pl.userId === rotGuest.user.id);
        return (
          typeof w.elapsedSec === 'number' &&
          w.elapsedSec > 0 &&
          Boolean(hState && !hState.inWater) &&
          Boolean(gState && gState.inWater)
        );
      },
      'rotating platform OBB water check & elapsedSec broadcast'
    );
    clearInterval(rotInputTimer);
    assert.ok(rotWorldSync);
    console.log('   PASS: Rotating-platform elapsedSec sync and exact OBB water/platform discrimination verified.');

    console.log('12. Testing Scenario 12: Sea serpent initial yaw/depth, obstacle patrol & dynamic emerged transition...');
    const level3Def = LEVELS.find((l) => l.id === 3)!;
    const initMonsters = (rotStartPkt.payload as {
      monsters: Array<{ id: string; x: number; y: number; z: number; yaw: number; emerged: boolean }>;
    }).monsters;
    assert.ok(initMonsters.length >= 1, 'Level 3 must spawn at least one serpent');
    const firstMc = level3Def.monsters[0];
    const expectedYaw = Math.atan2(
      (firstMc.waypoints[1]?.[0] ?? firstMc.waypoints[0][0]) - firstMc.waypoints[0][0],
      (firstMc.waypoints[1]?.[1] ?? firstMc.waypoints[0][1]) - firstMc.waypoints[0][1]
    ) || 0;
    assert.ok(
      Math.abs(initMonsters[0].y - (level3Def.bounds.waterY - 0.14)) < 0.02,
      `Serpent must spawn at calm waterY - 0.14 (got ${initMonsters[0].y})`
    );
    assert.ok(
      Math.abs(initMonsters[0].yaw - expectedYaw) < 0.02,
      `Serpent initial yaw must face waypoint[1] (got ${initMonsters[0].yaw}, expected ${expectedYaw})`
    );
    assert.equal(initMonsters[0].emerged, false);

    // Create a fresh room on Level 1 and place swimmer ~25m laterally from monster[0] in open water
    const sHost = await makeClient('SerpentHost');
    const sGuest = await makeClient('SerpentGuest');
    const sCreated = sHost.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'serpent room created');
    sHost.send('room:create', { levelId: 1 });
    await sCreated;
    const sJoinRes = await sGuest.http('/api/rooms/join', 'POST', { code: sHost.currentRoom!.code });
    assert.equal(sJoinRes.status, 200);
    await new Promise((r) => setTimeout(r, 30));
    const sStartWait = sHost.waitFor((p) => p.event === 'game:started', 'serpent game:started');
    sHost.send('room:start');
    const sStartPkt = await sStartWait;
    const level1Def = LEVELS.find((l) => l.id === 1)!;
    const sInitM = (sStartPkt.payload as { monsters: Array<{ x: number; z: number }> }).monsters[0];
    const farX = sInitM.x > 0 ? level1Def.bounds.minX + 4 : level1Def.bounds.maxX - 4;
    const farZ = sInitM.z;

    let sawUnemergedChase = false;
    let sawEmergedChase = false;
    const sInputTimer = setInterval(() => {
      sHost.send('game:input_state', {
        x: level1Def.spawn[0],
        y: level1Def.spawn[1],
        z: level1Def.spawn[2],
        yaw: Math.PI,
        animState: 'idle',
        horizSpeed: 0,
        inWater: false,
        checkpointIndex: 0,
        escaped: false,
      });
      sGuest.send('game:input_state', {
        x: farX,
        y: level1Def.bounds.waterY - 0.5,
        z: farZ,
        yaw: 0,
        animState: 'swim',
        horizSpeed: 2,
        inWater: true,
        checkpointIndex: 0,
        escaped: false,
      });
    }, 35);

    await sHost.waitFor(
      (p) => {
        if (p.event !== 'game:world') return false;
        const w = p.payload as WorldSyncPayload;
        const m = w.monsters[0];
        assert.ok(Number.isFinite(m.x) && Number.isFinite(m.y) && Number.isFinite(m.z), 'Monster coords finite');
        assert.ok(
          m.x >= level1Def.bounds.minX && m.x <= level1Def.bounds.maxX &&
          m.z >= level1Def.bounds.minZ && m.z <= level1Def.bounds.maxZ,
          'Monster stays inside arena bounds'
        );
        if (m.alert === 'chasing' && !m.emerged) sawUnemergedChase = true;
        if (m.alert === 'chasing' && m.emerged) sawEmergedChase = true;
        return sawUnemergedChase && sawEmergedChase;
      },
      'serpent starts chase un-emerged at long distance then emerges < 10.5m'
    );
    clearInterval(sInputTimer);
    assert.ok(sawUnemergedChase, 'Serpent should begin long-range chase un-emerged');
    assert.ok(sawEmergedChase, 'Serpent should dynamically emerge once within 10.5m');
    console.log('   PASS: Sea serpent spawn yaw/depth, bounds safety & dynamic emerged transition verified.');

    console.log('13. Testing Scenario 13: Push targeting physics (self, distance, facing cone, wall & platform obstruction)...');
    const wallTestDef = LEVELS.find((l) => l.id === 1)!;
    // Cannot push self
    assert.equal(
      selectPushTarget(wallTestDef, { userId: 'a', x: 0, y: 1.2, z: 0, yaw: 0 }, [{ userId: 'a', x: 0, y: 1.2, z: 1.0 }], 0),
      null,
      'Player must never be able to push self'
    );
    // Out-of-range rejected (> 2.6m)
    assert.equal(
      selectPushTarget(wallTestDef, { userId: 'a', x: 0, y: 1.2, z: 0, yaw: 0 }, [{ userId: 'b', x: 0, y: 1.2, z: 3.1 }], 0),
      null,
      'Out-of-range target must be rejected'
    );
    // Target behind attacker rejected (facing +Z with yaw=0, target at -Z)
    assert.equal(
      selectPushTarget(wallTestDef, { userId: 'a', x: 0, y: 1.2, z: 0, yaw: 0 }, [{ userId: 'b', x: 0, y: 1.2, z: -1.5 }], 0),
      null,
      'Target behind attacker must be rejected'
    );
    // Valid target in front on open deck
    const [sx, sy, sz] = wallTestDef.spawn;
    const validSel = selectPushTarget(
      wallTestDef,
      { userId: 'a', x: sx, y: sy, z: sz, yaw: Math.PI },
      [{ userId: 'b', x: sx + 0.2, y: sy, z: sz - 1.4 }],
      0
    );
    assert.ok(validSel, 'Valid nearby teammate in front must be selectable');
    assert.equal(validSel.userId, 'b');
    assert.ok(validSel.dirZ < -0.7, `Push direction should point toward -Z (got dirZ=${validSel.dirZ})`);

    // Wall / platform obstruction between two positions
    const firstWall = wallTestDef.walls[0];
    if (firstWall) {
      const [wx, wy, wz] = firstWall.position;
      const [ww, , wd] = firstWall.size;
      const blocked = isPushPathObstructed(
        wallTestDef,
        wx,
        wy - 0.5,
        wz - wd * 0.5 - 0.5,
        wx,
        wy - 0.5,
        wz + wd * 0.5 + 0.5,
        0
      );
      assert.equal(blocked || ww <= 0, true, 'Push path through solid wall must be obstructed');
    }
    console.log('   PASS: Push physics rules (self, distance, facing cone, obstruction) verified.');

    console.log('14. Testing Scenario 14: Authoritative Friends-Mode Push sync, cooldown, recovery immunity & eligibility...');
    const pHost = await makeClient('PushHost');
    const pFriend = await makeClient('PushFriend');
    const pHelper = await makeClient('PushHelper');
    const pCreated = pHost.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'push room created');
    pHost.send('room:create', { levelId: 3 });
    await pCreated;
    assert.equal((await pFriend.http('/api/rooms/join', 'POST', { code: pHost.currentRoom!.code })).status, 200);
    assert.equal((await pHelper.http('/api/rooms/join', 'POST', { code: pHost.currentRoom!.code })).status, 200);
    await new Promise((r) => setTimeout(r, 40));

    const pStarted = pHost.waitFor((p) => p.event === 'game:started', 'push match started');
    pHost.send('room:start');
    await pStarted;

    // Place all 3 players on Sector 3 rotating bridge at [-2.37, 1.12, -4.45]
    pHost.send('game:input_state', {
      x: -2.37,
      y: 1.12,
      z: -3.8,
      yaw: Math.PI, // facing -Z toward pFriend
      animState: 'idle',
      horizSpeed: 0,
      inWater: false,
      checkpointIndex: 0,
      escaped: false,
    });
    pFriend.send('game:input_state', {
      x: -2.37,
      y: 1.12,
      z: -5.1,
      yaw: 0,
      animState: 'idle',
      horizSpeed: 0,
      inWater: false,
      checkpointIndex: 0,
      escaped: false,
    });
    pHelper.send('game:input_state', {
      x: -1.9,
      y: 1.12,
      z: -3.9,
      yaw: Math.PI,
      animState: 'idle',
      horizSpeed: 0,
      inWater: false,
      checkpointIndex: 0,
      escaped: false,
    });
    await new Promise((r) => setTimeout(r, 60));

    // Host pushes Friend -> all 3 clients must receive the exact pushed event
    const hostPushEv = pHost.waitFor(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'pushed',
      'host receives pushed event'
    );
    const friendPushEv = pFriend.waitFor(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'pushed',
      'target receives pushed event'
    );
    const helperPushEv = pHelper.waitFor(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'pushed',
      'third teammate receives pushed event'
    );
    pHost.send('game:push', { yaw: Math.PI, x: -2.37, y: 1.12, z: -3.8 });
    const [evH, evF, evHl] = await Promise.all([hostPushEv, friendPushEv, helperPushEv]);
    const pushPayload = evF.payload as GamePlayerEvent;
    assert.equal(pushPayload.userId, pFriend.user.id);
    assert.equal(pushPayload.attackerUserId, pHost.user.id);
    assert.equal(pushPayload.force, PUSH_FORCE_HORIZ);
    assert.equal(pushPayload.liftY, PUSH_FORCE_LIFT_Y);
    assert.ok((pushPayload.dirZ ?? 0) < -0.7, 'Push event dirZ should send target toward -Z');
    assert.deepEqual(evH.payload, evF.payload);
    assert.deepEqual(evHl.payload, evF.payload);

    // Spam guard: immediate second push from pHost must be ignored by cooldown, and pHelper cannot instant chain-push pFriend during recovery immunity
    const prevFriendPushCount = pFriend.packets.filter(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'pushed'
    ).length;
    pHost.send('game:push', { yaw: Math.PI, x: -2.37, y: 1.12, z: -3.8 });
    pHelper.send('game:push', { yaw: Math.PI, x: -1.9, y: 1.12, z: -3.9 });
    await new Promise((r) => setTimeout(r, 90));
    const afterSpamCount = pFriend.packets.filter(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'pushed'
    ).length;
    assert.equal(afterSpamCount, prevFriendPushCount, 'Cooldown & recovery immunity must block rapid / chained pushes');

    // Player waiting at Sector 3 portal holds at portal for teammates, is immune to pushing, and team advances together once all reach portal
    const level3 = LEVELS.find((l) => l.id === 3)!;
    await new Promise((r) => setTimeout(r, Math.max(PUSH_COOLDOWN_MS, PUSH_RECOVERY_MS) + 60));
    const friendEscapedWait = pHost.waitFor(
      (p) =>
        p.event === 'game:player_event' &&
        (p.payload as GamePlayerEvent).kind === 'escaped' &&
        (p.payload as GamePlayerEvent).userId === pFriend.user.id,
      'friend waits at level 3 portal'
    );
    pFriend.send('game:input_state', {
      levelId: 3,
      x: level3.exit.position[0],
      y: level3.exit.position[1],
      z: level3.exit.position[2],
      yaw: 0,
      animState: 'idle',
      horizSpeed: 0,
      inWater: false,
      checkpointIndex: 0,
      escaped: true,
    });
    await friendEscapedWait;
    assert.equal(
      pFriend.packets.filter((p) => p.event === 'game:level_changed').length,
      0,
      'Solo portal arrival must NOT advance before remaining teammates arrive'
    );

    // Waiting player at portal cannot be pushed by host
    pHost.send('game:input_state', {
      levelId: 3,
      x: level3.exit.position[0],
      y: level3.exit.position[1],
      z: level3.exit.position[2] + 1.0,
      yaw: Math.PI,
      animState: 'idle',
      horizSpeed: 0,
      inWater: false,
      checkpointIndex: 0,
      escaped: false,
    });
    await new Promise((r) => setTimeout(r, 40));
    pHost.send('game:push', {
      yaw: Math.PI,
      x: level3.exit.position[0],
      y: level3.exit.position[1],
      z: level3.exit.position[2] + 1.0,
    });
    await new Promise((r) => setTimeout(r, 80));
    const afterPortalPushCount = pFriend.packets.filter(
      (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'pushed'
    ).length;
    assert.equal(afterPortalPushCount, prevFriendPushCount, 'Players waiting at exit portal must not be pushable');

    // Remaining two teammates reach portal -> all 3 transition to Level 4 together in the same room
    const allLv4Waits = Promise.all([
      pHost.waitFor(
        (p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 4,
        'host advances to level 4 with team'
      ),
      pFriend.waitFor(
        (p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 4,
        'friend advances to level 4 with team'
      ),
      pHelper.waitFor(
        (p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 4,
        'helper advances to level 4 with team'
      ),
    ]);
    pHost.send('game:input_state', {
      levelId: 3,
      x: level3.exit.position[0],
      y: level3.exit.position[1],
      z: level3.exit.position[2],
      yaw: 0,
      animState: 'idle',
      horizSpeed: 0,
      inWater: false,
      checkpointIndex: 0,
      escaped: true,
    });
    pHelper.send('game:input_state', {
      levelId: 3,
      x: level3.exit.position[0],
      y: level3.exit.position[1],
      z: level3.exit.position[2],
      yaw: 0,
      animState: 'idle',
      horizSpeed: 0,
      inWater: false,
      checkpointIndex: 0,
      escaped: true,
    });
    const [hLv4, fLv4, hlLv4] = await allLv4Waits;
    assert.equal((hLv4.payload as GameLevelChangedPayload).prevLevelId, 3);
    assert.equal((fLv4.payload as GameLevelChangedPayload).levelId, 4);
    assert.equal((hlLv4.payload as GameLevelChangedPayload).levelId, 4);
    assert.equal(pHost.currentRoom?.levelId, 4);
    assert.equal(pFriend.currentRoom?.id, pHost.currentRoom?.id);
    assert.equal(pHelper.currentRoom?.id, pHost.currentRoom?.id);
    console.log('   PASS: Authoritative Friends-Mode Push sync, cooldown, recovery immunity, portal immunity & team progression verified.');

    console.log('15. Testing Scenario 15: Spin-free sea serpent waypoint navigation (Friends Mode & Level 1 pockets)...');
    {
      const {
        resolvePatrolTarget,
        computeSerpentSteerAngle,
        updateSerpentProgressAndYaw,
        shortestAngleDiffNav,
      } = await import('../src/game/serpentNav.ts');
      const { LevelSpatialGrid } = await import('../src/game/spatialGrid.ts');

      // Edge case A: zero-length direction vector must damp drift & keep yaw steady without spinning
      const navZero = {
        x: 5,
        z: -20,
        yaw: 1.23,
        velX: 0.1,
        velZ: -0.1,
        wpIndex: 0,
        waypoints: [[5, -20]] as [number, number][],
        avoidDir: 1 as const,
        avoidTimer: 0,
        stuckTimer: 0,
        navBestDist: Infinity,
        navStallTimer: 0,
        navAnchorX: 5,
        navAnchorZ: -20,
        navAnchorTimer: 0,
        navTurnAccum: 0,
      };
      const tgtZero = resolvePatrolTarget(navZero, 0.88, () => false);
      assert.equal(tgtZero.hasTarget, false);
      const steerZero = computeSerpentSteerAngle(navZero, tgtZero.tx, tgtZero.tz, 0.88, false, () => false);
      assert.equal(steerZero.valid, false);
      for (let k = 0; k < 30; k++) {
        updateSerpentProgressAndYaw(navZero, navZero.x, navZero.z, navZero.yaw, 3.1, 1 / 30, false, false, () => false);
      }
      assert.equal(navZero.yaw, 1.23, 'Zero-length target must preserve yaw without spinning');

      // Edge case B: already-reached & blocked nearby waypoints skip cleanly to next valid waypoint
      const navSkip = {
        x: 0,
        z: 0,
        yaw: 0,
        velX: 0,
        velZ: 0,
        wpIndex: 0,
        waypoints: [
          [0.2, 0.2],
          [1.0, -0.5],
          [4, -3],
          [30, -30],
        ] as [number, number][],
        avoidDir: 1 as const,
        avoidTimer: 0,
        stuckTimer: 0,
        navBestDist: Infinity,
        navStallTimer: 0,
        navAnchorX: 0,
        navAnchorZ: 0,
        navAnchorTimer: 0,
        navTurnAccum: 0,
      };
      const tgtSkip = resolvePatrolTarget(navSkip, 0.88, (x, z) => Math.hypot(x - 4, z + 3) < 1.5);
      assert.equal(navSkip.wpIndex, 3);
      assert.equal(tgtSkip.hasTarget, true);

      // Edge case C: Exact Level 1 m2 bug pocket (s1_m1_p2 between [-20.48,-64] and [15.36,-66.56])
      const lv1 = LEVELS.find((l) => l.id === 1)!;
      const lv1Grid = new LevelSpatialGrid(lv1);
      const lv1M2 = lv1.monsters[1];
      const m2Scale = lv1M2.scale || 1;
      const sp: number[] = [];
      const sw: number[] = [];
      const blockedLv1 = (x: number, z: number, rad: number) => {
        const b = lv1.bounds;
        if (x - rad < b.minX || x + rad > b.maxX || z - rad < b.minZ || z + rad > b.maxZ) return true;
        const rSq = rad * rad;
        lv1Grid.queryRadius(x, z, rad + 0.5, sp, sw);
        for (const pi of sp) {
          const p = lv1.platforms[pi];
          const [px, py, pz] = p.position;
          const [pw, ph, pd] = p.size;
          if (b.waterY + 4.2 * m2Scale <= py - ph * 0.5 + 0.05 || b.waterY - 1.5 >= py + ph * 0.5 - 0.05) continue;
          const dx = x - Math.max(px - pw * 0.5, Math.min(x, px + pw * 0.5));
          const dz = z - Math.max(pz - pd * 0.5, Math.min(z, pz + pd * 0.5));
          if (dx * dx + dz * dz < rSq) return true;
        }
        return false;
      };
      const navM2 = {
        x: lv1M2.waypoints[0][0],
        z: lv1M2.waypoints[0][1],
        yaw: Math.atan2(lv1M2.waypoints[1][0] - lv1M2.waypoints[0][0], lv1M2.waypoints[1][1] - lv1M2.waypoints[0][1]),
        velX: 0,
        velZ: 0,
        wpIndex: 1,
        waypoints: lv1M2.waypoints,
        avoidDir: -1 as const,
        avoidTimer: 0,
        stuckTimer: 0,
        navBestDist: Infinity,
        navStallTimer: 0,
        navAnchorX: lv1M2.waypoints[0][0],
        navAnchorZ: lv1M2.waypoints[0][1],
        navAnchorTimer: 0,
        navTurnAccum: 0,
      };
      let reachedWp2 = false;
      let maxM2YawRate = 0;
      let wYaw = 0;
      let wT = 0;
      const dtSim = 1 / 30;
      for (let s = 0; s < 30 * 30; s++) {
        const tgt = resolvePatrolTarget(navM2, 0.88 * m2Scale, blockedLv1);
        if (navM2.wpIndex >= 2) reachedWp2 = true;
        const prevY = navM2.yaw;
        const st = tgt.hasTarget
          ? computeSerpentSteerAngle(navM2, tgt.tx, tgt.tz, 0.88 * m2Scale, false, blockedLv1)
          : { valid: false, steerAngle: navM2.yaw };
        if (st.valid) {
          const blend = Math.min(1, 8.5 * dtSim);
          navM2.velX += (Math.sin(st.steerAngle) * lv1M2.patrolSpeed - navM2.velX) * blend;
          navM2.velZ += (Math.cos(st.steerAngle) * lv1M2.patrolSpeed - navM2.velZ) * blend;
        }
        const px = navM2.x;
        const pz = navM2.z;
        navM2.x += navM2.velX * dtSim;
        navM2.z += navM2.velZ * dtSim;
        updateSerpentProgressAndYaw(navM2, px, pz, prevY, lv1M2.patrolSpeed, dtSim, false, st.valid, blockedLv1);
        wYaw += Math.abs(shortestAngleDiffNav(prevY, navM2.yaw));
        wT += dtSim;
        if (wT >= 2.0) {
          maxM2YawRate = Math.max(maxM2YawRate, wYaw / wT);
          wYaw = 0;
          wT = 0;
        }
      }
      assert.ok(reachedWp2, 'Lv1 m2 must navigate cleanly past s1_m1_p2 to subsequent waypoints');
      assert.ok(maxM2YawRate < 2.8, `Lv1 m2 must not spin (maxYawRate=${maxM2YawRate.toFixed(2)})`);

      // Live Friends-Mode server patrol check on Level 1
      const navHost = await makeClient('NavHost');
      const navGuest = await makeClient('NavGuest');
      const navCreated = navHost.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'nav room created');
      navHost.send('room:create', { levelId: 1 });
      await navCreated;
      const navJoin = await navGuest.http('/api/rooms/join', 'POST', { code: navHost.currentRoom!.code });
      assert.equal(navJoin.status, 200);
      await new Promise((r) => setTimeout(r, 30));
      const navStart = navHost.waitFor((p) => p.event === 'game:started', 'nav game:started');
      navHost.send('room:start');
      await navStart;

      let prevServerYaw: number | null = null;
      let serverMaxDelta = 0;
      let serverTicks = 0;
      let m2MovedDist = 0;
      let m2StartPos: [number, number] | null = null;
      await navHost.waitFor((p) => {
        if (p.event !== 'game:world') return false;
        const w = p.payload as WorldSyncPayload;
        const m2 = w.monsters[1];
        if (!m2StartPos) m2StartPos = [m2.x, m2.z];
        m2MovedDist = Math.hypot(m2.x - m2StartPos[0], m2.z - m2StartPos[1]);
        if (prevServerYaw !== null) {
          const d = Math.abs(shortestAngleDiffNav(prevServerYaw, m2.yaw));
          if (d > serverMaxDelta) serverMaxDelta = d;
        }
        prevServerYaw = m2.yaw;
        serverTicks++;
        return serverTicks >= 12;
      }, 'server serpent patrol ticks');
      assert.ok(m2MovedDist > 1.0, `Friends-mode serpent m2 must swim smoothly (moved ${m2MovedDist.toFixed(2)}m)`);
      assert.ok(serverMaxDelta < 0.8, `Friends-mode serpent yaw must change smoothly (maxStep=${serverMaxDelta.toFixed(2)})`);

      // Verify characterId ('female') syncs through authoritative server WorldSyncPayload
      const guestSpawn = navGuest.currentRoom!.players.find((p) => p.userId === navGuest.user.id);
      assert.ok(guestSpawn);
      navGuest.send('game:input_state', {
        x: 0,
        y: 1.2,
        z: 3,
        yaw: Math.PI,
        animState: 'idle',
        horizSpeed: 0,
        inWater: false,
        checkpointIndex: 0,
        escaped: false,
        characterId: 'female',
      });
      await navHost.waitFor((p) => {
        if (p.event !== 'game:world') return false;
        const w = p.payload as WorldSyncPayload;
        const g = w.players.find((pl) => pl.userId === navGuest.user.id);
        return g?.characterId === 'female';
      }, 'guest female characterId synced in world snapshot');
      console.log('   PASS: Spin-free serpent patrol, zero-length yaw hold, blocked waypoint skip, character sync & Friends-Mode sync verified.');
    }

    console.log('16. Testing Scenario 16: Lazy level catalog equivalence & swept spatial grid coverage...');
    {
      const { LevelSpatialGrid } = await import('../src/game/spatialGrid.ts');
      assert.equal(TOTAL_LEVELS, 100);
      assert.equal(isValidLevelId(1), true);
      assert.equal(isValidLevelId(100), true);
      assert.equal(isValidLevelId(0), false);
      assert.equal(isValidLevelId(101), false);

      for (let id = 1; id <= TOTAL_LEVELS; id++) {
        const sum = getLevelSummary(id);
        const full = getLevelById(id);
        assert.ok(sum, `getLevelSummary(${id}) must exist`);
        assert.ok(full, `getLevelById(${id}) must exist`);
        assert.equal(sum.id, full.id);
        assert.equal(sum.code, full.code);
        assert.equal(sum.name, full.name);
        assert.equal(sum.subtitle, full.subtitle);
        assert.equal(sum.difficulty, full.difficulty);
        assert.equal(sum.timeLimit, full.timeLimit);
        assert.deepEqual(sum.parTimes, full.parTimes);
      }

      // Synthetic level covering static, moving, Y-rotating, and X/Z/XY/ZY tilted rotating platforms
      const synthDef = {
        bounds: { minX: -60, maxX: 60, minZ: -60, maxZ: 60, floorY: -4, waterY: 0, ceilingY: 14 },
        platforms: [
          { id: 'static_p', kind: 'static' as const, position: [-40, 1, -40] as [number, number, number], size: [4, 1, 4] as [number, number, number] },
          { id: 'moving_p', kind: 'moving' as const, position: [0, 1, 0] as [number, number, number], size: [4, 1, 4] as [number, number, number], moveOffset: [22, 0, -18] as [number, number, number], moveSpeed: 1 },
          { id: 'rot_y', kind: 'static' as const, position: [25, 1, 25] as [number, number, number], size: [12, 1, 2] as [number, number, number], rotationSpeed: 0.9, rotationAxis: 'y' as const },
          { id: 'rot_x', kind: 'static' as const, position: [-25, 2, 25] as [number, number, number], size: [2, 10, 2] as [number, number, number], rotationSpeed: 0.8, rotationAxis: 'x' as const },
          { id: 'rot_z', kind: 'static' as const, position: [-25, 2, -5] as [number, number, number], size: [10, 6, 2] as [number, number, number], rotationSpeed: 0.8, rotationAxis: 'z' as const },
          { id: 'rot_xy', kind: 'static' as const, position: [35, 2, -25] as [number, number, number], size: [8, 8, 2] as [number, number, number], rotationSpeed: 0.7, rotationAxis: 'xy' as const },
          { id: 'rot_zy', kind: 'static' as const, position: [0, 2, -45] as [number, number, number], size: [2, 8, 8] as [number, number, number], rotationSpeed: 0.7, rotationAxis: 'zy' as const },
        ],
        walls: [],
      };
      const grid = new LevelSpatialGrid(synthDef, 12);
      assert.equal(grid.dynamicPlatIndices.length, 0, 'All platforms must be indexed in swept spatial buckets');

      const foundPlats: number[] = [];
      // Moving platform start & end offset corners
      grid.queryBox(-1.5, 1.5, -1.5, 1.5, foundPlats, null);
      assert.ok(foundPlats.includes(1), 'Moving platform must be found at origin');
      grid.queryBox(21, 23, -19, -17, foundPlats, null);
      assert.ok(foundPlats.includes(1), 'Moving platform must be found at end of moveOffset');

      // Y-rotating long beam rotated 90 degrees (reaches Z = 25 +- 6)
      grid.queryBox(24.5, 25.5, 30.2, 31.0, foundPlats, null);
      assert.ok(foundPlats.includes(2), 'Y-rotating platform must be found across rotated sweep radius');

      // Tilted X/Z/XY/ZY platforms across their swept bounding spheres
      grid.queryBox(-25.5, -24.5, 29.5, 30.1, foundPlats, null);
      assert.ok(foundPlats.includes(3), 'X-tilted platform must be found across 3D sweep');
      grid.queryBox(-30.5, -29.8, -5.5, -4.5, foundPlats, null);
      assert.ok(foundPlats.includes(4), 'Z-tilted platform must be found across 3D sweep');
      grid.queryBox(39.8, 40.5, -25.5, -24.5, foundPlats, null);
      assert.ok(foundPlats.includes(5), 'XY-tilted platform must be found across 3D sweep');
      grid.queryBox(4.8, 5.5, -45.5, -44.5, foundPlats, null);
      assert.ok(foundPlats.includes(6), 'ZY-tilted platform must be found across 3D sweep');
      console.log('   PASS: All 100 level summaries match full definitions & swept spatial grid covers all moving/rotating envelopes.');
    }

    console.log('17. Testing Scenario 17: Game version compatibility enforcement across REST, WS & invites...');
    {
      const verHost = await makeClient('VerHost');
      const verOld = await makeClient('VerOld');
      const verOk = await makeClient('VerOk');

      // Befriend verHost and verOld for invite version mismatch test
      await verHost.http('/api/friends/request', 'POST', { targetUserId: verOld.user.id });
      const oldReqs = await verOld.http('/api/friends');
      const vReqId = (oldReqs.data.incomingRequests as Array<{ id: string }>)[0].id;
      await verOld.http('/api/friends/respond', 'POST', { requestId: vReqId, action: 'accept' });

      // Mismatched REST create fails with clear update error
      const badRestCreate = await verOld.http('/api/rooms', 'POST', {
        levelId: 1,
        gameVersion: '0.0.1-old',
      });
      assert.equal(badRestCreate.status, 400);
      assert.match(String(badRestCreate.data.error), /update your game/i);

      // Mismatched WS create fails with clear update error
      const badWsCreateWait = verOld.waitFor((p) => p.event === 'room:error', 'ws version mismatch on create');
      verOld.send('room:create', { levelId: 1, gameVersion: '0.0.1-old' });
      const badWsCreate = await badWsCreateWait;
      assert.match((badWsCreate.payload as { message: string }).message, /update your game/i);

      // Valid host creates room
      const vCreatedWait = verHost.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'verHost room created');
      verHost.send('room:create', { levelId: 1, gameVersion: GAME_VERSION });
      await vCreatedWait;
      const vCode = verHost.currentRoom!.code;
      assert.equal(verHost.currentRoom!.gameVersion, GAME_VERSION);

      // Mismatched REST join fails
      const badRestJoin = await verOld.http('/api/rooms/join', 'POST', {
        code: vCode,
        gameVersion: '0.9.0',
      });
      assert.equal(badRestJoin.status, 400);
      assert.match(String(badRestJoin.data.error), /version|update your game/i);

      // Mismatched WS join fails
      const badWsJoinWait = verOld.waitFor((p) => p.event === 'room:error', 'ws version mismatch on join');
      verOld.send('room:join_code', { code: vCode, gameVersion: '2.0.0-beta' });
      const badWsJoin = await badWsJoinWait;
      assert.match((badWsJoin.payload as { message: string }).message, /version|update your game/i);

      // Mismatched invite acceptance fails cleanly and keeps player out of room
      const vInvWait = verOld.waitFor((p) => p.event === 'invite:received', 'invite received for version test');
      verHost.send('room:invite', { friendUserId: verOld.user.id });
      const vInvPkt = await vInvWait;
      const vInv = vInvPkt.payload as RoomInviteInfo;
      const badInvWait = verOld.waitFor((p) => p.event === 'room:error', 'invite version mismatch rejected');
      verOld.send('room:invite_respond', { inviteId: vInv.id, accept: true, gameVersion: '0.8.5' });
      const badInvErr = await badInvWait;
      assert.match((badInvErr.payload as { message: string }).message, /version|update your game/i);
      assert.equal(verHost.currentRoom!.players.length, 1);

      // Matching version joins cleanly
      const okJoinWait = verOk.waitFor(
        (p) => p.event === 'room:state' && (p.payload as RoomState | null)?.players.length === 2,
        'matching version joins'
      );
      verOk.send('room:join_code', { code: vCode, gameVersion: GAME_VERSION });
      await okJoinWait;
      console.log('   PASS: Exact game version compatibility enforced across WS, REST, and room invites.');
    }

    console.log('18. Testing Scenario 18: 4-player Team-Based Portal Wait, staggered arrival, teammate death while waiting & duplicate transition guards...');
    {
      const p1 = await makeClient('TeamP1');
      const p2 = await makeClient('TeamP2');
      const p3 = await makeClient('TeamP3');
      const p4 = await makeClient('TeamP4');
      const crWait = p1.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'team room created');
      p1.send('room:create', { levelId: 1, gameVersion: GAME_VERSION });
      await crWait;
      const roomCode = p1.currentRoom!.code;
      assert.equal((await p2.http('/api/rooms/join', 'POST', { code: roomCode, gameVersion: GAME_VERSION })).status, 200);
      assert.equal((await p3.http('/api/rooms/join', 'POST', { code: roomCode, gameVersion: GAME_VERSION })).status, 200);
      assert.equal((await p4.http('/api/rooms/join', 'POST', { code: roomCode, gameVersion: GAME_VERSION })).status, 200);
      await new Promise((r) => setTimeout(r, 40));

      await Promise.all([
        (async () => {
          const waits = [p1, p2, p3, p4].map((c, idx) => c.waitFor((p) => p.event === 'game:started', `p${idx + 1} started`));
          p1.send('room:start');
          await Promise.all(waits);
        })(),
      ]);

      assert.equal(p1.currentRoom?.levelId, 1);
      const lv1Def = getLevelById(1);
      const lv2Def = getLevelById(2);
      const sendAtExit = (c: TestClient, levelDef = lv1Def) => {
        c.send('game:input_state', {
          levelId: levelDef.id,
          x: levelDef.exit.position[0],
          y: levelDef.exit.position[1],
          z: levelDef.exit.position[2],
          yaw: 0,
          animState: 'run',
          horizSpeed: 6,
          inWater: false,
          checkpointIndex: 1,
          escaped: false,
        });
      };

      // Mandatory Test 1 & 2: P1 reaches portal first -> waits at portal; P2 & P3 arrive at staggered times
      sendAtExit(p1);
      await new Promise((r) => setTimeout(r, 40));
      p1.send('game:portal_enter', { levelId: 1 });
      const world1of4 = await p1.waitFor(
        (p) =>
          p.event === 'game:world' &&
          (p.payload as WorldSyncPayload).portalReadyCount === 1 &&
          (p.payload as WorldSyncPayload).portalTotalCount === 4,
        'world reports 1/4 ready at portal'
      );
      assert.equal((world1of4.payload as WorldSyncPayload).levelId, 1);
      assert.equal(
        p1.packets.filter((p) => p.event === 'game:level_changed').length,
        0,
        'P1 must wait at Level 1 portal while P2..P4 are still playing'
      );

      // Duplicate portal_enter spam from P1 must not increment count or cause duplicate transitions later
      p1.send('game:portal_enter', { levelId: 1 });
      sendAtExit(p2);
      await new Promise((r) => setTimeout(r, 40));
      p2.send('game:portal_enter', { levelId: 1 });
      sendAtExit(p3);
      await new Promise((r) => setTimeout(r, 40));
      p3.send('game:portal_enter', { levelId: 1 });

      const world3of4 = await p2.waitFor(
        (p) =>
          p.event === 'game:world' &&
          (p.payload as WorldSyncPayload).portalReadyCount === 3 &&
          (p.payload as WorldSyncPayload).portalTotalCount === 4,
        'world reports 3/4 ready at portal'
      );
      assert.equal((world3of4.payload as WorldSyncPayload).levelId, 1);
      for (const c of [p1, p2, p3, p4]) {
        assert.equal(c.packets.filter((p) => p.event === 'game:level_changed').length, 0, 'No player advances before all 4 reach portal');
      }

      // Mandatory Test 3: P4 dies while P1, P2, P3 wait at Level 1 portal -> P1..P3 remain waiting & P4 respawns on Level 1 (no Spectator Mode)
      const p4CaughtWait = p4.waitFor(
        (p) =>
          p.event === 'game:player_event' &&
          (p.payload as GamePlayerEvent).kind === 'caught' &&
          (p.payload as GamePlayerEvent).userId === p4.user.id,
        'p4 caught while teammates wait at portal',
        EAT_DURATION_MS + 4500
      );
      const p4RespawnWait = p4.waitFor(
        (p) =>
          p.event === 'game:player_event' &&
          (p.payload as GamePlayerEvent).kind === 'respawned' &&
          (p.payload as GamePlayerEvent).userId === p4.user.id,
        'p4 respawns automatically on Level 1',
        EAT_DURATION_MS + RESPAWN_DELAY_MS + 4500
      );
      const p4SwimLoop = setInterval(() => {
        p4.send('game:input_state', {
          levelId: 1,
          x: 0,
          y: lv1Def.bounds.waterY - 0.8,
          z: -24,
          yaw: 0,
          animState: 'swim',
          horizSpeed: 2,
          inWater: true,
          checkpointIndex: 1,
          escaped: false,
        });
      }, 45);
      await p4CaughtWait;
      clearInterval(p4SwimLoop);
      const p4RespEv = (await p4RespawnWait).payload as GamePlayerEvent;
      assert.equal(p4RespEv.levelId, 1, 'Dead player respawns at Level 1 start while teammates wait');

      // Verify waiting players are still at Level 1 portal and room is unbroken
      const waitCheckWorld = await p1.waitFor(
        (p) => {
          if (p.event !== 'game:world') return false;
          const w = p.payload as WorldSyncPayload;
          const s1 = w.players.find((pl) => pl.userId === p1.user.id);
          const s4 = w.players.find((pl) => pl.userId === p4.user.id);
          return Boolean(
            w.levelId === 1 &&
              w.portalReadyCount === 3 &&
              s1?.escaped &&
              s4?.alive &&
              !s4?.escaped &&
              s4?.checkpointIndex === 0
          );
        },
        'p1..p3 stay waiting at portal while p4 auto-respawns at CP 0'
      );
      assert.ok(waitCheckWorld);

      // Mandatory Test 6: P4 reaches portal -> ALL 4 players transition together to Level 2
      const teamLv2Waits = Promise.all(
        [p1, p2, p3, p4].map((c, idx) =>
          c.waitFor(
            (p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 2,
            `p${idx + 1} receives synchronized transition to level 2`
          )
        )
      );
      sendAtExit(p4);
      await new Promise((r) => setTimeout(r, 40));
      p4.send('game:portal_enter', { levelId: 1 });
      // Also spam portal_enter from P1 to ensure no double transition to Level 3
      p1.send('game:portal_enter', { levelId: 1 });
      const lv2Pkts = await teamLv2Waits;
      for (const pkt of lv2Pkts) {
        const pay = pkt.payload as GameLevelChangedPayload;
        assert.equal(pay.prevLevelId, 1);
        assert.equal(pay.levelId, 2);
        assert.ok(pay.players.every((pl) => pl.levelId === 2 && !pl.escaped && pl.alive && pl.checkpointIndex === 0));
      }
      await new Promise((r) => setTimeout(r, 100));
      for (const c of [p1, p2, p3, p4]) {
        assert.equal(
          c.packets.filter((p) => p.event === 'game:level_changed').length,
          1,
          'Exactly one level transition event per level completion (no duplicate transitions)'
        );
        c.send('game:level_ready', { levelId: 2 });
        assert.equal(c.currentRoom?.id, p1.currentRoom?.id);
        assert.equal(c.currentRoom?.levelId, 2);
      }

      // Verify everyone is active on Level 2 spawn deck
      const lv2WorldPkt = await p1.waitFor(
        (p) =>
          p.event === 'game:world' &&
          (p.payload as WorldSyncPayload).levelId === 2 &&
          (p.payload as WorldSyncPayload).players.every((pl) => pl.levelId === 2 && pl.alive && !pl.escaped),
        'all 4 players synced onto Level 2 together'
      );
      assert.equal((lv2WorldPkt.payload as WorldSyncPayload).portalReadyCount, 0);
      assert.ok(lv2Def.id === 2);
      console.log('   PASS: 4-player Team Portal Wait, staggered arrival, teammate death during wait, duplicate guard & synchronized Level 2 transition verified.');
    }

    console.log('19. Testing Scenario 19: Level 2+ checkpoint-0 respawn (NOT Level 1), simultaneous deaths & no Spectator Mode...');
    {
      const d1 = await makeClient('DeathLv2_1');
      const d2 = await makeClient('DeathLv2_2');
      const d3 = await makeClient('DeathLv2_3');
      const crWait = d1.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'death-lv2 room created');
      d1.send('room:create', { levelId: 1, gameVersion: GAME_VERSION });
      await crWait;
      const code = d1.currentRoom!.code;
      await d2.http('/api/rooms/join', 'POST', { code, gameVersion: GAME_VERSION });
      await d3.http('/api/rooms/join', 'POST', { code, gameVersion: GAME_VERSION });
      await new Promise((r) => setTimeout(r, 40));

      await Promise.all([
        (async () => {
          const st = [d1, d2, d3].map((c, i) => c.waitFor((p) => p.event === 'game:started', `d${i + 1} started`));
          d1.send('room:start');
          await Promise.all(st);
        })(),
      ]);

      // Advance team cleanly from Level 1 -> Level 2 together
      const lv1Def = getLevelById(1);
      const lv2Def = getLevelById(2);
      const lv2TransWaits = Promise.all(
        [d1, d2, d3].map((c, i) =>
          c.waitFor((p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 2, `d${i + 1} -> lv2`)
        )
      );
      for (const c of [d1, d2, d3]) {
        c.send('game:input_state', {
          levelId: 1,
          x: lv1Def.exit.position[0],
          y: lv1Def.exit.position[1],
          z: lv1Def.exit.position[2],
          yaw: 0,
          animState: 'run',
          horizSpeed: 6,
          inWater: false,
          checkpointIndex: 1,
          escaped: true,
        });
      }
      await lv2TransWaits;
      for (const c of [d1, d2, d3]) c.send('game:level_ready', { levelId: 2 });
      await new Promise((r) => setTimeout(r, 60));

      // Place d3 safely waiting at Level 2 exit portal; send d1 and d2 into Level 2 water simultaneously with checkpointIndex=1
      d3.send('game:input_state', {
        levelId: 2,
        x: lv2Def.exit.position[0],
        y: lv2Def.exit.position[1],
        z: lv2Def.exit.position[2],
        yaw: 0,
        animState: 'idle',
        horizSpeed: 0,
        inWater: false,
        checkpointIndex: 1,
        escaped: true,
      });

      const d1CaughtWait = d1.waitFor(
        (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'caught' && (p.payload as GamePlayerEvent).userId === d1.user.id,
        'd1 caught on Level 2',
        EAT_DURATION_MS + 6000
      );
      const d2CaughtWait = d2.waitFor(
        (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'caught' && (p.payload as GamePlayerEvent).userId === d2.user.id,
        'd2 caught simultaneously on Level 2',
        EAT_DURATION_MS + 6000
      );
      const d1RespWait = d1.waitFor(
        (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'respawned' && (p.payload as GamePlayerEvent).userId === d1.user.id,
        'd1 respawned on Level 2',
        EAT_DURATION_MS + RESPAWN_DELAY_MS + 6500
      );
      const d2RespWait = d2.waitFor(
        (p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'respawned' && (p.payload as GamePlayerEvent).userId === d2.user.id,
        'd2 respawned on Level 2',
        EAT_DURATION_MS + RESPAWN_DELAY_MS + 6500
      );

      // Position d1 and d2 near Level 2's serpents in water so both are caught concurrently
      const lv2M0 = lv2Def.monsters[0];
      const lv2M1 = lv2Def.monsters[1] || lv2Def.monsters[0];
      const simSwimTimer = setInterval(() => {
        d1.send('game:input_state', {
          levelId: 2,
          x: lv2M0.waypoints[0][0],
          y: lv2Def.bounds.waterY - 0.8,
          z: lv2M0.waypoints[0][1],
          yaw: 0,
          animState: 'swim',
          horizSpeed: 2,
          inWater: true,
          checkpointIndex: 1,
          escaped: false,
        });
        d2.send('game:input_state', {
          levelId: 2,
          x: lv2M1.waypoints[0][0] + 0.5,
          y: lv2Def.bounds.waterY - 0.8,
          z: lv2M1.waypoints[0][1] + 0.5,
          yaw: 0,
          animState: 'swim',
          horizSpeed: 2,
          inWater: true,
          checkpointIndex: 1,
          escaped: false,
        });
      }, 45);

      await Promise.all([d1CaughtWait, d2CaughtWait]);
      clearInterval(simSwimTimer);
      const [r1Pkt, r2Pkt] = await Promise.all([d1RespWait, d2RespWait]);
      const r1Ev = r1Pkt.payload as GamePlayerEvent;
      const r2Ev = r2Pkt.payload as GamePlayerEvent;

      // Mandatory Tests 4 & 5: Respawn at Level 2 checkpoint 0 (NOT Level 1!), simultaneous deaths cleanly handled, no Spectator Mode
      const lv2StartPlat = lv2Def.platforms[0];
      const maxDx = Math.max(2.5, (lv2StartPlat?.size[0] ?? 6.4) * 0.5);
      const maxDz = Math.max(2.5, (lv2StartPlat?.size[2] ?? 5.4) * 0.5);
      const cx2 = lv2StartPlat ? lv2StartPlat.position[0] : lv2Def.spawn[0];
      const cz2 = lv2StartPlat ? lv2StartPlat.position[2] : lv2Def.spawn[2];
      assert.equal(r1Ev.levelId, 2, 'd1 must respawn on current Level 2, NOT Level 1');
      assert.equal(r2Ev.levelId, 2, 'd2 must respawn on current Level 2, NOT Level 1');
      assert.ok(Math.abs((r1Ev.x ?? 999) - cx2) <= maxDx, 'd1 respawn X on Level 2 starting deck');
      assert.ok(Math.abs((r1Ev.z ?? 999) - cz2) <= maxDz, 'd1 respawn Z on Level 2 starting deck');
      assert.ok(Math.abs((r2Ev.x ?? 999) - cx2) <= maxDx, 'd2 respawn X on Level 2 starting deck');
      assert.ok(Math.abs((r2Ev.z ?? 999) - cz2) <= maxDz, 'd2 respawn Z on Level 2 starting deck');

      await new Promise((r) => setTimeout(r, 80));
      assert.equal(
        d1.packets.filter((p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'respawned' && (p.payload as GamePlayerEvent).userId === d1.user.id).length,
        1,
        'No duplicate respawn for d1'
      );
      assert.equal(
        d2.packets.filter((p) => p.event === 'game:player_event' && (p.payload as GamePlayerEvent).kind === 'respawned' && (p.payload as GamePlayerEvent).userId === d2.user.id).length,
        1,
        'No duplicate respawn for d2'
      );

      // Confirm world stays on Level 2, d1/d2 alive at CP 0, d3 still waiting at portal (`escaped: true`)
      const lv2PostRespWorld = await d1.waitFor((p) => {
        if (p.event !== 'game:world') return false;
        const w = p.payload as WorldSyncPayload;
        const p1s = w.players.find((pl) => pl.userId === d1.user.id);
        const p2s = w.players.find((pl) => pl.userId === d2.user.id);
        const p3s = w.players.find((pl) => pl.userId === d3.user.id);
        return Boolean(
          w.levelId === 2 &&
            w.matchResult === null &&
            p1s?.alive &&
            p1s.checkpointIndex === 0 &&
            p1s.levelId === 2 &&
            p2s?.alive &&
            p2s.checkpointIndex === 0 &&
            p2s.levelId === 2 &&
            p3s?.alive &&
            p3s.escaped &&
            p3s.levelId === 2
        );
      }, 'post-simultaneous-respawn world check on Level 2');
      assert.ok(lv2PostRespWorld);

      // Now respawned d1 & d2 recover controls, reach Level 2 exit portal -> all 3 transition to Level 3 together!
      const lv3TransWaits = Promise.all(
        [d1, d2, d3].map((c, i) =>
          c.waitFor((p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 3, `d${i + 1} -> lv3 together`)
        )
      );
      for (const c of [d1, d2]) {
        c.send('game:input_state', {
          levelId: 2,
          x: lv2Def.exit.position[0],
          y: lv2Def.exit.position[1],
          z: lv2Def.exit.position[2],
          yaw: 0,
          animState: 'run',
          horizSpeed: 6,
          inWater: false,
          checkpointIndex: 1,
          escaped: true,
        });
      }
      await lv3TransWaits;
      assert.equal(d1.currentRoom?.levelId, 3);
      assert.equal(d2.currentRoom?.id, d1.currentRoom?.id);
      assert.equal(d3.currentRoom?.id, d1.currentRoom?.id);
      console.log('   PASS: Level 2+ checkpoint-0 respawn, simultaneous deaths, preserved portal wait & Level 3 team transition verified.');
    }

    console.log('20. Testing Scenario 20: Disconnect/leave edge cases, Level 100 completion & Cloudflare GameRoomDO Team Portal/Respawn parity...');
    {
      // Part A: Permanent leave unblocks teammates waiting at portal; Final Level 100 requires all active teammates before match completion
      const e1 = await makeClient('Edge1');
      const e2 = await makeClient('Edge2');
      const e3 = await makeClient('Edge3');
      const crWait = e1.waitFor((p) => p.event === 'room:state' && p.payload !== null, 'edge room created');
      e1.send('room:create', { levelId: 99, gameVersion: GAME_VERSION });
      await crWait;
      await e2.http('/api/rooms/join', 'POST', { code: e1.currentRoom!.code, gameVersion: GAME_VERSION });
      await e3.http('/api/rooms/join', 'POST', { code: e1.currentRoom!.code, gameVersion: GAME_VERSION });
      await new Promise((r) => setTimeout(r, 40));
      const startWaits = Promise.all([
        e1.waitFor((p) => p.event === 'game:started', 'e1 started lv99'),
        e2.waitFor((p) => p.event === 'game:started', 'e2 started lv99'),
        e3.waitFor((p) => p.event === 'game:started', 'e3 started lv99'),
      ]);
      e1.send('room:start');
      await startWaits;

      const lv99Def = getLevelById(99);
      const lv100Def = getLevelById(100);
      // e1 & e2 wait at Level 99 portal while e3 leaves permanently -> e1 & e2 must immediately advance to Level 100 without getting stuck
      for (const c of [e1, e2]) {
        c.send('game:input_state', {
          levelId: 99,
          x: lv99Def.exit.position[0],
          y: lv99Def.exit.position[1],
          z: lv99Def.exit.position[2],
          yaw: 0,
          animState: 'idle',
          horizSpeed: 0,
          inWater: false,
          checkpointIndex: 0,
          escaped: true,
        });
      }
      await new Promise((r) => setTimeout(r, 50));
      assert.equal(e1.packets.filter((p) => p.event === 'game:level_changed').length, 0, 'Wait while e3 still in room');

      const lv100AdvanceWaits = Promise.all([
        e1.waitFor((p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 100, 'e1 -> lv100 when e3 leaves'),
        e2.waitFor((p) => p.event === 'game:level_changed' && (p.payload as GameLevelChangedPayload).levelId === 100, 'e2 -> lv100 when e3 leaves'),
      ]);
      e3.send('room:leave');
      await lv100AdvanceWaits;
      e1.send('game:level_ready', { levelId: 100 });
      e2.send('game:level_ready', { levelId: 100 });

      // On final Level 100, e1 reaching portal must wait for e2; only when e2 also reaches Level 100 portal does match finish cleanly
      e1.send('game:input_state', {
        levelId: 100,
        x: lv100Def.exit.position[0],
        y: lv100Def.exit.position[1],
        z: lv100Def.exit.position[2],
        yaw: 0,
        animState: 'idle',
        horizSpeed: 0,
        inWater: false,
        checkpointIndex: 0,
        escaped: true,
      });
      await new Promise((r) => setTimeout(r, 80));
      assert.equal(e1.currentRoom?.status, 'playing', 'Level 100 must wait for last teammate before ending match');

      const finalEndWaits = Promise.all([
        e1.waitFor((p) => p.event === 'game:world' && Boolean((p.payload as WorldSyncPayload).matchResult), 'e1 match finalized'),
        e2.waitFor((p) => p.event === 'game:world' && Boolean((p.payload as WorldSyncPayload).matchResult), 'e2 match finalized'),
        e1.waitFor((p) => p.event === 'room:state' && (p.payload as RoomState | null)?.status === 'ended', 'e1 room status ended'),
      ]);
      e2.send('game:input_state', {
        levelId: 100,
        x: lv100Def.exit.position[0],
        y: lv100Def.exit.position[1],
        z: lv100Def.exit.position[2],
        yaw: 0,
        animState: 'idle',
        horizSpeed: 0,
        inWater: false,
        checkpointIndex: 0,
        escaped: true,
      });
      const [fin1, fin2] = await finalEndWaits;
      const r1 = (fin1.payload as WorldSyncPayload).matchResult!;
      assert.equal(r1.reason, 'all_resolved');
      assert.equal(r1.outcome, 'escaped');
      assert.equal(r1.survivors.length, 2);
      assert.equal((fin2.payload as WorldSyncPayload).matchResult?.survivors.length, 2);

      // Part B: Cloudflare GameRoomDO Team-Based Portal Wait, Simultaneous Arrival, and No Spectator Mode parity
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

      const u1 = { id: 'cf-p1', username: 'CFProg1', avatarColor: '#52e0b6', avatarIcon: 'diver' as const, status: 'online' as const };
      const u2 = { id: 'cf-p2', username: 'CFProg2', avatarColor: '#f59e0b', avatarIcon: 'diver' as const, status: 'online' as const };
      await callRoom({ action: 'create', roomId: 'cf-prog-room', code: 'CFPRG1', user: u1, levelId: 1, gameVersion: GAME_VERSION });

      const badVerJoin = await callRoom({ action: 'join', user: u2, gameVersion: '0.1.0-mismatch' });
      assert.match(String(badVerJoin.error), /version|update your game/i);

      await callRoom({ action: 'join', user: u2, gameVersion: GAME_VERSION });
      await callRoom({ action: 'start', userId: u1.id });

      const lv1Def = getLevelById(1);
      const lv2Def = getLevelById(2);
      await callRoom({
        action: 'input_state',
        userId: u1.id,
        packet: {
          levelId: 1,
          x: lv1Def.exit.position[0],
          y: lv1Def.exit.position[1],
          z: lv1Def.exit.position[2],
          yaw: 0,
          animState: 'run',
          horizSpeed: 6,
          inWater: false,
          checkpointIndex: 1,
          escaped: false,
        },
      });

      // u1 enters portal alone -> waits at Level 1 portal (`portalReadyCount=1, portalTotalCount=2`), NO `game:level_changed` yet
      const p1PortalRes = await callRoom({ action: 'portal_enter', userId: u1.id, levelId: 1 });
      assert.equal(
        p1PortalRes.batch?.deliveries?.some((d: any) => d.event === 'game:level_changed'),
        false,
        'Cloudflare DO must NOT advance u1 alone before u2 reaches portal'
      );
      const p1WaitingWorld = p1PortalRes.batch?.deliveries?.find(
        (d: any) => d.event === 'game:world' && d.targetUserId === u1.id
      )?.payload as WorldSyncPayload | undefined;
      assert.equal(p1WaitingWorld?.portalReadyCount, 1);
      assert.equal(p1WaitingWorld?.portalTotalCount, 2);

      // u2 enters Level 1 portal -> Cloudflare DO emits game:level_changed (Level 2) for BOTH u1 and u2 simultaneously
      await callRoom({
        action: 'input_state',
        userId: u2.id,
        packet: {
          levelId: 1,
          x: lv1Def.exit.position[0],
          y: lv1Def.exit.position[1],
          z: lv1Def.exit.position[2],
          yaw: 0,
          animState: 'run',
          horizSpeed: 6,
          inWater: false,
          checkpointIndex: 1,
          escaped: false,
        },
      });
      const p2PortalRes = await callRoom({ action: 'portal_enter', userId: u2.id, levelId: 1 });
      const lvlChangedDelivs = (p2PortalRes.batch?.deliveries || []).filter((d: any) => d.event === 'game:level_changed');
      assert.equal(lvlChangedDelivs.length, 2, 'Cloudflare DO must transition both teammates together');
      assert.ok(lvlChangedDelivs.every((d: any) => d.payload.levelId === 2 && d.payload.prevLevelId === 1));

      await callRoom({ action: 'level_ready', userId: u1.id, levelId: 2 });
      await callRoom({ action: 'level_ready', userId: u2.id, levelId: 2 });
      const lv2InputRes = await callRoom({
        action: 'input_state',
        userId: u1.id,
        packet: {
          levelId: 2,
          x: lv2Def.spawn[0],
          y: lv2Def.spawn[1],
          z: lv2Def.spawn[2],
          yaw: Math.PI,
          animState: 'walk',
          horizSpeed: 4,
          inWater: false,
          checkpointIndex: 0,
          escaped: false,
        },
      });
      const u1WorldDeliv = lv2InputRes.batch?.deliveries?.find(
        (d: any) => d.event === 'game:world' && d.targetUserId === u1.id
      );
      const u1StateAfterPortal = (u1WorldDeliv?.payload as WorldSyncPayload)?.players?.find(
        (pl: any) => pl.userId === u1.id
      );
      const u2StateAfterPortal = (u1WorldDeliv?.payload as WorldSyncPayload)?.players?.find(
        (pl: any) => pl.userId === u2.id
      );
      assert.equal(u1StateAfterPortal?.levelId, 2);
      assert.equal(u2StateAfterPortal?.levelId, 2);

      // Verify temporary disconnect keeps waiting teammate paused cleanly until reconnect
      await callRoom({ action: 'disconnect', userId: u2.id });
      await callRoom({
        action: 'input_state',
        userId: u1.id,
        packet: {
          levelId: 2,
          x: lv2Def.exit.position[0],
          y: lv2Def.exit.position[1],
          z: lv2Def.exit.position[2],
          yaw: 0,
          animState: 'idle',
          horizSpeed: 0,
          inWater: false,
          checkpointIndex: 1,
          escaped: true,
        },
      });
      const waitDuringDisc = await callRoom({ action: 'portal_enter', userId: u1.id, levelId: 2 });
      assert.equal(
        waitDuringDisc.batch?.deliveries?.some((d: any) => d.event === 'game:level_changed'),
        false,
        'Cloudflare DO must tolerate temporary disconnect grace window without abandoning u2'
      );
      const reconRes = await callRoom({ action: 'reconnect', userId: u2.id });
      assert.equal(reconRes.room?.levelId, 2, 'Reconnecting player must rejoin current team level 2');

      await callRoom({ action: 'close', userId: u1.id });
      console.log('   PASS: Disconnect/leave edge cases, Level 100 completion & Cloudflare GameRoomDO Team Portal parity verified.');
    }

    console.log('\nALL 20 MULTIPLAYER, VERSION, TEAM PORTAL PROGRESSION, RESPAWN, NO-SPECTATOR & CLOUDFLARE PARITY SCENARIOS PASSED!');
  } finally {
    for (const c of clients) {
      c.close();
    }
    await srv.close();
  }
}

runTests().catch((err) => {
  console.error('TEST FAILURE:', err);
  process.exit(1);
});
