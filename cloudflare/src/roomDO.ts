import { getLevelById, isValidLevelId, TOTAL_LEVELS } from '../../src/game/levels.ts';
import {
  computeSerpentSteerAngle,
  normalizeAngle,
  resetSerpentNavProgress,
  resolvePatrolTarget,
  selectClosestPatrolWaypoint,
  SERPENT_INTIMIDATION_LOWER_SEC,
  SERPENT_INTIMIDATION_RISE_SEC,
  SERPENT_WATCH_DURATION_SEC,
  SERPENT_WATCH_FREEZE_GRACE_SEC,
  shortestAngleDiffNav,
  updateSerpentProgressAndYaw,
} from '../../src/game/serpentNav.ts';
import { LevelSpatialGrid } from '../../src/game/spatialGrid.ts';
import { normalizeCharacterId, type CharacterId, type LevelDef, type PlatformDef } from '../../src/game/types.ts';
import {
  checkVersionCompatibility,
  EAT_DURATION_MS,
  GAME_VERSION,
  LEVEL_TRANSITION_GRACE_MS,
  MAX_ROOM_PLAYERS,
  MIN_ROOM_PLAYERS,
  PUSH_COOLDOWN_MS,
  PUSH_FORCE_HORIZ,
  PUSH_FORCE_LIFT_Y,
  PUSH_RECOVERY_MS,
  RESPAWN_DELAY_MS,
  type ClientPlayerInputPacket,
  type ClientPushPacket,
  type GameLevelChangedPayload,
  type GamePlayerEvent,
  type MatchResultSummary,
  type MonsterNetState,
  type PlayerAnimSyncState,
  type PlayerNetState,
  type PublicUser,
  type RoomInviteInfo,
  type RoomPlayerInfo,
  type RoomState,
  type RoomVoicePeerState,
  type VoiceSignalKind,
} from '../../src/multiplayer/types.ts';
import { computePushNudge, selectPushTarget } from '../../src/multiplayer/pushPhysics.ts';
import type { DeliveryMessage, Env, RoomDeliveryBatch } from './types.ts';

interface InternalRoomPlayer extends RoomPlayerInfo {
  characterId: CharacterId;
  levelId: number;
  transitioning: boolean;
  transitioningUntil: number;
  respawnAt: number;
  deaths: number;
  levelTimeRemaining: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  animState: PlayerAnimSyncState;
  horizSpeed: number;
  velX: number;
  velY: number;
  velZ: number;
  inWater: boolean;
  lastUpdateAt: number;
  lastPushAt: number;
  pushImmunityUntil: number;
  beingEatenUntil: number;
  beingEatenByMonsterId: string | null;
  moveAttemptWhileWatched?: boolean;
  wasInWaterSession?: boolean;
  waterEpoch?: number;
  pendingPlatformWatchEpoch?: number;
  watchHandledEpoch?: number;
}

interface InternalMonster extends MonsterNetState {
  waypoints: [number, number][];
  patrolSpeed: number;
  chaseSpeed: number;
  waterDetectionRadius: number;
  scale: number;
  avoidDir: 1 | -1;
  avoidTimer: number;
  stuckTimer: number;
  navBestDist: number;
  navStallTimer: number;
  navAnchorX: number;
  navAnchorZ: number;
  navAnchorTimer: number;
  navTurnAccum: number;
  diveTimer?: number;
  attackHoldUntil?: number;
  eatingStartAt?: number;
  eatingUntil?: number;
  visionState?:
    | 'none'
    | 'lunging'
    | 'platform_approaching'
    | 'platform_rising'
    | 'staring'
    | 'platform_lowering'
    | 'platform_retreating'
    | 'platform_lunging';
  visionUntil?: number;
  lookAwayYaw?: number;
  watchAnchorX?: number;
  watchAnchorZ?: number;
  watchGraceUntil?: number;
  intimStartedAt?: number;
  intimDurationMs?: number;
  platformSpotX?: number;
  platformSpotZ?: number;
  intimYaw?: number;
}

interface InternalLevelSim {
  levelId: number;
  levelDef: LevelDef;
  monsters: InternalMonster[];
}

interface InternalRoom {
  id: string;
  code: string;
  hostId: string;
  levelId: number;
  gameVersion: string;
  status: 'lobby' | 'playing' | 'ended';
  players: Map<string, InternalRoomPlayer>;
  invites: Map<string, RoomInviteInfo>;
  monsters: InternalMonster[];
  levelSims: Map<number, InternalLevelSim>;
  levelDef: LevelDef;
  matchStartTime: number | null;
  timeRemaining: number;
  matchResult: MatchResultSummary | null;
  lastTickMs: number;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function r3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function r2(v: number): number {
  return Math.round(v * 100) / 100;
}

function shortestAngleDiff(a: number, b: number): number {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

function worldToRotatedPlatformLocal(
  dx: number,
  dy: number,
  dz: number,
  plat: PlatformDef,
  elapsedSec: number
): [number, number, number] {
  const axis = plat.rotationAxis || 'y';
  const spd = plat.rotationSpeed || 0;
  const phase = plat.rotationPhase ?? 0;
  const ang = phase + elapsedSec * spd;

  let rx = 0;
  let ry = 0;
  let rz = 0;
  if (axis === 'y') ry = ang;
  else if (axis === 'x') rx = ang;
  else if (axis === 'z') rz = ang;
  else if (axis === 'xy') {
    rx = phase * 0.7 + elapsedSec * spd * 0.55;
    ry = ang;
  } else if (axis === 'zy') {
    ry = ang;
    rz = phase * 0.7 + elapsedSec * spd * 0.55;
  }

  // Inverse of Three.js 'YXZ' Euler (R_y(ry) * R_x(rx) * R_z(rz)) is R_z(-rz) * R_x(-rx) * R_y(-ry)
  const cy = Math.cos(ry);
  const sy = Math.sin(ry);
  const x1 = dx * cy - dz * sy;
  const y1 = dy;
  const z1 = dx * sy + dz * cy;

  const cx = Math.cos(rx);
  const sx = Math.sin(rx);
  const x2 = x1;
  const y2 = y1 * cx + z1 * sx;
  const z2 = -y1 * sx + z1 * cx;

  const cz = Math.cos(rz);
  const sz = Math.sin(rz);
  const lx = x2 * cz + y2 * sz;
  const ly = -x2 * sz + y2 * cz;
  const lz = z2;

  return [lx, ly, lz];
}

function getLevelDef(levelId: number): LevelDef {
  return getLevelById(levelId);
}

export class GameRoomDO {
  private ctx: DurableObjectState;
  private env: Env;
  private room: InternalRoom | null = null;
  private tickTimer: ReturnType<typeof setInterval> | null = null;
  private disconnectDeadlines = new Map<string, number>();
  private lastWorldBroadcastMs = 0;
  private dirtyPlayersSinceBroadcast = false;
  private tickDeliverInFlight = false;
  private spatialGrids = new Map<number, LevelSpatialGrid>();
  private scratchPlats: number[] = [];
  private scratchWalls: number[] = [];
  private aliveInWaterScratch: InternalRoomPlayer[] = [];
  private claimedTargetsScratch = new Set<string>();
  private _cbDef: LevelDef | null = null;
  private _cbBotY = 0;
  private _cbTopY = 0;
  private _cbElapsed = 0;
  private readonly checkMonsterBlockedFn = (cx: number, cz: number, cr: number): boolean =>
    this.isCircleBlocked(this._cbDef!, cx, cz, cr, this._cbBotY, this._cbTopY, this._cbElapsed);

  constructor(ctx: DurableObjectState, env: Env) {
    this.ctx = ctx;
    this.env = env;
  }

  private scheduleNextAlarm(delayMs = 100): void {
    const now = Date.now();
    let targetMs = Infinity;
    if (this.room && this.room.status === 'playing') {
      targetMs = now + Math.max(50, delayMs);
    }
    for (const deadline of this.disconnectDeadlines.values()) {
      if (deadline < targetMs) targetMs = deadline;
    }
    if (!Number.isFinite(targetMs)) {
      if (typeof this.ctx?.storage?.deleteAlarm === 'function') {
        void this.ctx.storage.deleteAlarm().catch(() => {});
      }
      return;
    }
    if (typeof this.ctx?.storage?.setAlarm === 'function') {
      void this.ctx.storage.setAlarm(Math.max(now + 25, targetMs)).catch(() => {});
    }
  }

  public async alarm(): Promise<void> {
    const now = Date.now();
    const batch = this.createBatch();
    let didWork = false;

    // Expire any disconnected players past grace deadline
    if (this.room && this.disconnectDeadlines.size > 0) {
      for (const [uid, deadline] of Array.from(this.disconnectDeadlines.entries())) {
        if (now >= deadline) {
          this.disconnectDeadlines.delete(uid);
          const p = this.room?.players.get(uid);
          if (p && !p.connected) {
            this.removePlayerInternal(uid, 'left', batch);
            didWork = true;
          }
        }
      }
    }

    if (this.room && this.room.status === 'playing') {
      if (this.dirtyPlayersSinceBroadcast || now - this.lastWorldBroadcastMs >= 40) {
        this.advanceRoomSimulation(this.room, now, batch, true);
        didWork = true;
      }
    }

    if (didWork) {
      await this.deliverBatchAsync(batch);
    }

    if ((this.room && this.room.status === 'playing') || this.disconnectDeadlines.size > 0) {
      this.scheduleNextAlarm(90);
    }
  }

  private getLevelGrid(def: LevelDef): LevelSpatialGrid {
    let grid = this.spatialGrids.get(def.id);
    if (!grid) {
      grid = new LevelSpatialGrid(def, 12);
      this.spatialGrids.set(def.id, grid);
    }
    return grid;
  }

  private getDisconnectGraceMs(): number {
    const parsed = Number(this.env.DISCONNECT_GRACE_MS || 8000);
    return Number.isFinite(parsed) && parsed >= 500 ? parsed : 8000;
  }

  private makeRoomPlayer(user: PublicUser, isHost: boolean, levelDef: LevelDef): InternalRoomPlayer {
    return {
      userId: user.id,
      username: user.username,
      avatarColor: user.avatarColor,
      avatarIcon: user.avatarIcon,
      characterId: 'male',
      isHost,
      isBot: false,
      connected: true,
      alive: true,
      escaped: false,
      levelId: levelDef.id,
      transitioning: false,
      transitioningUntil: 0,
      respawnAt: 0,
      deaths: 0,
      levelTimeRemaining: levelDef.timeLimit,
      checkpointIndex: 0,
      x: levelDef.spawn[0],
      y: levelDef.spawn[1],
      z: levelDef.spawn[2],
      yaw: Math.PI,
      animState: 'idle',
      horizSpeed: 0,
      velX: 0,
      velY: 0,
      velZ: 0,
      inWater: false,
      lastUpdateAt: Date.now(),
      lastPushAt: 0,
      pushImmunityUntil: 0,
      beingEatenUntil: 0,
      beingEatenByMonsterId: null,
      moveAttemptWhileWatched: false,
      wasInWaterSession: false,
      waterEpoch: 0,
      pendingPlatformWatchEpoch: 0,
      watchHandledEpoch: 0,
      voice: {
        voiceActive: false,
        micEnabled: false,
        selfMuted: false,
        speaking: false,
      },
    };
  }

  private serializeRoom(room: InternalRoom): RoomState {
    return {
      id: room.id,
      code: room.code,
      hostId: room.hostId,
      levelId: room.levelId,
      gameVersion: room.gameVersion,
      status: room.status,
      isPrivate: true,
      minPlayers: MIN_ROOM_PLAYERS,
      maxPlayers: MAX_ROOM_PLAYERS,
      players: Array.from(room.players.values()).map((p) => ({
        userId: p.userId,
        username: p.username,
        avatarColor: p.avatarColor,
        avatarIcon: p.avatarIcon,
        characterId: p.characterId,
        isHost: p.userId === room.hostId,
        isBot: false,
        connected: p.connected,
        alive: p.alive,
        escaped: p.escaped,
        levelId: p.levelId,
        transitioning: p.transitioning,
        deaths: p.deaths,
        checkpointIndex: p.checkpointIndex,
        voice: { ...p.voice },
      })),
      invitedFriendIds: Array.from(room.invites.keys()),
      matchStartTime: room.matchStartTime,
    };
  }

  private serializeNetPlayers(room: InternalRoom, compact = false): PlayerNetState[] {
    const out: PlayerNetState[] = [];
    for (const p of room.players.values()) {
      if (compact) {
        out.push({
          userId: p.userId,
          characterId: p.characterId,
          levelId: p.levelId,
          transitioning: p.transitioning,
          x: r3(p.x),
          y: r3(p.y),
          z: r3(p.z),
          yaw: r3(p.yaw),
          animState: p.animState,
          horizSpeed: r2(p.horizSpeed),
          velX: r2(p.velX),
          velY: r2(p.velY),
          velZ: r2(p.velZ),
          inWater: p.inWater,
          alive: p.alive,
          escaped: p.escaped,
          checkpointIndex: p.checkpointIndex,
        });
      } else {
        out.push({
          userId: p.userId,
          username: p.username,
          avatarColor: p.avatarColor,
          characterId: p.characterId,
          levelId: p.levelId,
          transitioning: p.transitioning,
          x: r3(p.x),
          y: r3(p.y),
          z: r3(p.z),
          yaw: r3(p.yaw),
          animState: p.animState,
          horizSpeed: r2(p.horizSpeed),
          velX: r2(p.velX),
          velY: r2(p.velY),
          velZ: r2(p.velZ),
          inWater: p.inWater,
          alive: p.alive,
          escaped: p.escaped,
          checkpointIndex: p.checkpointIndex,
        });
      }
    }
    return out;
  }

  private serializeNetMonstersList(monsters: InternalMonster[]): MonsterNetState[] {
    const out: MonsterNetState[] = [];
    for (let i = 0; i < monsters.length; i++) {
      const m = monsters[i];
      out.push({
        id: m.id,
        x: r3(m.x),
        y: r3(m.y),
        z: r3(m.z),
        yaw: r3(m.yaw),
        velX: r2(m.velX),
        velZ: r2(m.velZ),
        alert: m.alert,
        targetUserId: m.targetUserId,
        emerged: m.emerged,
        wpIndex: m.wpIndex,
        eatingUserId: m.eatingUserId ?? null,
        eatingProgress: r3(m.eatingProgress ?? 0),
        watchingUserId: m.watchingUserId ?? null,
        lookAway: Boolean(m.lookAway),
        intimPhase: m.intimPhase ?? null,
        intimProgress: r3(m.intimProgress ?? 0),
      });
    }
    return out;
  }

  private roomPlayerIds(room: InternalRoom): string[] {
    return Array.from(room.players.keys());
  }

  private async deliverBatchAsync(batch: RoomDeliveryBatch): Promise<void> {
    if (
      batch.deliveries.length === 0 &&
      !batch.closedRoom &&
      !batch.removedUsers?.length &&
      !batch.joinedUsers?.length &&
      !batch.presenceChangedUserIds?.length &&
      !batch.purgedInvites?.length
    ) {
      return;
    }
    const hubId = this.env.PRESENCE_HUB.idFromName('global-presence-hub');
    const hub = this.env.PRESENCE_HUB.get(hubId);
    await hub.fetch('https://presence-hub/internal/deliver', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(batch),
    });
  }

  private stopLoopAndTimers(): void {
    if (this.tickTimer) {
      clearInterval(this.tickTimer);
      this.tickTimer = null;
    }
    this.disconnectDeadlines.clear();
    if (typeof this.ctx?.storage?.deleteAlarm === 'function') {
      void this.ctx.storage.deleteAlarm().catch(() => {});
    }
  }

  private createBatch(): RoomDeliveryBatch {
    return {
      deliveries: [],
      joinedUsers: [],
      removedUsers: [],
      presenceChangedUserIds: [],
      purgedInvites: [],
    };
  }

  private pushRoomBroadcast(batch: RoomDeliveryBatch, room: InternalRoom, event: string, payload: unknown): void {
    const targets = this.roomPlayerIds(room);
    if (targets.length === 0) return;
    batch.deliveries.push({ targetUserIds: targets, event, payload });
  }

  private transferHost(room: InternalRoom, previousHostId: string, batch: RoomDeliveryBatch): void {
    const candidates = Array.from(room.players.values()).filter((p) => p.userId !== previousHostId);
    if (candidates.length === 0) return;

    const nextHost = candidates.find((c) => c.connected) || candidates[0];
    const oldHost = room.players.get(previousHostId);
    if (oldHost) oldHost.isHost = false;
    nextHost.isHost = true;
    room.hostId = nextHost.userId;

    const ev: GamePlayerEvent = {
      userId: nextHost.userId,
      username: nextHost.username,
      kind: 'host_transferred',
    };
    this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
  }

  private destroyRoomInternal(reason: string, batch: RoomDeliveryBatch): void {
    const room = this.room;
    if (!room) return;
    this.stopLoopAndTimers();

    for (const [uid, inv] of room.invites.entries()) {
      batch.purgedInvites!.push({ userId: uid, inviteId: inv.id, reason: 'room_closed' });
    }
    room.invites.clear();

    const affectedUserIds = Array.from(room.players.keys());
    this.pushRoomBroadcast(batch, room, 'room:closed', { roomId: room.id, reason });
    batch.closedRoom = {
      roomId: room.id,
      code: room.code,
      affectedUserIds,
    };
    for (const uid of affectedUserIds) {
      batch.presenceChangedUserIds!.push(uid);
    }

    room.players.clear();
    this.room = null;
  }

  private removePlayerInternal(
    userId: string,
    reason: 'left' | 'kicked',
    batch: RoomDeliveryBatch
  ): void {
    const room = this.room;
    if (!room) return;
    const removed = room.players.get(userId);
    if (!removed) return;

    this.disconnectDeadlines.delete(userId);

    const wasHost = room.hostId === userId;
    room.players.delete(userId);
    for (const sim of room.levelSims.values()) {
      this.clearPlayerFromLevelMonsters(sim, userId);
    }
    this.pruneEmptyLevelSims(room);
    batch.removedUsers!.push({ userId, roomId: room.id });
    batch.presenceChangedUserIds!.push(userId);

    this.pushRoomBroadcast(batch, room, 'voice:peer_left', { roomId: room.id, userId });

    if (reason === 'kicked') {
      batch.deliveries.push({
        targetUserId: userId,
        event: 'room:kicked',
        payload: {
          roomId: room.id,
          reason: 'You were removed from the room by the host.',
        },
      });
    } else {
      batch.deliveries.push({
        targetUserId: userId,
        event: 'room:state',
        payload: null,
      });
    }

    if (room.players.size === 0) {
      this.destroyRoomInternal('Room closed because all players left.', batch);
      return;
    }

    if (wasHost) {
      this.transferHost(room, userId, batch);
    }

    if (room.status === 'playing') {
      const ev: GamePlayerEvent = {
        userId: removed.userId,
        username: removed.username,
        kind: 'left',
      };
      this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
      this.evaluateTeamPortalProgression(room, Date.now(), batch);
    }

    if (this.room) {
      const summary = this.serializeRoom(room);
      batch.roomSummary = summary;
      this.pushRoomBroadcast(batch, room, 'room:state', summary);
    }
  }

  private isPlayerOnPlatform(
    def: LevelDef,
    x: number,
    y: number,
    z: number,
    elapsedSec = 0,
    clientInWater = false
  ): boolean {
    // Below the waterline is always wet — never falsely protected by a nearby low platform
    if (y < def.bounds.waterY - 0.12) {
      return false;
    }
    // When client already reports swimming in water, only reject water if clearly standing inside platform footprint
    const margin = clientInWater ? 0.04 : 0.36;
    const minBelowTop = clientInWater ? 0.06 : 0.16;
    this.getLevelGrid(def).queryRadius(x, z, 2.5, this.scratchPlats, null);
    for (let i = 0; i < this.scratchPlats.length; i++) {
      const plat = def.platforms[this.scratchPlats[i]];
      const [pw, ph, pd] = plat.size;
      let px = plat.position[0];
      let py = plat.position[1];
      let pz = plat.position[2];
      if (plat.kind === 'moving' && plat.moveOffset) {
        const speed = plat.moveSpeed ?? 1.0;
        const phase = plat.movePhase ?? 0;
        const factor = 0.5 - 0.5 * Math.cos(elapsedSec * speed + phase);
        px += plat.moveOffset[0] * factor;
        py += plat.moveOffset[1] * factor;
        pz += plat.moveOffset[2] * factor;
      }

      if (plat.rotationSpeed) {
        const hw = pw * 0.5;
        const hh = ph * 0.5;
        const hd = pd * 0.5;
        const axis = plat.rotationAxis || 'y';
        const rotMargin = clientInWater ? 0.08 : 0.48;
        if (axis === 'y') {
          const topY = py + hh;
          if (y >= topY - minBelowTop && y <= topY + 0.5 && y >= def.bounds.waterY - 0.06) {
            const [lx, , lz] = worldToRotatedPlatformLocal(x - px, 0, z - pz, plat, elapsedSec);
            if (Math.abs(lx) <= hw + rotMargin && Math.abs(lz) <= hd + rotMargin) {
              return true;
            }
          }
        } else if (y >= def.bounds.waterY - 0.06) {
          const [lx, ly, lz] = worldToRotatedPlatformLocal(x - px, y - py, z - pz, plat, elapsedSec);
          const ox = Math.max(0, Math.abs(lx) - hw);
          const oy = Math.max(0, Math.abs(ly) - hh);
          const oz = Math.max(0, Math.abs(lz) - hd);
          if (ox * ox + oy * oy + oz * oz <= rotMargin * rotMargin) {
            return true;
          }
        }
        continue;
      }
      const topY = py + ph * 0.5;
      if (y >= topY - minBelowTop && y <= topY + 0.45 && y >= def.bounds.waterY - 0.06) {
        if (plat.kind === 'moving' && plat.moveOffset) {
          const minX = Math.min(plat.position[0], plat.position[0] + plat.moveOffset[0]) - pw * 0.5 - margin;
          const maxX = Math.max(plat.position[0], plat.position[0] + plat.moveOffset[0]) + pw * 0.5 + margin;
          const minZ = Math.min(plat.position[2], plat.position[2] + plat.moveOffset[2]) - pd * 0.5 - margin;
          const maxZ = Math.max(plat.position[2], plat.position[2] + plat.moveOffset[2]) + pd * 0.5 + margin;
          if (x >= minX && x <= maxX && z >= minZ && z <= maxZ) return true;
        } else if (Math.abs(x - px) <= pw * 0.5 + margin && Math.abs(z - pz) <= pd * 0.5 + margin) {
          return true;
        }
      }
    }
    return false;
  }

  private isPlayerInWaterServer(
    def: LevelDef,
    x: number,
    y: number,
    z: number,
    clientInWater: boolean,
    elapsedSec = 0
  ): boolean {
    if (y < def.bounds.waterY - 0.12) return true;
    if (this.isPlayerOnPlatform(def, x, y, z, elapsedSec, clientInWater)) return false;
    if (clientInWater && y <= def.bounds.waterY + 0.06) return true;
    return y < def.bounds.waterY - 0.04;
  }

  private findPlatformUnderPlayerServer(
    def: LevelDef,
    x: number,
    y: number,
    z: number,
    elapsedSec = 0
  ): PlatformDef | null {
    if (y < def.bounds.waterY - 0.12) return null;
    const margin = 0.42;
    const minBelowTop = 0.18;
    this.getLevelGrid(def).queryRadius(x, z, 2.5, this.scratchPlats, null);
    let bestPlat: PlatformDef | null = null;
    let bestTop = -Infinity;
    for (let i = 0; i < this.scratchPlats.length; i++) {
      const plat = def.platforms[this.scratchPlats[i]];
      if (!plat || plat.kind === 'disappearing') continue;
      const [pw, ph, pd] = plat.size;
      let px = plat.position[0];
      let py = plat.position[1];
      let pz = plat.position[2];
      if (plat.kind === 'moving' && plat.moveOffset) {
        const speed = plat.moveSpeed ?? 1.0;
        const phase = plat.movePhase ?? 0;
        const factor = 0.5 - 0.5 * Math.cos(elapsedSec * speed + phase);
        px += plat.moveOffset[0] * factor;
        py += plat.moveOffset[1] * factor;
        pz += plat.moveOffset[2] * factor;
      }
      const topY = py + ph * 0.5;
      if (y < topY - minBelowTop || y > topY + 0.55 || y < def.bounds.waterY - 0.06) continue;
      if (plat.rotationSpeed) {
        const [lx, , lz] = worldToRotatedPlatformLocal(x - px, 0, z - pz, plat, elapsedSec);
        if (Math.abs(lx) <= pw * 0.5 + margin && Math.abs(lz) <= pd * 0.5 + margin && topY > bestTop) {
          bestTop = topY;
          bestPlat = plat;
        }
      } else if (Math.abs(x - px) <= pw * 0.5 + margin && Math.abs(z - pz) <= pd * 0.5 + margin && topY > bestTop) {
        bestTop = topY;
        bestPlat = plat;
      }
    }
    return bestPlat;
  }

  private computeServerPlatformEdgeSpot(
    def: LevelDef,
    m: InternalMonster,
    plat: PlatformDef,
    playerX: number,
    playerZ: number,
    monsterRadius: number,
    botY: number,
    topY: number,
    elapsedSec: number
  ): { x: number; z: number } {
    const [pw, , pd] = plat.size;
    let px = plat.position[0];
    let pz = plat.position[2];
    if (plat.kind === 'moving' && plat.moveOffset) {
      const factor = 0.5 - 0.5 * Math.cos(elapsedSec * (plat.moveSpeed ?? 1.0) + (plat.movePhase ?? 0));
      px += plat.moveOffset[0] * factor;
      pz += plat.moveOffset[2] * factor;
    }
    const hw = pw * 0.5;
    const hd = pd * 0.5;
    const clearance = monsterRadius + 0.62 * m.scale;
    const axis = plat.rotationAxis || 'y';
    const isRotY = Boolean(plat.rotationSpeed && axis === 'y');
    const ang = isRotY ? (plat.rotationPhase ?? 0) + elapsedSec * (plat.rotationSpeed || 0) : 0;
    const cosA = Math.cos(ang);
    const sinA = Math.sin(ang);

    let mlx = m.x - px;
    let mlz = m.z - pz;
    let plx = playerX - px;
    let plz = playerZ - pz;
    if (isRotY) {
      const [rlxM, , rlzM] = worldToRotatedPlatformLocal(mlx, 0, mlz, plat, elapsedSec);
      mlx = rlxM;
      mlz = rlzM;
      const [rlxP, , rlzP] = worldToRotatedPlatformLocal(plx, 0, plz, plat, elapsedSec);
      plx = rlxP;
      plz = rlzP;
    }
    plx = clamp(plx, -hw * 0.82, hw * 0.82);
    plz = clamp(plz, -hd * 0.82, hd * 0.82);

    const candsX = [
      plx,
      plx,
      -hw - clearance,
      hw + clearance,
      0,
      0,
      -hw - clearance,
      hw + clearance,
      mlx >= 0 ? hw + clearance : -hw - clearance,
    ];
    const candsZ = [
      -hd - clearance,
      hd + clearance,
      plz,
      plz,
      -hd - clearance,
      hd + clearance,
      0,
      0,
      mlz >= 0 ? hd + clearance : -hd - clearance,
    ];

    let bestScore = Infinity;
    let bestWx = m.x;
    let bestWz = m.z;
    let found = false;
    for (let c = 0; c < 9; c++) {
      const lx = candsX[c];
      const lz = candsZ[c];
      const wx = px + (isRotY ? lx * cosA + lz * sinA : lx);
      const wz = pz + (isRotY ? -lx * sinA + lz * cosA : lz);
      if (this.isCircleBlocked(def, wx, wz, monsterRadius * 0.92, botY, topY, elapsedSec)) {
        continue;
      }
      const dMon = Math.hypot(wx - m.x, wz - m.z);
      const dPly = Math.hypot(wx - playerX, wz - playerZ);
      const score = dMon * 0.55 + dPly;
      if (score < bestScore) {
        bestScore = score;
        bestWx = wx;
        bestWz = wz;
        found = true;
      }
    }
    if (!found) {
      const dx = m.x - px;
      const dz = m.z - pz;
      const len = Math.hypot(dx, dz) || 1;
      const fallbackDist = Math.max(hw, hd) + clearance;
      bestWx = clamp(px + (dx / len) * fallbackDist, def.bounds.minX + monsterRadius + 0.5, def.bounds.maxX - monsterRadius - 0.5);
      bestWz = clamp(pz + (dz / len) * fallbackDist, def.bounds.minZ + monsterRadius + 0.5, def.bounds.maxZ - monsterRadius - 0.5);
    }
    return { x: bestWx, z: bestWz };
  }

  private pickServerIntimidationYaw(
    def: LevelDef,
    m: InternalMonster,
    targetX: number,
    targetZ: number,
    monsterRadius: number,
    botY: number,
    topY: number,
    elapsedSec: number
  ): number {
    const toPlayerYaw = Math.atan2(targetX - m.x, targetZ - m.z);
    const candA = toPlayerYaw + 0.7;
    const candB = toPlayerYaw - 0.7;
    const checkTailClear = (yaw: number): boolean => {
      const bx = -Math.sin(yaw);
      const bz = -Math.cos(yaw);
      return (
        !this.isCircleBlocked(def, m.x + bx * 4.0 * m.scale, m.z + bz * 4.0 * m.scale, monsterRadius * 0.75, botY, topY, elapsedSec) &&
        !this.isCircleBlocked(def, m.x + bx * 8.0 * m.scale, m.z + bz * 8.0 * m.scale, monsterRadius * 0.7, botY, topY, elapsedSec)
      );
    };
    const clearA = checkTailClear(candA);
    const clearB = checkTailClear(candB);
    if (clearA && !clearB) return candA;
    if (clearB && !clearA) return candB;
    return Math.abs(shortestAngleDiff(m.yaw, candA)) <= Math.abs(shortestAngleDiff(m.yaw, candB)) ? candA : candB;
  }

  private pickServerRetreatTarget(
    def: LevelDef,
    m: InternalMonster,
    awayFromX: number,
    awayFromZ: number,
    monsterRadius: number,
    botY: number,
    topY: number,
    elapsedSec: number
  ): { x: number; z: number; yaw: number } {
    const awayYaw = Math.atan2(m.x - awayFromX, m.z - awayFromZ);
    const offsets = [0, 0.45, -0.45, 0.9, -0.9, 1.35, -1.35];
    const dists = [16, 12, 9, 6];
    let bestYaw = awayYaw;
    let bestClear = -1;
    for (let k = 0; k < offsets.length; k++) {
      const ang = awayYaw + offsets[k];
      const sx = Math.sin(ang);
      const sz = Math.cos(ang);
      let clearDist = 0;
      for (let dIdx = 0; dIdx < dists.length; dIdx++) {
        const dist = dists[dIdx];
        if (!this.isCircleBlocked(def, m.x + sx * dist, m.z + sz * dist, monsterRadius * 0.95, botY, topY, elapsedSec)) {
          clearDist = dist;
        } else {
          break;
        }
      }
      if (clearDist > bestClear) {
        bestClear = clearDist;
        bestYaw = ang;
        if (clearDist >= 16) break;
      }
    }
    const targetDist = Math.max(10, bestClear > 0 ? bestClear : 14);
    return {
      x: clamp(m.x + Math.sin(bestYaw) * targetDist, def.bounds.minX + monsterRadius + 0.5, def.bounds.maxX - monsterRadius - 0.5),
      z: clamp(m.z + Math.cos(bestYaw) * targetDist, def.bounds.minZ + monsterRadius + 0.5, def.bounds.maxZ - monsterRadius - 0.5),
      yaw: bestYaw,
    };
  }

  private isCircleBlocked(
    def: LevelDef,
    x: number,
    z: number,
    radius: number,
    botY: number,
    topY: number,
    elapsedSec = 0,
    checkWalls = true,
    checkPlats = true
  ): boolean {
    const b = def.bounds;
    if (checkWalls && (x - radius < b.minX || x + radius > b.maxX || z - radius < b.minZ || z + radius > b.maxZ)) {
      return true;
    }
    const rSq = radius * radius;
    this.getLevelGrid(def).queryRadius(x, z, radius + 0.5, this.scratchPlats, this.scratchWalls);
    if (checkWalls) {
      for (let i = 0; i < this.scratchWalls.length; i++) {
        const w = def.walls[this.scratchWalls[i]];
        const [wx, wy, wz] = w.position;
        const [ww, wh, wd] = w.size;
        if (topY <= wy - wh * 0.5 + 0.05 || botY >= wy + wh * 0.5 - 0.05) continue;
        const cx = clamp(x, wx - ww * 0.5, wx + ww * 0.5);
        const cz = clamp(z, wz - wd * 0.5, wz + wd * 0.5);
        const dx = x - cx;
        const dz = z - cz;
        if (dx * dx + dz * dz < rSq) return true;
      }
    }
    if (!checkPlats) return false;
    for (let i = 0; i < this.scratchPlats.length; i++) {
      const plat = def.platforms[this.scratchPlats[i]];
      if (plat.kind === 'disappearing') continue;
      let [px, py, pz] = plat.position;
      if (plat.kind === 'moving' && plat.moveOffset) {
        const speed = plat.moveSpeed ?? 1.0;
        const phase = plat.movePhase ?? 0;
        const factor = 0.5 - 0.5 * Math.cos(elapsedSec * speed + phase);
        px += plat.moveOffset[0] * factor;
        py += plat.moveOffset[1] * factor;
        pz += plat.moveOffset[2] * factor;
      }
      const [pw, ph, pd] = plat.size;
      const hw = pw * 0.5;
      const hh = ph * 0.5;
      const hd = pd * 0.5;

      if (!plat.rotationSpeed) {
        if (topY <= py - hh + 0.05 || botY >= py + hh - 0.05) continue;
        const cx = clamp(x, px - hw, px + hw);
        const cz = clamp(z, pz - hd, pz + hd);
        const dx = x - cx;
        const dz = z - cz;
        if (dx * dx + dz * dz < rSq) return true;
      } else if ((plat.rotationAxis || 'y') === 'y') {
        if (topY <= py - hh + 0.05 || botY >= py + hh - 0.05) continue;
        const [lx, , lz] = worldToRotatedPlatformLocal(x - px, 0, z - pz, plat, elapsedSec);
        const dlx = lx - clamp(lx, -hw, hw);
        const dlz = lz - clamp(lz, -hd, hd);
        if (dlx * dlx + dlz * dlz < rSq) return true;
      } else {
        const spanY = Math.max(0.1, topY - botY);
        const padY = Math.min(radius * 0.5, spanY * 0.25);
        const yMin = botY + padY;
        const yMax = topY - padY;
        for (let sIdx = 0; sIdx < 4; sIdx++) {
          const sy = yMin + ((yMax - yMin) * sIdx) / 3;
          const [lx, ly, lz] = worldToRotatedPlatformLocal(x - px, sy - py, z - pz, plat, elapsedSec);
          const dlx = lx - clamp(lx, -hw, hw);
          const dly = ly - clamp(ly, -hh, hh);
          const dlz = lz - clamp(lz, -hd, hd);
          if (dlx * dlx + dly * dly + dlz * dlz < rSq) return true;
        }
      }
    }
    return false;
  }

  private resolveMonsterCircleSolids(
    def: LevelDef,
    m: InternalMonster,
    radius: number,
    botY: number,
    topY: number,
    elapsedSec: number
  ): void {
    const b = def.bounds;
    const rSq = radius * radius;
    const grid = this.getLevelGrid(def);

    for (let iter = 0; iter < 2; iter++) {
      grid.queryRadius(m.x, m.z, radius + 0.5, this.scratchPlats, this.scratchWalls);
      for (let i = 0; i < this.scratchPlats.length; i++) {
        const plat = def.platforms[this.scratchPlats[i]];
        if (plat.kind === 'disappearing') continue;
        let [px, py, pz] = plat.position;
        if (plat.kind === 'moving' && plat.moveOffset) {
          const speed = plat.moveSpeed ?? 1.0;
          const phase = plat.movePhase ?? 0;
          const factor = 0.5 - 0.5 * Math.cos(elapsedSec * speed + phase);
          px += plat.moveOffset[0] * factor;
          py += plat.moveOffset[1] * factor;
          pz += plat.moveOffset[2] * factor;
        }
        const [pw, ph, pd] = plat.size;
        const hw = pw * 0.5;
        const hh = ph * 0.5;
        const hd = pd * 0.5;
        const axis = plat.rotationAxis || 'y';

        if (!plat.rotationSpeed) {
          if (topY <= py - hh + 0.05 || botY >= py + hh - 0.05) continue;
          const lx = m.x - px;
          const lz = m.z - pz;
          const dlx = lx - clamp(lx, -hw, hw);
          const dlz = lz - clamp(lz, -hd, hd);
          const dSq = dlx * dlx + dlz * dlz;
          if (dSq < rSq) {
            let wx = 0;
            let wz = 0;
            if (dSq > 1e-8) {
              const d = Math.sqrt(dSq);
              const pen = radius - d;
              wx = (dlx / d) * pen;
              wz = (dlz / d) * pen;
            } else {
              const penX = hw + radius - Math.abs(lx);
              const penZ = hd + radius - Math.abs(lz);
              if (penX < penZ) wx = lx >= 0 ? penX : -penX;
              else wz = lz >= 0 ? penZ : -penZ;
            }
            m.x += wx;
            m.z += wz;
            const wLen = Math.hypot(wx, wz);
            if (wLen > 1e-6) {
              const nx = wx / wLen;
              const nz = wz / wLen;
              const vn = m.velX * nx + m.velZ * nz;
              if (vn < 0) {
                m.velX -= vn * nx;
                m.velZ -= vn * nz;
              }
            }
          }
        } else if (axis === 'y') {
          if (topY <= py - hh + 0.05 || botY >= py + hh - 0.05) continue;
          const ang = (plat.rotationPhase ?? 0) + elapsedSec * plat.rotationSpeed;
          const [lx, , lz] = worldToRotatedPlatformLocal(m.x - px, 0, m.z - pz, plat, elapsedSec);
          const dlx = lx - clamp(lx, -hw, hw);
          const dlz = lz - clamp(lz, -hd, hd);
          const dSq = dlx * dlx + dlz * dlz;
          if (dSq < rSq) {
            let lwx = 0;
            let lwz = 0;
            if (dSq > 1e-8) {
              const d = Math.sqrt(dSq);
              const pen = radius - d;
              lwx = (dlx / d) * pen;
              lwz = (dlz / d) * pen;
            } else {
              const penX = hw + radius - Math.abs(lx);
              const penZ = hd + radius - Math.abs(lz);
              if (penX < penZ) lwx = lx >= 0 ? penX : -penX;
              else lwz = lz >= 0 ? penZ : -penZ;
            }
            const cy = Math.cos(ang);
            const sy = Math.sin(ang);
            const wx = lwx * cy + lwz * sy;
            const wz = -lwx * sy + lwz * cy;
            m.x += wx;
            m.z += wz;
            const wLen = Math.hypot(wx, wz);
            if (wLen > 1e-6) {
              const nx = wx / wLen;
              const nz = wz / wLen;
              const vn = m.velX * nx + m.velZ * nz;
              if (vn < 0) {
                m.velX -= vn * nx;
                m.velZ -= vn * nz;
              }
            }
          }
        } else {
          const spanY = Math.max(0.1, topY - botY);
          const padY = Math.min(radius * 0.5, spanY * 0.25);
          const yMin = botY + padY;
          const yMax = topY - padY;
          const spd = plat.rotationSpeed;
          const phase = plat.rotationPhase ?? 0;
          const ry = phase + elapsedSec * spd;
          const rx = axis === 'x' ? ry : axis === 'xy' ? phase * 0.7 + elapsedSec * spd * 0.55 : 0;
          const rz = axis === 'z' ? ry : axis === 'zy' ? phase * 0.7 + elapsedSec * spd * 0.55 : 0;
          const ey = axis === 'x' || axis === 'z' ? 0 : ry;

          let bestDSq = rSq;
          let bestLwx = 0;
          let bestLwz = 0;
          for (let sIdx = 0; sIdx < 4; sIdx++) {
            const sy = yMin + ((yMax - yMin) * sIdx) / 3;
            const [lx, ly, lz] = worldToRotatedPlatformLocal(m.x - px, sy - py, m.z - pz, plat, elapsedSec);
            const dlx = lx - clamp(lx, -hw, hw);
            const dly = ly - clamp(ly, -hh, hh);
            const dlz = lz - clamp(lz, -hd, hd);
            const dSq = dlx * dlx + dly * dly + dlz * dlz;
            if (dSq < bestDSq) {
              bestDSq = dSq;
              if (dSq > 1e-8) {
                const d = Math.sqrt(dSq);
                const pen = radius - d;
                bestLwx = (dlx / d) * pen;
                bestLwz = (dlz / d) * pen;
              } else {
                const penX = hw + radius - Math.abs(lx);
                const penZ = hd + radius - Math.abs(lz);
                bestLwx = penX < penZ ? (lx >= 0 ? penX : -penX) : 0;
                bestLwz = penX < penZ ? 0 : (lz >= 0 ? penZ : -penZ);
              }
            }
          }
          if (bestDSq < rSq) {
            const cz = Math.cos(rz);
            const sz = Math.sin(rz);
            const x1 = bestLwx * cz;
            const y1 = bestLwx * sz;
            const z1 = bestLwz;
            const cx = Math.cos(rx);
            const sx = Math.sin(rx);
            const x2 = x1;
            const z2 = y1 * sx + z1 * cx;
            const cy = Math.cos(ey);
            const sy2 = Math.sin(ey);
            const wx = x2 * cy + z2 * sy2;
            const wz = -x2 * sy2 + z2 * cy;
            m.x += wx;
            m.z += wz;
            const wLen = Math.hypot(wx, wz);
            if (wLen > 1e-6) {
              const nx = wx / wLen;
              const nz = wz / wLen;
              const vn = m.velX * nx + m.velZ * nz;
              if (vn < 0) {
                m.velX -= vn * nx;
                m.velZ -= vn * nz;
              }
            }
          }
        }
      }

      for (let i = 0; i < this.scratchWalls.length; i++) {
        const w = def.walls[this.scratchWalls[i]];
        const [ww, wh, wd] = w.size;
        const [wxPos, wyPos, wzPos] = w.position;
        if (topY <= wyPos - wh * 0.5 + 0.05 || botY >= wyPos + wh * 0.5 - 0.05) continue;
        const cx = clamp(m.x, wxPos - ww * 0.5, wxPos + ww * 0.5);
        const cz = clamp(m.z, wzPos - wd * 0.5, wzPos + wd * 0.5);
        const dx = m.x - cx;
        const dz = m.z - cz;
        const dSq = dx * dx + dz * dz;
        if (dSq < rSq) {
          let wx = 0;
          let wz = 0;
          if (dSq > 1e-8) {
            const d = Math.sqrt(dSq);
            const pen = radius - d;
            wx = (dx / d) * pen;
            wz = (dz / d) * pen;
          } else {
            const dLeft = m.x - (wxPos - ww * 0.5 - radius);
            const dRight = wxPos + ww * 0.5 + radius - m.x;
            const dFront = m.z - (wzPos - wd * 0.5 - radius);
            const dBack = wzPos + wd * 0.5 + radius - m.z;
            const minPen = Math.min(dLeft, dRight, dFront, dBack);
            if (minPen === dLeft) wx = -dLeft;
            else if (minPen === dRight) wx = dRight;
            else if (minPen === dFront) wz = -dFront;
            else wz = dBack;
          }
          m.x += wx;
          m.z += wz;
          const wLen = Math.hypot(wx, wz);
          if (wLen > 1e-6) {
            const nx = wx / wLen;
            const nz = wz / wLen;
            const vn = m.velX * nx + m.velZ * nz;
            if (vn < 0) {
              m.velX -= vn * nx;
              m.velZ -= vn * nz;
            }
          }
        }
      }

      m.x = clamp(m.x, b.minX + radius + 0.15, b.maxX - radius - 0.15);
      m.z = clamp(m.z, b.minZ + radius + 0.15, b.maxZ - radius - 0.15);
    }
  }

  private buildMonstersForLevel(def: LevelDef): InternalMonster[] {
    const waterY = def.bounds.waterY;
    return def.monsters.map((mc, idx) => {
      const startWp = mc.waypoints[0] || [0, -25];
      const nextWp = mc.waypoints[1] || startWp;
      const initYaw = Math.atan2(nextWp[0] - startWp[0], nextWp[1] - startWp[1]) || 0;
      return {
        id: mc.id,
        x: startWp[0],
        y: waterY - 0.14,
        z: startWp[1],
        yaw: initYaw,
        velX: 0,
        velZ: 0,
        alert: 'calm',
        targetUserId: null,
        emerged: false,
        wpIndex: 1 % Math.max(1, mc.waypoints.length),
        eatingUserId: null,
        eatingProgress: 0,
        eatingStartAt: 0,
        eatingUntil: 0,
        waypoints: mc.waypoints,
        patrolSpeed: mc.patrolSpeed,
        chaseSpeed: mc.chaseSpeed,
        waterDetectionRadius: mc.waterDetectionRadius,
        scale: mc.scale || 1.0,
        avoidDir: idx % 2 === 0 ? 1 : -1,
        avoidTimer: 0,
        stuckTimer: 0,
        navBestDist: Infinity,
        navStallTimer: 0,
        navAnchorX: startWp[0],
        navAnchorZ: startWp[1],
        navAnchorTimer: 0,
        navTurnAccum: 0,
        watchingUserId: null,
        lookAway: false,
        intimPhase: null,
        intimProgress: 0,
        visionState: 'none',
        visionUntil: 0,
        lookAwayYaw: initYaw,
        watchAnchorX: 0,
        watchAnchorZ: 0,
        watchGraceUntil: 0,
        intimStartedAt: 0,
        intimDurationMs: 0,
        platformSpotX: 0,
        platformSpotZ: 0,
        intimYaw: initYaw,
      };
    });
  }

  private ensureLevelSim(room: InternalRoom, levelId: number): InternalLevelSim {
    let sim = room.levelSims.get(levelId);
    if (!sim) {
      const def = getLevelDef(levelId);
      sim = {
        levelId,
        levelDef: def,
        monsters: this.buildMonstersForLevel(def),
      };
      room.levelSims.set(levelId, sim);
    }
    return sim;
  }

  private clearPlayerFromLevelMonsters(sim: InternalLevelSim, userId: string): void {
    for (const m of sim.monsters) {
      if (m.eatingUserId === userId) {
        m.eatingUserId = null;
        m.eatingStartAt = 0;
        m.eatingUntil = 0;
        m.eatingProgress = 0;
      }
      if (m.watchingUserId === userId) {
        m.watchingUserId = null;
      }
      if (m.targetUserId === userId) {
        m.targetUserId = null;
        m.alert = 'calm';
        m.emerged = false;
        m.visionState = 'none';
        m.visionUntil = 0;
        m.watchingUserId = null;
        m.lookAway = false;
        m.watchGraceUntil = 0;
        m.intimPhase = null;
        m.intimProgress = 0;
        m.intimStartedAt = 0;
        m.intimDurationMs = 0;
      }
    }
  }

  private pruneEmptyLevelSims(room: InternalRoom): void {
    if (room.status !== 'playing') return;
    for (const levelId of room.levelSims.keys()) {
      if (levelId !== room.levelId) {
        room.levelSims.delete(levelId);
      }
    }
  }

  private placePlayerOnLevelSpawn(
    def: LevelDef,
    p: InternalRoomPlayer,
    slotIdx = 0,
    totalOnDeck = 1
  ): void {
    const startPlat = def.platforms[0];
    const deckCenterX = startPlat ? startPlat.position[0] : def.spawn[0];
    const deckCenterZ = startPlat ? startPlat.position[2] : def.spawn[2];
    const usableW = Math.max(1.8, (startPlat ? startPlat.size[0] : 6.4) - 1.8);
    const usableD = Math.max(1.8, (startPlat ? startPlat.size[2] : 5.4) - 1.8);
    if (totalOnDeck <= 1) {
      p.x = def.spawn[0];
      p.y = def.spawn[1];
      p.z = def.spawn[2];
    } else {
      const cols = Math.min(4, Math.max(1, Math.ceil(Math.sqrt(totalOnDeck * (usableW / usableD)))));
      const rows = Math.max(1, Math.ceil(totalOnDeck / cols));
      const col = slotIdx % cols;
      const row = Math.floor(slotIdx / cols);
      const colsInThisRow = row === rows - 1 ? totalOnDeck - row * cols : cols;
      const fx = colsInThisRow <= 1 ? 0 : col / (colsInThisRow - 1) - 0.5;
      const fz = rows <= 1 ? 0 : row / (rows - 1) - 0.5;
      p.x = deckCenterX + fx * usableW;
      p.y = def.spawn[1];
      p.z = deckCenterZ + fz * usableD;
    }
    p.yaw = Math.PI;
    p.animState = 'idle';
    p.horizSpeed = 0;
    p.velX = 0;
    p.velY = 0;
    p.velZ = 0;
    p.inWater = false;
  }

  private handlePlayerReachedPortal(
    room: InternalRoom,
    p: InternalRoomPlayer,
    currentDef: LevelDef,
    now: number,
    batch: RoomDeliveryBatch
  ): void {
    if (room.status !== 'playing' || room.matchResult) return;
    if (!p.connected || !p.alive || p.transitioning || p.escaped || p.levelId !== room.levelId) return;

    const curSim = room.levelSims.get(room.levelId);
    if (curSim) {
      this.clearPlayerFromLevelMonsters(curSim, p.userId);
    }

    p.escaped = true;
    p.inWater = false;
    p.horizSpeed = 0;
    p.velX = 0;
    p.velY = 0;
    p.velZ = 0;
    p.animState = 'idle';
    p.beingEatenUntil = 0;
    p.beingEatenByMonsterId = null;
    p.moveAttemptWhileWatched = false;
    p.wasInWaterSession = false;
    p.waterEpoch = 0;
    p.pendingPlatformWatchEpoch = 0;
    p.watchHandledEpoch = 0;
    p.lastUpdateAt = now;
    this.dirtyPlayersSinceBroadcast = true;

    const ev: GamePlayerEvent = {
      userId: p.userId,
      username: p.username,
      kind: 'escaped',
      levelId: currentDef.id,
      x: r3(p.x),
      y: r3(p.y),
      z: r3(p.z),
    };
    this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
    const summary = this.serializeRoom(room);
    batch.roomSummary = summary;
    this.pushRoomBroadcast(batch, room, 'room:state', summary);
    this.evaluateTeamPortalProgression(room, now, batch, p);
  }

  private evaluateTeamPortalProgression(
    room: InternalRoom,
    now: number,
    batch: RoomDeliveryBatch,
    triggeringPlayer?: InternalRoomPlayer
  ): void {
    if (room.status !== 'playing' || room.matchResult) return;

    let connectedCount = 0;
    let readyCount = 0;
    const totalCount = room.players.size;
    for (const pl of room.players.values()) {
      if (pl.connected) connectedCount++;
      if (pl.connected && pl.alive && pl.escaped && !pl.transitioning && pl.levelId === room.levelId) {
        readyCount++;
      }
    }

    if (connectedCount === 0 || totalCount === 0) {
      this.finalizeMatch(room, 'all_caught', batch);
      return;
    }

    if (readyCount < totalCount) {
      return;
    }

    const prevLevelId = room.levelId;
    if (prevLevelId >= TOTAL_LEVELS) {
      this.finalizeMatch(room, 'all_resolved', batch);
      return;
    }

    const nextLevelId = prevLevelId + 1;
    const nextDef = getLevelDef(nextLevelId);
    room.levelSims.clear();
    room.levelId = nextLevelId;
    room.levelDef = nextDef;
    room.timeRemaining = nextDef.timeLimit;

    const nextSim = this.ensureLevelSim(room, nextLevelId);
    room.monsters = nextSim.monsters;

    const allPlayers = Array.from(room.players.values());
    allPlayers.forEach((pl, idx) => {
      pl.levelId = nextLevelId;
      pl.transitioning = true;
      pl.transitioningUntil = now + LEVEL_TRANSITION_GRACE_MS;
      pl.checkpointIndex = 0;
      pl.alive = true;
      pl.escaped = false;
      pl.respawnAt = 0;
      pl.beingEatenUntil = 0;
      pl.beingEatenByMonsterId = null;
      pl.moveAttemptWhileWatched = false;
      pl.wasInWaterSession = false;
      pl.waterEpoch = 0;
      pl.pendingPlatformWatchEpoch = 0;
      pl.watchHandledEpoch = 0;
      pl.levelTimeRemaining = nextDef.timeLimit;
      pl.pushImmunityUntil = now + 1200;
      this.placePlayerOnLevelSpawn(nextDef, pl, idx, allPlayers.length);
      pl.lastUpdateAt = now;
    });
    this.dirtyPlayersSinceBroadcast = true;

    const elapsedSec = room.matchStartTime ? Math.max(0, (now - room.matchStartTime) / 1000) : 0;
    const levelChangePayload: GameLevelChangedPayload = {
      roomId: room.id,
      levelId: nextLevelId,
      prevLevelId,
      timeLimit: nextDef.timeLimit,
      elapsedSec,
      players: this.serializeNetPlayers(room),
      monsters: this.serializeNetMonstersList(nextSim.monsters),
    };
    this.pushRoomBroadcast(batch, room, 'game:level_changed', levelChangePayload);

    const actor = triggeringPlayer || allPlayers.find((pl) => pl.userId === room.hostId) || allPlayers[0];
    if (actor) {
      const advEv: GamePlayerEvent = {
        userId: actor.userId,
        username: actor.username,
        kind: 'level_advanced',
        levelId: nextLevelId,
        prevLevelId,
        x: r3(actor.x),
        y: r3(actor.y),
        z: r3(actor.z),
      };
      this.pushRoomBroadcast(batch, room, 'game:player_event', advEv);
    }

    const summary = this.serializeRoom(room);
    batch.roomSummary = summary;
    this.pushRoomBroadcast(batch, room, 'room:state', summary);
  }

  private respawnPlayerAtLevelStart(
    room: InternalRoom,
    p: InternalRoomPlayer,
    now: number,
    batch: RoomDeliveryBatch
  ): void {
    p.levelId = room.levelId;
    const def = getLevelDef(room.levelId);
    const sim = this.ensureLevelSim(room, room.levelId);
    this.clearPlayerFromLevelMonsters(sim, p.userId);

    const peersOnLevel = Array.from(room.players.values()).filter(
      (pl) => pl.connected && pl.levelId === room.levelId
    );
    const slotIdx = Math.max(0, peersOnLevel.findIndex((pl) => pl.userId === p.userId));
    this.placePlayerOnLevelSpawn(def, p, slotIdx, Math.max(1, peersOnLevel.length));

    p.alive = true;
    p.escaped = false;
    p.transitioning = false;
    p.transitioningUntil = 0;
    p.respawnAt = 0;
    p.checkpointIndex = 0;
    p.levelTimeRemaining = def.timeLimit;
    p.beingEatenUntil = 0;
    p.beingEatenByMonsterId = null;
    p.moveAttemptWhileWatched = false;
    p.wasInWaterSession = false;
    p.waterEpoch = 0;
    p.pendingPlatformWatchEpoch = 0;
    p.watchHandledEpoch = 0;
    p.pushImmunityUntil = now + 1200;
    p.lastUpdateAt = now;
    this.dirtyPlayersSinceBroadcast = true;

    const ev: GamePlayerEvent = {
      userId: p.userId,
      username: p.username,
      kind: 'respawned',
      levelId: room.levelId,
      x: r3(p.x),
      y: r3(p.y),
      z: r3(p.z),
      yaw: r3(p.yaw),
    };
    this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
  }

  private updateLevelSimMonsters(
    room: InternalRoom,
    sim: InternalLevelSim,
    dt: number,
    now: number,
    batch: RoomDeliveryBatch
  ): void {
    const def = sim.levelDef;
    const monsters = sim.monsters;
    const waterY = def.bounds.waterY;
    const elapsedSec = room.matchStartTime ? Math.max(0, (now - room.matchStartTime) / 1000) : 0;
    const aliveInWater = this.aliveInWaterScratch;
    aliveInWater.length = 0;
    for (const p of room.players.values()) {
      if (
        p.levelId === sim.levelId &&
        !p.transitioning &&
        p.alive &&
        !p.escaped &&
        p.connected &&
        p.inWater &&
        (!p.beingEatenUntil || now >= p.beingEatenUntil)
      ) {
        aliveInWater.push(p);
      }
    }
    const claimedTargets = this.claimedTargetsScratch;
    claimedTargets.clear();

    // Guarantee EVERY swimmer in water is immediately chased by the nearest available serpent
    // (deprioritizing serpents actively watching another player on a platform if a free serpent exists)
    const forcedChaseByMonster = new Map<number, InternalRoomPlayer>();
    if (aliveInWater.length > 0 && monsters.length > 0) {
      const assignedMonsters = new Set<number>();
      for (const swimmer of aliveInWater) {
        let bestIdx = -1;
        let bestScore = Infinity;
        for (let mi = 0; mi < monsters.length; mi++) {
          if (assignedMonsters.has(mi)) continue;
          const mm = monsters[mi];
          if (mm.eatingUntil && mm.eatingUserId && now < mm.eatingUntil) continue;
          const isBusyWatchingOther =
            mm.targetUserId !== swimmer.userId &&
            (mm.visionState === 'platform_approaching' ||
              mm.visionState === 'platform_rising' ||
              mm.visionState === 'staring' ||
              mm.visionState === 'platform_lowering' ||
              mm.visionState === 'platform_lunging');
          const d = Math.hypot(swimmer.x - mm.x, swimmer.z - mm.z) + (isBusyWatchingOther ? 500 : 0);
          if (d < bestScore) {
            bestScore = d;
            bestIdx = mi;
          }
        }
        if (bestIdx >= 0) {
          assignedMonsters.add(bestIdx);
          forcedChaseByMonster.set(bestIdx, swimmer);
          claimedTargets.add(swimmer.userId);
        }
      }
    }

    // Assign exactly ONE serpent per player who just exited water onto a platform (independent per player)
    for (const p of room.players.values()) {
      if (
        p.levelId !== sim.levelId ||
        p.transitioning ||
        !p.alive ||
        p.escaped ||
        !p.connected ||
        p.inWater ||
        (p.beingEatenUntil && now < p.beingEatenUntil)
      ) {
        continue;
      }
      const pendingEpoch = p.pendingPlatformWatchEpoch || 0;
      const handledEpoch = p.watchHandledEpoch || 0;
      if (pendingEpoch <= handledEpoch) continue;

      const alreadyWatchedByAny = monsters.some(
        (mm) =>
          mm.targetUserId === p.userId &&
          (mm.visionState === 'platform_approaching' ||
            mm.visionState === 'platform_rising' ||
            mm.visionState === 'staring' ||
            mm.visionState === 'platform_lunging')
      );
      if (alreadyWatchedByAny) {
        p.watchHandledEpoch = pendingEpoch;
        continue;
      }

      let bestFreeIdx = -1;
      let bestFreeDist = Infinity;
      for (let mi = 0; mi < monsters.length; mi++) {
        if (forcedChaseByMonster.has(mi)) continue;
        const mm = monsters[mi];
        if (mm.eatingUntil && mm.eatingUserId && now < mm.eatingUntil) continue;
        if (
          mm.visionState === 'platform_approaching' ||
          mm.visionState === 'platform_rising' ||
          mm.visionState === 'staring' ||
          mm.visionState === 'platform_lowering' ||
          mm.visionState === 'platform_lunging'
        ) {
          continue;
        }
        const d = Math.hypot(p.x - mm.x, p.z - mm.z);
        if (d < bestFreeDist) {
          bestFreeDist = d;
          bestFreeIdx = mi;
        }
      }

      if (bestFreeIdx >= 0) {
        const pm = monsters[bestFreeIdx];
        p.watchHandledEpoch = pendingEpoch;
        p.moveAttemptWhileWatched = false;
        const pmRadius = 0.88 * pm.scale;
        const pmBotY = waterY - 1.5;
        const pmTopY = waterY + 4.2 * pm.scale;
        const plat = this.findPlatformUnderPlayerServer(def, p.x, p.y, p.z, elapsedSec);
        const spot = plat
          ? this.computeServerPlatformEdgeSpot(def, pm, plat, p.x, p.z, pmRadius, pmBotY, pmTopY, elapsedSec)
          : { x: pm.x, z: pm.z };
        pm.targetUserId = p.userId;
        pm.watchingUserId = null;
        pm.lookAway = false;
        pm.alert = 'searching';
        pm.emerged = false;
        pm.attackHoldUntil = 0;
        pm.visionState = 'platform_approaching';
        pm.visionUntil = 0;
        pm.intimPhase = 'approaching';
        pm.intimProgress = 0;
        pm.intimStartedAt = now;
        pm.intimDurationMs = 9000;
        pm.platformSpotX = spot.x;
        pm.platformSpotZ = spot.z;
        resetSerpentNavProgress(pm);
      }
    }

    for (let i = 0; i < monsters.length; i++) {
      const m = monsters[i];

      if (m.eatingUntil && m.eatingUserId) {
        const victim = room.players.get(m.eatingUserId);
        if (!victim || !victim.connected || !victim.alive || victim.levelId !== sim.levelId || victim.transitioning) {
          m.eatingUserId = null;
          m.eatingStartAt = 0;
          m.eatingUntil = 0;
          m.eatingProgress = 0;
        } else if (now < m.eatingUntil) {
          m.alert = 'chasing';
          m.emerged = true;
          m.targetUserId = victim.userId;
          m.watchingUserId = null;
          m.lookAway = false;
          m.visionState = 'none';
          m.intimPhase = null;
          m.intimProgress = 0;
          m.velX *= 0.8;
          m.velZ *= 0.8;
          m.eatingProgress = clamp((now - (m.eatingStartAt || now)) / EAT_DURATION_MS, 0, 1);
          m.y += (waterY + 0.08 - m.y) * Math.min(1, 5.5 * dt);
          continue;
        } else {
          victim.alive = false;
          victim.respawnAt = now + RESPAWN_DELAY_MS;
          victim.deaths = (victim.deaths || 0) + 1;
          victim.beingEatenUntil = 0;
          victim.beingEatenByMonsterId = null;
          victim.moveAttemptWhileWatched = false;
          victim.wasInWaterSession = false;
          victim.pendingPlatformWatchEpoch = victim.waterEpoch || 0;
          victim.watchHandledEpoch = victim.waterEpoch || 0;
          this.dirtyPlayersSinceBroadcast = true;
          const ev: GamePlayerEvent = {
            userId: victim.userId,
            username: victim.username,
            kind: 'caught',
            levelId: victim.levelId,
            x: victim.x,
            y: victim.y,
            z: victim.z,
            monsterId: m.id,
          };
          this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
          m.eatingUserId = null;
          m.eatingStartAt = 0;
          m.eatingUntil = 0;
          m.eatingProgress = 0;
          m.attackHoldUntil = now + 450;
          m.visionState = 'none';
          m.visionUntil = 0;
          m.watchingUserId = null;
          m.lookAway = false;
          m.watchGraceUntil = 0;
          m.intimPhase = null;
          m.intimProgress = 0;
          m.intimStartedAt = 0;
          m.intimDurationMs = 0;
          m.velX = 0;
          m.velZ = 0;
          resetSerpentNavProgress(m);
        }
      }

      const forcedTarget = forcedChaseByMonster.get(i) || null;
      let target: InternalRoomPlayer | null = forcedTarget;
      let targetDist = forcedTarget ? Math.hypot(forcedTarget.x - m.x, forcedTarget.z - m.z) : Infinity;

      const isPlatformActiveState =
        m.visionState === 'platform_approaching' ||
        m.visionState === 'platform_rising' ||
        m.visionState === 'staring' ||
        m.visionState === 'platform_lowering' ||
        m.visionState === 'platform_lunging';

      if (!target && !isPlatformActiveState && aliveInWater.length > 0) {
        let unclaimedTarget: InternalRoomPlayer | null = null;
        let unclaimedDist = Infinity;
        let anyTarget: InternalRoomPlayer | null = null;
        let anyDist = Infinity;
        for (const p of aliveInWater) {
          const d = Math.hypot(p.x - m.x, p.z - m.z);
          if (d < anyDist) {
            anyDist = d;
            anyTarget = p;
          }
          if (!claimedTargets.has(p.userId) && d < unclaimedDist) {
            unclaimedDist = d;
            unclaimedTarget = p;
          }
        }
        if (unclaimedTarget && unclaimedDist <= m.waterDetectionRadius) {
          target = unclaimedTarget;
          targetDist = unclaimedDist;
        } else if (anyTarget && (anyDist <= m.waterDetectionRadius || i === 0)) {
          target = anyTarget;
          targetDist = anyDist;
        }
      }

      const shouldChaseWater = Boolean(target && (Boolean(forcedTarget) || targetDist <= m.waterDetectionRadius || i === 0));
      if (shouldChaseWater && target) {
        claimedTargets.add(target.userId);
      }
      const extremeChaseSpeed = Math.max(48.0, m.chaseSpeed * 7.2);
      const holdingStrike = Boolean(m.attackHoldUntil && now < m.attackHoldUntil);

      const monsterRadius = 0.88 * m.scale;
      const monsterBotY = waterY - 1.5;
      const monsterTopY = waterY + 4.2 * m.scale;
      this._cbDef = def;
      this._cbBotY = monsterBotY;
      this._cbTopY = monsterTopY;
      this._cbElapsed = elapsedSec;
      const checkMonsterBlocked = this.checkMonsterBlockedFn;

      if (shouldChaseWater && target) {
        // Water attack ALWAYS overrides any platform-watch/retreat: immediate chase & devour even if still
        m.attackHoldUntil = 0;
        const wasAlreadyLunging = m.visionState === 'lunging' && m.targetUserId === target.userId;
        m.visionState = 'lunging';
        m.visionUntil = 0;
        m.watchingUserId = null;
        m.lookAway = false;
        m.watchGraceUntil = 0;
        m.intimPhase = null;
        m.intimProgress = 0;
        m.intimStartedAt = 0;
        m.intimDurationMs = 0;
        m.alert = 'chasing';
        m.emerged = targetDist < 10.5 && m.y >= waterY - 0.55 * m.scale;
        m.targetUserId = target.userId;
        if (!wasAlreadyLunging) {
          const dx = target.x - m.x;
          const dz = target.z - m.z;
          const len = Math.hypot(dx, dz);
          if (len > 0.01) {
            const nx = dx / len;
            const nz = dz / len;
            m.yaw = Math.atan2(nx, nz);
            m.velX = nx * extremeChaseSpeed;
            m.velZ = nz * extremeChaseSpeed;
          }
          resetSerpentNavProgress(m);
        }
      } else if (isPlatformActiveState) {
        const platTarget = m.targetUserId ? room.players.get(m.targetUserId) || null : null;
        const validPlatTarget = Boolean(
          platTarget &&
            platTarget.levelId === sim.levelId &&
            !platTarget.transitioning &&
            platTarget.alive &&
            !platTarget.escaped &&
            platTarget.connected &&
            !platTarget.inWater &&
            (!platTarget.beingEatenUntil || now >= platTarget.beingEatenUntil)
        );

        if (!validPlatTarget || !platTarget) {
          m.visionState = 'none';
          m.visionUntil = 0;
          m.watchingUserId = null;
          m.lookAway = false;
          m.intimPhase = null;
          m.intimProgress = 0;
          m.targetUserId = null;
          m.alert = 'calm';
          m.emerged = false;
          selectClosestPatrolWaypoint(m, monsterRadius, checkMonsterBlocked);
        } else {
          target = platTarget;
          targetDist = Math.hypot(platTarget.x - m.x, platTarget.z - m.z);
          const intimElapsedMs = now - (m.intimStartedAt || now);
          const intimDurMs = Math.max(1, m.intimDurationMs || 1);

          if (m.visionState === 'platform_approaching') {
            m.alert = 'searching';
            m.watchingUserId = null;
            m.lookAway = false;
            m.emerged = false;
            m.intimPhase = 'approaching';
            m.intimProgress = clamp(intimElapsedMs / intimDurMs, 0, 1);

            const livePlat = this.findPlatformUnderPlayerServer(def, platTarget.x, platTarget.y, platTarget.z, elapsedSec);
            if (livePlat) {
              const spot = this.computeServerPlatformEdgeSpot(
                def,
                m,
                livePlat,
                platTarget.x,
                platTarget.z,
                monsterRadius,
                monsterBotY,
                monsterTopY,
                elapsedSec
              );
              m.platformSpotX = spot.x;
              m.platformSpotZ = spot.z;
            }
            const distToSpot = Math.hypot((m.platformSpotX ?? m.x) - m.x, (m.platformSpotZ ?? m.z) - m.z);
            if (distToSpot <= 1.25 * m.scale || (m.stuckTimer > 0.5 && distToSpot < 2.8 * m.scale)) {
              m.visionState = 'platform_rising';
              m.intimPhase = 'rising';
              m.intimStartedAt = now;
              m.intimDurationMs = Math.round(SERPENT_INTIMIDATION_RISE_SEC * 1000);
              m.intimProgress = 0;
              m.intimYaw = this.pickServerIntimidationYaw(
                def,
                m,
                platTarget.x,
                platTarget.z,
                monsterRadius,
                monsterBotY,
                monsterTopY,
                elapsedSec
              );
              m.watchingUserId = null;
              m.velX *= 0.15;
              m.velZ *= 0.15;
              m.stuckTimer = 0;
            } else if (intimElapsedMs >= intimDurMs) {
              const ret = this.pickServerRetreatTarget(def, m, platTarget.x, platTarget.z, monsterRadius, monsterBotY, monsterTopY, elapsedSec);
              m.platformSpotX = ret.x;
              m.platformSpotZ = ret.z;
              m.intimYaw = ret.yaw;
              m.visionState = 'platform_retreating';
              m.intimPhase = 'retreating';
              m.intimStartedAt = now;
              m.intimDurationMs = 2400;
              m.intimProgress = 0;
              m.alert = 'calm';
              m.watchingUserId = null;
              m.targetUserId = null;
            }
          } else if (m.visionState === 'platform_rising') {
            m.alert = 'searching';
            m.watchingUserId = null;
            m.lookAway = false;
            m.emerged = true;
            m.intimPhase = 'rising';
            m.intimProgress = clamp(intimElapsedMs / intimDurMs, 0, 1);
            m.velX *= Math.exp(-10 * dt);
            m.velZ *= Math.exp(-10 * dt);
            m.yaw = normalizeAngle(m.yaw + shortestAngleDiffNav(m.yaw, m.intimYaw ?? m.yaw) * Math.min(1, 5.5 * dt));

            if (intimElapsedMs >= intimDurMs) {
              // Serpent has arrived & risen -> NOW start fixed 5.0s watch + player freeze
              m.visionState = 'staring';
              m.intimPhase = 'staring';
              m.intimStartedAt = now;
              m.intimDurationMs = Math.round(SERPENT_WATCH_DURATION_SEC * 1000);
              m.intimProgress = 0;
              m.visionUntil = now + m.intimDurationMs;
              m.watchingUserId = platTarget.userId;
              m.watchGraceUntil = now + Math.round(SERPENT_WATCH_FREEZE_GRACE_SEC * 1000);
              m.watchAnchorX = platTarget.x;
              m.watchAnchorZ = platTarget.z;
              platTarget.moveAttemptWhileWatched = false;
              this.dirtyPlayersSinceBroadcast = true;
            }
          } else if (m.visionState === 'staring') {
            m.alert = 'searching';
            m.watchingUserId = platTarget.userId;
            m.lookAway = false;
            m.emerged = true;
            m.intimPhase = 'staring';
            m.intimProgress = clamp(intimElapsedMs / intimDurMs, 0, 1);
            m.velX *= Math.exp(-12 * dt);
            m.velZ *= Math.exp(-12 * dt);
            const toPlayerYaw = Math.atan2(platTarget.x - m.x, platTarget.z - m.z);
            const gazeDiff = shortestAngleDiffNav(m.yaw, toPlayerYaw);
            if (Math.abs(gazeDiff) > 0.78) {
              const pivotYaw = toPlayerYaw - Math.sign(gazeDiff) * 0.78;
              m.yaw = normalizeAngle(m.yaw + shortestAngleDiffNav(m.yaw, pivotYaw) * Math.min(1, 3.2 * dt));
            }

            if (now < (m.watchGraceUntil || 0) && platTarget.horizSpeed <= 0.45) {
              m.watchAnchorX = platTarget.x;
              m.watchAnchorZ = platTarget.z;
            }
            if (now < platTarget.pushImmunityUntil) {
              m.watchAnchorX = platTarget.x;
              m.watchAnchorZ = platTarget.z;
            }

            const pushedRecently = now < platTarget.pushImmunityUntil;
            const dispWhileWatched = Math.hypot(
              platTarget.x - (m.watchAnchorX ?? platTarget.x),
              platTarget.z - (m.watchAnchorZ ?? platTarget.z)
            );
            const movedWhileWatched =
              Boolean(platTarget.moveAttemptWhileWatched) ||
              (!pushedRecently &&
                now >= (m.watchGraceUntil || 0) &&
                (platTarget.horizSpeed > 0.55 || platTarget.animState === 'jump' || dispWhileWatched > 0.25));

            if (movedWhileWatched) {
              m.visionState = 'platform_lunging';
              m.visionUntil = 0;
              m.watchingUserId = platTarget.userId;
              m.intimPhase = null;
              m.intimProgress = 0;
              m.alert = 'chasing';
              m.emerged = true;
              platTarget.moveAttemptWhileWatched = false;
              const dxL = platTarget.x - m.x;
              const dzL = platTarget.z - m.z;
              const lLen = Math.hypot(dxL, dzL);
              if (lLen > 0.01) {
                m.yaw = Math.atan2(dxL, dzL);
                m.velX = (dxL / lLen) * extremeChaseSpeed;
                m.velZ = (dzL / lLen) * extremeChaseSpeed;
              }
            } else if (now >= (m.visionUntil || 0)) {
              m.visionState = 'platform_lowering';
              m.intimPhase = 'lowering';
              m.intimStartedAt = now;
              m.intimDurationMs = Math.round(SERPENT_INTIMIDATION_LOWER_SEC * 1000);
              m.intimProgress = 0;
              m.watchingUserId = null;
              m.alert = 'calm';
              platTarget.moveAttemptWhileWatched = false;
              this.dirtyPlayersSinceBroadcast = true;
            }
          } else if (m.visionState === 'platform_lowering') {
            m.alert = 'calm';
            m.watchingUserId = null;
            m.lookAway = false;
            m.emerged = true;
            m.intimPhase = 'lowering';
            m.intimProgress = clamp(intimElapsedMs / intimDurMs, 0, 1);
            m.velX *= Math.exp(-10 * dt);
            m.velZ *= Math.exp(-10 * dt);
            if (intimElapsedMs >= intimDurMs) {
              const ret = this.pickServerRetreatTarget(def, m, platTarget.x, platTarget.z, monsterRadius, monsterBotY, monsterTopY, elapsedSec);
              m.platformSpotX = ret.x;
              m.platformSpotZ = ret.z;
              m.intimYaw = ret.yaw;
              m.visionState = 'platform_retreating';
              m.intimPhase = 'retreating';
              m.intimStartedAt = now;
              m.intimDurationMs = 2600;
              m.intimProgress = 0;
              m.targetUserId = null;
            }
          } else if (m.visionState === 'platform_lunging') {
            m.alert = 'chasing';
            m.watchingUserId = platTarget.userId;
            m.lookAway = false;
            m.emerged = true;
            m.intimPhase = null;
            m.intimProgress = 0;
          }
        }
      } else if (m.visionState === 'platform_retreating') {
        m.alert = 'calm';
        m.watchingUserId = null;
        m.lookAway = false;
        m.emerged = false;
        m.targetUserId = null;
        const intimElapsedMs = now - (m.intimStartedAt || now);
        const intimDurMs = Math.max(1, m.intimDurationMs || 2600);
        m.intimPhase = 'retreating';
        m.intimProgress = clamp(intimElapsedMs / intimDurMs, 0, 1);
        const distToSpot = Math.hypot((m.platformSpotX ?? m.x) - m.x, (m.platformSpotZ ?? m.z) - m.z);
        if (intimElapsedMs >= intimDurMs || (intimElapsedMs >= 1200 && distToSpot < 2.2)) {
          m.visionState = 'none';
          m.intimPhase = null;
          m.intimProgress = 0;
          selectClosestPatrolWaypoint(m, monsterRadius, checkMonsterBlocked);
        }
      } else if (holdingStrike) {
        m.alert = 'chasing';
        m.emerged = true;
        m.visionState = 'none';
        m.watchingUserId = null;
        m.lookAway = false;
        m.intimPhase = null;
        m.intimProgress = 0;
        m.velX *= 0.85;
        m.velZ *= 0.85;
      } else {
        m.attackHoldUntil = 0;
        m.visionState = 'none';
        m.visionUntil = 0;
        m.watchingUserId = null;
        m.lookAway = false;
        m.watchGraceUntil = 0;
        m.intimPhase = null;
        m.intimProgress = 0;
        if (m.alert !== 'calm') {
          m.alert = 'calm';
          m.emerged = false;
          m.targetUserId = null;
          const spd = Math.hypot(m.velX, m.velZ);
          if (spd > m.patrolSpeed && spd > 0.01) {
            const damp = m.patrolSpeed / spd;
            m.velX *= damp;
            m.velZ *= damp;
          }
          m.avoidTimer = 0;
          selectClosestPatrolWaypoint(m, monsterRadius, checkMonsterBlocked);
        }
      }

      const isChasing = m.alert === 'chasing';
      const isMooredIntim =
        m.visionState === 'platform_rising' ||
        m.visionState === 'staring' ||
        m.visionState === 'platform_lowering';
      const platformOverhead =
        !isMooredIntim &&
        this.isCircleBlocked(
          def,
          m.x,
          m.z,
          monsterRadius * 1.15,
          m.y - 0.12 * m.scale,
          waterY + 1.25 * m.scale,
          elapsedSec,
          false,
          true
        );

      let tx = m.x;
      let tz = m.z;
      let moveSpeed = m.patrolSpeed;
      let hasMoveTarget = false;
      let useFastSteer = false;

      if (m.visionState === 'platform_approaching') {
        tx = m.platformSpotX ?? m.x;
        tz = m.platformSpotZ ?? m.z;
        const dSpot = Math.hypot(tx - m.x, tz - m.z);
        const fastCap = Math.max(24.0, m.chaseSpeed * 3.6);
        moveSpeed = dSpot > 3.6 * m.scale ? fastCap : 4.5 + (fastCap - 4.5) * (dSpot / (3.6 * m.scale));
        hasMoveTarget = dSpot > 0.25;
        useFastSteer = true;
      } else if (isMooredIntim || holdingStrike) {
        tx = m.x;
        tz = m.z;
        moveSpeed = 0;
        hasMoveTarget = false;
      } else if (m.visionState === 'platform_retreating') {
        tx = m.platformSpotX ?? m.x;
        tz = m.platformSpotZ ?? m.z;
        moveSpeed = m.patrolSpeed * 1.45;
        hasMoveTarget = Math.hypot(tx - m.x, tz - m.z) > 0.4;
      } else if ((m.visionState === 'lunging' || m.visionState === 'platform_lunging' || m.alert === 'chasing') && target) {
        tx = target.x;
        tz = target.z;
        moveSpeed = extremeChaseSpeed;
        hasMoveTarget = Math.hypot(tx - m.x, tz - m.z) > 0.05;
        useFastSteer = true;
      } else {
        const patrolTarget = resolvePatrolTarget(m, monsterRadius, checkMonsterBlocked);
        tx = patrolTarget.tx;
        tz = patrolTarget.tz;
        hasMoveTarget = patrolTarget.hasTarget;
      }

      const minSafeY = def.bounds.floorY + 0.55 * m.scale;
      let lowestNearbyPlatBottom = waterY - 0.35;
      this.getLevelGrid(def).queryRadius(m.x, m.z, useFastSteer ? 16 : 10, this.scratchPlats, null);
      for (let pi = 0; pi < this.scratchPlats.length; pi++) {
        const plat = def.platforms[this.scratchPlats[pi]];
        if (!plat || plat.kind === 'disappearing') continue;
        let py = plat.position[1];
        if (plat.kind === 'moving' && plat.moveOffset) {
          const factor = 0.5 - 0.5 * Math.cos(elapsedSec * (plat.moveSpeed ?? 1.0) + (plat.movePhase ?? 0));
          py += plat.moveOffset[1] * factor;
        }
        const platBot = py - plat.size[1] * 0.5;
        if (platBot <= waterY + 0.6 && platBot > def.bounds.floorY + 0.8 && platBot < lowestNearbyPlatBottom) {
          lowestNearbyPlatBottom = platBot;
        }
      }
      const deepDiveY = clamp(
        Math.min(waterY - 2.05 * m.scale, lowestNearbyPlatBottom - 0.62 * m.scale),
        minSafeY,
        waterY - 1.15 * m.scale
      );
      const diveHalfH = 0.5 * m.scale;
      const diveBotY = deepDiveY - diveHalfH;
      const diveTopY = deepDiveY + diveHalfH;
      const strikeOverPlatform = m.visionState === 'platform_lunging';
      const allowDive =
        !holdingStrike &&
        !isMooredIntim &&
        !strikeOverPlatform &&
        (m.visionState !== 'platform_approaching' || Math.hypot(tx - m.x, tz - m.z) > 3.2 * m.scale);

      let corridorPlatformBlocked = false;
      if (allowDive && hasMoveTarget) {
        const dTgt = Math.hypot(tx - m.x, tz - m.z);
        if (dTgt > 0.5) {
          const invD = 1 / dTgt;
          const dirX = (tx - m.x) * invD;
          const dirZ = (tz - m.z) * invD;
          const maxProbe = Math.min(dTgt, useFastSteer ? 18.0 : 9.0);
          const stepLen = Math.max(1.0, monsterRadius * 1.45);
          for (let d = 0.45; d <= maxProbe; d += stepLen) {
            const sx = m.x + dirX * d;
            const sz = m.z + dirZ * d;
            if (this.isCircleBlocked(def, sx, sz, monsterRadius * 0.9, diveBotY, diveTopY, elapsedSec, true, false)) {
              break;
            }
            if (
              this.isCircleBlocked(
                def,
                sx,
                sz,
                monsterRadius * 1.02,
                waterY - 1.05 * m.scale,
                waterY + 1.05 * m.scale,
                elapsedSec,
                false,
                true
              ) &&
              !this.isCircleBlocked(def, sx, sz, monsterRadius * 0.86, diveBotY, diveTopY, elapsedSec, false, true)
            ) {
              corridorPlatformBlocked = true;
              break;
            }
          }
        }
        if (!corridorPlatformBlocked) {
          const fwdX = m.x + Math.sin(m.yaw) * monsterRadius * 1.45;
          const fwdZ = m.z + Math.cos(m.yaw) * monsterRadius * 1.45;
          if (
            this.isCircleBlocked(
              def,
              fwdX,
              fwdZ,
              monsterRadius * 1.02,
              waterY - 1.05 * m.scale,
              waterY + 1.05 * m.scale,
              elapsedSec,
              false,
              true
            ) &&
            !this.isCircleBlocked(def, fwdX, fwdZ, monsterRadius * 0.86, diveBotY, diveTopY, elapsedSec, false, true)
          ) {
            corridorPlatformBlocked = true;
          }
        }
      }

      if (allowDive && (platformOverhead || corridorPlatformBlocked || (isChasing && m.stuckTimer > 0.16))) {
        m.diveTimer = Math.max(m.diveTimer || 0, useFastSteer ? 0.55 : 0.75);
      } else if ((m.diveTimer || 0) > 0) {
        if (
          ((isChasing && target && targetDist < 3.6) || !allowDive) &&
          !platformOverhead &&
          !corridorPlatformBlocked
        ) {
          m.diveTimer = 0;
        } else {
          m.diveTimer = Math.max(0, (m.diveTimer || 0) - dt);
        }
      }

      const wantDive =
        !holdingStrike &&
        !isMooredIntim &&
        !strikeOverPlatform &&
        (platformOverhead || (allowDive && (corridorPlatformBlocked || (m.diveTimer || 0) > 0)));
      if (wantDive || m.y <= waterY - 0.85 * m.scale) {
        this._cbBotY = diveBotY;
        this._cbTopY = diveTopY;
      } else {
        this._cbBotY = monsterBotY;
        this._cbTopY = monsterTopY;
      }

      const subSteps = useFastSteer ? 6 : 1;
      const subDt = dt / subSteps;
      const catchDist = 2.15 * m.scale;
      let startedEating = false;
      const desiredY =
        isMooredIntim
          ? waterY - 0.14
          : wantDive || platformOverhead
            ? deepDiveY
            : m.alert === 'chasing' && (m.emerged || holdingStrike)
              ? waterY + 0.08
              : waterY - 0.14;
      const emergedFrac = m.emerged && m.y >= waterY - 0.55 * m.scale ? 1 : 0;

      if (!isMooredIntim) {
        for (let step = 0; step < subSteps; step++) {
          if (desiredY < m.y) {
            const diveRate = (useFastSteer ? 34.0 : 12.0) * m.scale;
            m.y = Math.max(desiredY, m.y - diveRate * subDt);
          } else if (desiredY > m.y) {
            const subOverhead =
              !strikeOverPlatform &&
              this.isCircleBlocked(
                def,
                m.x,
                m.z,
                monsterRadius * 1.15,
                m.y - 0.12 * m.scale,
                waterY + 1.25 * m.scale,
                elapsedSec,
                false,
                true
              );
            if (!subOverhead) {
              const riseRate = (useFastSteer ? 24.0 : 7.5) * m.scale;
              m.y = Math.min(desiredY, m.y + riseRate * subDt);
            }
          }
          m.y = clamp(m.y, def.bounds.floorY + 0.72 * m.scale, waterY + 0.08);

          const physBotY = m.y - 0.72 * m.scale;
          const physTopY = m.y + (0.68 + emergedFrac * 3.45) * m.scale;
          const prevX = m.x;
          const prevZ = m.z;
          const prevYaw = m.yaw;
          const steer = hasMoveTarget
            ? computeSerpentSteerAngle(
                m,
                tx,
                tz,
                monsterRadius,
                useFastSteer,
                strikeOverPlatform
                  ? (cx: number, cz: number, cr: number) =>
                      this.isCircleBlocked(def, cx, cz, cr, physBotY, physTopY, elapsedSec, true, false)
                  : checkMonsterBlocked
              )
            : { valid: false, steerAngle: m.yaw };

          if (steer.valid && moveSpeed > 0.05) {
            const blend = Math.min(1, (useFastSteer ? 65.0 : 8.5) * subDt);
            m.velX += (Math.sin(steer.steerAngle) * moveSpeed - m.velX) * blend;
            m.velZ += (Math.cos(steer.steerAngle) * moveSpeed - m.velZ) * blend;

            m.x += m.velX * subDt;
            m.z += m.velZ * subDt;

            if (!strikeOverPlatform) {
              this.resolveMonsterCircleSolids(def, m, monsterRadius, physBotY, physTopY, elapsedSec);

              if (this.isCircleBlocked(def, m.x, m.z, monsterRadius - 0.02, physBotY, physTopY, elapsedSec)) {
                let escaped = false;
                const escRadii = [0.25, 0.5, 0.85, 1.25];
                for (const stepR of escRadii) {
                  if (escaped) break;
                  for (let k = 0; k < 8; k++) {
                    const ang = (k * Math.PI) / 4;
                    const txEsc = m.x + Math.sin(ang) * stepR;
                    const tzEsc = m.z + Math.cos(ang) * stepR;
                    if (!this.isCircleBlocked(def, txEsc, tzEsc, monsterRadius, physBotY, physTopY, elapsedSec)) {
                      m.x = txEsc;
                      m.z = tzEsc;
                      escaped = true;
                      break;
                    }
                  }
                }
                if (!escaped) {
                  m.x = prevX;
                  m.z = prevZ;
                }
              }
            }
          }

          updateSerpentProgressAndYaw(
            m,
            prevX,
            prevZ,
            prevYaw,
            moveSpeed,
            subDt,
            useFastSteer,
            steer.valid,
            strikeOverPlatform
              ? (cx: number, cz: number, cr: number) =>
                  this.isCircleBlocked(def, cx, cz, cr, physBotY, physTopY, elapsedSec, true, false)
              : checkMonsterBlocked,
            24.0
          );
          if (wantDive && m.y > deepDiveY + 0.35 * m.scale) {
            m.stuckTimer = 0;
          }

          if (isChasing && m.visionState === 'lunging') {
            for (const p of aliveInWater) {
              if (!p.alive || p.escaped || !p.inWater || (p.beingEatenUntil && now < p.beingEatenUntil)) continue;
              const dToVictim = Math.hypot(p.x - m.x, p.z - m.z);
              if (dToVictim < catchDist) {
                const midX = (m.x + p.x) * 0.5;
                const midZ = (m.z + p.z) * 0.5;
                const wallBlocking = this.isCircleBlocked(
                  def,
                  midX,
                  midZ,
                  0.28,
                  Math.min(m.y, p.y) - 0.25,
                  Math.max(m.y, p.y) + 0.5,
                  elapsedSec,
                  true,
                  false
                );
                if (wallBlocking) continue;
                p.beingEatenUntil = now + EAT_DURATION_MS;
                p.beingEatenByMonsterId = m.id;
                p.moveAttemptWhileWatched = false;
                m.eatingUserId = p.userId;
                m.eatingStartAt = now;
                m.eatingUntil = now + EAT_DURATION_MS;
                m.eatingProgress = 0;
                m.attackHoldUntil = 0;
                m.emerged = true;
                m.targetUserId = p.userId;
                m.watchingUserId = null;
                m.lookAway = false;
                m.visionState = 'none';
                m.intimPhase = null;
                m.intimProgress = 0;
                m.velX = 0;
                m.velZ = 0;
                this.dirtyPlayersSinceBroadcast = true;
                const ev: GamePlayerEvent = {
                  userId: p.userId,
                  username: p.username,
                  kind: 'eating_started',
                  x: p.x,
                  y: p.y,
                  z: p.z,
                  monsterId: m.id,
                };
                this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
                startedEating = true;
                break;
              }
            }
            if (startedEating) break;
          } else if (isChasing && m.visionState === 'platform_lunging' && target) {
            if (target.alive && !target.escaped && (!target.beingEatenUntil || now >= target.beingEatenUntil)) {
              const effCatchDist = Math.max(catchDist, 2.8 * m.scale);
              const dToVictim = Math.hypot(target.x - m.x, target.z - m.z);
              if (dToVictim < effCatchDist) {
                const midX = (m.x + target.x) * 0.5;
                const midZ = (m.z + target.z) * 0.5;
                const wallBlocking = this.isCircleBlocked(
                  def,
                  midX,
                  midZ,
                  0.28,
                  Math.min(m.y, target.y) - 0.25,
                  Math.max(m.y, target.y) + 0.5,
                  elapsedSec,
                  true,
                  false
                );
                if (!wallBlocking) {
                  target.beingEatenUntil = now + EAT_DURATION_MS;
                  target.beingEatenByMonsterId = m.id;
                  target.moveAttemptWhileWatched = false;
                  m.eatingUserId = target.userId;
                  m.eatingStartAt = now;
                  m.eatingUntil = now + EAT_DURATION_MS;
                  m.eatingProgress = 0;
                  m.attackHoldUntil = 0;
                  m.emerged = true;
                  m.targetUserId = target.userId;
                  m.watchingUserId = null;
                  m.lookAway = false;
                  m.visionState = 'none';
                  m.intimPhase = null;
                  m.intimProgress = 0;
                  m.velX = 0;
                  m.velZ = 0;
                  this.dirtyPlayersSinceBroadcast = true;
                  const ev: GamePlayerEvent = {
                    userId: target.userId,
                    username: target.username,
                    kind: 'eating_started',
                    x: target.x,
                    y: target.y,
                    z: target.z,
                    monsterId: m.id,
                  };
                  this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
                  startedEating = true;
                  break;
                }
              }
            }
          }
        }
      } else {
        m.y += (waterY - 0.14 - m.y) * Math.min(1, 14 * dt);
      }

      if (m.eatingUntil && now < m.eatingUntil) {
        m.emerged = true;
      } else if (m.attackHoldUntil && now < m.attackHoldUntil) {
        m.emerged = true;
      } else if (isMooredIntim || m.visionState === 'platform_lunging') {
        m.emerged = !wantDive && m.y >= waterY - 0.55 * m.scale;
      } else if ((m.alert === 'chasing' || m.visionState === 'lunging') && target) {
        m.emerged = !wantDive && m.y >= waterY - 0.55 * m.scale && Math.hypot(target.x - m.x, target.z - m.z) < 10.5;
      }
    }

    // Monster-vs-monster separation so multi-serpent rooms never overlap heads
    for (let i = 0; i < monsters.length; i++) {
      for (let j = i + 1; j < monsters.length; j++) {
        const a = monsters[i];
        const b = monsters[j];
        const anchoredA =
          a.visionState === 'platform_rising' ||
          a.visionState === 'staring' ||
          a.visionState === 'platform_lowering';
        const anchoredB =
          b.visionState === 'platform_rising' ||
          b.visionState === 'staring' ||
          b.visionState === 'platform_lowering';
        if (anchoredA && anchoredB) continue;
        const dx = b.x - a.x;
        const dz = b.z - a.z;
        const dist = Math.hypot(dx, dz);
        const minSep = (a.scale + b.scale) * 0.92;
        if (dist < minSep && dist > 1e-4) {
          const push = minSep - dist;
          const shareA = anchoredA ? 0 : anchoredB ? 1 : 0.5;
          const shareB = 1 - shareA;
          const nx = dx / dist;
          const nz = dz / dist;
          a.x -= nx * push * shareA;
          a.z -= nz * push * shareA;
          b.x += nx * push * shareB;
          b.z += nz * push * shareB;
          if (shareA > 0) {
            this.resolveMonsterCircleSolids(def, a, 0.9 * a.scale, a.y - 0.72 * a.scale, a.y + 0.72 * a.scale, elapsedSec);
          }
          if (shareB > 0) {
            this.resolveMonsterCircleSolids(def, b, 0.9 * b.scale, b.y - 0.72 * b.scale, b.y + 0.72 * b.scale, elapsedSec);
          }
        }
      }
    }
  }

  private evaluateMatchResolution(room: InternalRoom, batch: RoomDeliveryBatch): void {
    this.evaluateTeamPortalProgression(room, Date.now(), batch);
  }

  private finalizeMatch(
    room: InternalRoom,
    reason: 'all_resolved' | 'timeout' | 'all_caught',
    batch: RoomDeliveryBatch
  ): void {
    if (room.matchResult) return;
    if (this.tickTimer) {
      clearInterval(this.tickTimer);
      this.tickTimer = null;
    }

    const survivors = Array.from(room.players.values())
      .filter((p) => p.escaped)
      .map((p) => ({ userId: p.userId, username: p.username }));

    const elapsed = room.matchStartTime ? Math.round(((Date.now() - room.matchStartTime) / 1000) * 10) / 10 : 0;
    room.status = 'ended';
    room.matchResult = {
      outcome: survivors.length > 0 ? 'escaped' : 'failed',
      elapsed,
      survivors,
      reason,
    };

    for (const p of room.players.values()) {
      p.beingEatenUntil = 0;
      p.beingEatenByMonsterId = null;
      batch.presenceChangedUserIds!.push(p.userId);
    }
    for (const sim of room.levelSims.values()) {
      for (const m of sim.monsters) {
        m.eatingUserId = null;
        m.eatingStartAt = 0;
        m.eatingUntil = 0;
        m.eatingProgress = 0;
      }
    }
    const now = Date.now();
    const elapsedSec = room.matchStartTime ? r2(Math.max(0, (now - room.matchStartTime) / 1000)) : 0;
    const netPlayers = this.serializeNetPlayers(room, true);
    const sim = room.levelSims.get(room.levelId);
    const monsters = sim ? this.serializeNetMonstersList(sim.monsters) : [];
    const portalTotalCount = room.players.size;
    let portalReadyCount = 0;
    for (const pl of room.players.values()) {
      if (pl.connected && pl.alive && pl.escaped && !pl.transitioning && pl.levelId === room.levelId) {
        portalReadyCount++;
      }
    }
    for (const p of room.players.values()) {
      if (!p.connected) continue;
      batch.deliveries.push({
        targetUserId: p.userId,
        event: 'game:world',
        payload: {
          roomId: room.id,
          levelId: room.levelId,
          serverTime: now,
          timeRemaining: r2(p.levelTimeRemaining),
          elapsedSec,
          portalReadyCount,
          portalTotalCount,
          players: netPlayers,
          monsters,
          matchResult: room.matchResult,
        },
      });
    }
    const summary = this.serializeRoom(room);
    batch.roomSummary = summary;
    this.pushRoomBroadcast(batch, room, 'room:state', summary);
  }

  private advanceRoomSimulation(
    room: InternalRoom,
    now: number,
    batch: RoomDeliveryBatch,
    forceBroadcast = false
  ): void {
    if (room.status !== 'playing') return;

    const elapsedMs = Math.max(0, now - room.lastTickMs);
    let anyStateChanged = false;
    if (elapsedMs >= 10) {
      const dt = Math.min(0.1, elapsedMs / 1000);
      room.lastTickMs = now;

      for (const p of room.players.values()) {
        if (!p.connected) continue;

        if (p.transitioning && now >= p.transitioningUntil) {
          p.transitioning = false;
          p.transitioningUntil = 0;
          anyStateChanged = true;
          this.dirtyPlayersSinceBroadcast = true;
        }

        if (!p.alive && !p.escaped && p.respawnAt > 0 && now >= p.respawnAt) {
          this.respawnPlayerAtLevelStart(room, p, now, batch);
          anyStateChanged = true;
          continue;
        }

        if (p.alive && !p.escaped && !p.transitioning) {
          p.levelTimeRemaining = Math.max(0, p.levelTimeRemaining - dt);
          if (p.levelTimeRemaining <= 0) {
            p.alive = false;
            p.respawnAt = now + RESPAWN_DELAY_MS;
            p.deaths = (p.deaths || 0) + 1;
            p.beingEatenUntil = 0;
            p.beingEatenByMonsterId = null;
            this.dirtyPlayersSinceBroadcast = true;
            const sim = room.levelSims.get(p.levelId);
            if (sim) this.clearPlayerFromLevelMonsters(sim, p.userId);
            const ev: GamePlayerEvent = {
              userId: p.userId,
              username: p.username,
              kind: 'caught',
              levelId: p.levelId,
              x: r3(p.x),
              y: r3(p.y),
              z: r3(p.z),
            };
            this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
            anyStateChanged = true;
          }
        }
      }

      this.pruneEmptyLevelSims(room);
      for (const sim of room.levelSims.values()) {
        this.updateLevelSimMonsters(room, sim, dt, now, batch);
      }

      this.evaluateTeamPortalProgression(room, now, batch);
      if (anyStateChanged && room.status === 'playing') {
        const summary = this.serializeRoom(room);
        batch.roomSummary = summary;
        this.pushRoomBroadcast(batch, room, 'room:state', summary);
      }
    }

    if (room.status !== 'playing' && !room.matchResult) return;

    if (forceBroadcast || this.dirtyPlayersSinceBroadcast || now - this.lastWorldBroadcastMs >= 36) {
      this.lastWorldBroadcastMs = now;
      this.dirtyPlayersSinceBroadcast = false;
      const elapsedSec = room.matchStartTime ? r2(Math.max(0, (now - room.matchStartTime) / 1000)) : 0;
      const netPlayers = this.serializeNetPlayers(room, true);
      const sim = room.levelSims.get(room.levelId);
      const monsters = sim ? this.serializeNetMonstersList(sim.monsters) : [];
      const portalTotalCount = room.players.size;
      let portalReadyCount = 0;
      for (const pl of room.players.values()) {
        if (pl.connected && pl.alive && pl.escaped && !pl.transitioning && pl.levelId === room.levelId) {
          portalReadyCount++;
        }
      }
      for (const p of room.players.values()) {
        if (!p.connected) continue;
        batch.deliveries.push({
          targetUserId: p.userId,
          event: 'game:world',
          payload: {
            roomId: room.id,
            levelId: room.levelId,
            serverTime: now,
            timeRemaining: r2(p.levelTimeRemaining),
            elapsedSec,
            portalReadyCount,
            portalTotalCount,
            players: netPlayers,
            monsters,
            matchResult: room.matchResult,
          },
        });
      }
    }
  }

  private async runTick(): Promise<void> {
    const room = this.room;
    if (!room || room.status !== 'playing') return;
    if (this.tickDeliverInFlight) return;

    const now = Date.now();
    // Flush immediately if player state changed since last broadcast or 40ms elapsed
    if (!this.dirtyPlayersSinceBroadcast && now - this.lastWorldBroadcastMs < 40) return;

    const batch = this.createBatch();
    this.advanceRoomSimulation(room, now, batch, true);

    this.tickDeliverInFlight = true;
    try {
      await this.deliverBatchAsync(batch);
    } finally {
      this.tickDeliverInFlight = false;
    }
  }

  public async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (request.method !== 'POST' || url.pathname !== '/internal/action') {
      return new Response('Not found', { status: 404 });
    }

    let body: Record<string, unknown>;
    try {
      body = (await request.json()) as Record<string, unknown>;
    } catch {
      return Response.json({ error: 'Invalid JSON' }, { status: 400 });
    }

    const action = String(body.action || '');
    const batch = this.createBatch();

    try {
      switch (action) {
        case 'create': {
          const roomId = String(body.roomId || '');
          const code = String(body.code || '');
          const user = body.user as PublicUser;
          const versionErr = checkVersionCompatibility(body.gameVersion, GAME_VERSION);
          if (versionErr) {
            return Response.json({ error: versionErr }, { status: 400 });
          }
          const rawLevel = typeof body.levelId === 'number' ? body.levelId : 1;
          const safeLevelId = isValidLevelId(rawLevel) ? rawLevel : 1;
          const def = getLevelDef(safeLevelId);

          this.stopLoopAndTimers();
          const room: InternalRoom = {
            id: roomId,
            code,
            hostId: user.id,
            levelId: safeLevelId,
            gameVersion: String(body.gameVersion).trim(),
            status: 'lobby',
            players: new Map(),
            invites: new Map(),
            monsters: [],
            levelSims: new Map(),
            levelDef: def,
            matchStartTime: null,
            timeRemaining: def.timeLimit,
            matchResult: null,
            lastTickMs: Date.now(),
          };
          room.players.set(user.id, this.makeRoomPlayer(user, true, def));
          this.room = room;

          const serialized = this.serializeRoom(room);
          batch.joinedUsers!.push({ userId: user.id, roomId: room.id, code: room.code });
          batch.roomSummary = serialized;
          this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          return Response.json({ room: serialized, batch });
        }

        case 'join': {
          const room = this.room;
          if (!room) {
            return Response.json(
              { error: 'Private room not found. Check the room code and try again.' },
              { status: 404 }
            );
          }
          const versionErr = checkVersionCompatibility(body.gameVersion, room.gameVersion);
          if (versionErr) {
            return Response.json({ error: versionErr }, { status: 400 });
          }
          const user = body.user as PublicUser;
          const existing = room.players.get(user.id);
          if (existing) {
            existing.connected = true;
            this.disconnectDeadlines.delete(user.id);
            const serialized = this.serializeRoom(room);
            batch.roomSummary = serialized;
            this.pushRoomBroadcast(batch, room, 'room:state', serialized);
            return Response.json({ room: serialized, batch });
          }

          if (room.status !== 'lobby') {
            return Response.json({ error: 'This match is already in progress.' }, { status: 400 });
          }
          if (room.players.size >= MAX_ROOM_PLAYERS) {
            return Response.json(
              { error: `Room is full (${MAX_ROOM_PLAYERS}/${MAX_ROOM_PLAYERS} players).` },
              { status: 400 }
            );
          }

          const pendingInvite = room.invites.get(user.id);
          if (pendingInvite) {
            room.invites.delete(user.id);
            batch.purgedInvites!.push({
              userId: user.id,
              inviteId: pendingInvite.id,
              reason: 'accepted',
            });
          }

          room.players.set(user.id, this.makeRoomPlayer(user, false, room.levelDef));
          const serialized = this.serializeRoom(room);
          batch.joinedUsers!.push({ userId: user.id, roomId: room.id, code: room.code });
          batch.roomSummary = serialized;
          this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          return Response.json({ room: serialized, batch });
        }

        case 'reconnect': {
          const room = this.room;
          if (!room) return Response.json({ room: null, batch });
          const userId = String(body.userId || '');
          const rp = room.players.get(userId);
          if (rp) {
            rp.connected = true;
            this.disconnectDeadlines.delete(userId);
            const serialized = this.serializeRoom(room);
            batch.roomSummary = serialized;
            this.pushRoomBroadcast(batch, room, 'room:state', serialized);
            let gameStarted: Record<string, unknown> | null = null;
            if (room.status === 'playing') {
              const sim = this.ensureLevelSim(room, room.levelId);
              gameStarted = {
                roomId: room.id,
                levelId: room.levelId,
                timeLimit: rp.levelTimeRemaining,
                elapsedSec: room.matchStartTime
                  ? Math.max(0, (Date.now() - room.matchStartTime) / 1000)
                  : 0,
                players: this.serializeNetPlayers(room),
                monsters: this.serializeNetMonstersList(sim.monsters),
              };
            }
            return Response.json({ room: serialized, gameStarted, batch });
          }
          return Response.json({ room: null, gameStarted: null, batch });
        }

        case 'disconnect': {
          const room = this.room;
          if (!room) return Response.json({ batch });
          const userId = String(body.userId || '');
          const player = room.players.get(userId);
          if (!player) return Response.json({ batch });

          player.connected = false;
          player.voice = {
            voiceActive: false,
            micEnabled: false,
            selfMuted: false,
            speaking: false,
          };

          if (room.status !== 'playing') {
            this.removePlayerInternal(userId, 'left', batch);
          } else {
            if (room.hostId === userId) {
              this.transferHost(room, userId, batch);
            }
            const connectedCount = Array.from(room.players.values()).filter((p) => p.connected).length;
            if (connectedCount === 0) {
              this.destroyRoomInternal('All players disconnected.', batch);
            } else {
              const serialized = this.serializeRoom(room);
              batch.roomSummary = serialized;
              this.pushRoomBroadcast(batch, room, 'room:state', serialized);

              this.disconnectDeadlines.set(userId, Date.now() + this.getDisconnectGraceMs());
              this.scheduleNextAlarm(90);
            }
          }
          return Response.json({ batch });
        }

        case 'leave': {
          const userId = String(body.userId || '');
          this.removePlayerInternal(userId, 'left', batch);
          return Response.json({ batch });
        }

        case 'close': {
          const room = this.room;
          const userId = String(body.userId || '');
          if (!room) return Response.json({ error: 'Room not found.' }, { status: 404 });
          if (room.hostId !== userId) {
            return Response.json({ error: 'Only the room host can close the room.' }, { status: 403 });
          }
          this.destroyRoomInternal('Room closed by the host.', batch);
          return Response.json({ batch });
        }

        case 'kick': {
          const room = this.room;
          const hostUserId = String(body.userId || '');
          const targetUserId = String(body.targetUserId || '');
          if (!room) return Response.json({ error: 'Room not found.' }, { status: 404 });
          if (room.hostId !== hostUserId) {
            return Response.json({ error: 'Only the room host can remove players.' }, { status: 403 });
          }
          if (targetUserId === hostUserId) {
            return Response.json({ error: 'Host cannot remove themselves.' }, { status: 400 });
          }
          if (!room.players.has(targetUserId)) {
            return Response.json({ error: 'Player is not in this room.' }, { status: 404 });
          }
          this.removePlayerInternal(targetUserId, 'kicked', batch);
          return Response.json({ room: this.room ? this.serializeRoom(this.room) : null, batch });
        }

        case 'set_level': {
          const room = this.room;
          const hostUserId = String(body.userId || '');
          const levelId = typeof body.levelId === 'number' ? body.levelId : 1;
          if (!room) return Response.json({ error: 'Room not found.' }, { status: 404 });
          if (room.hostId !== hostUserId) {
            return Response.json({ error: 'Only the room host can change the sector.' }, { status: 403 });
          }
          if (room.status !== 'lobby') {
            return Response.json({ error: 'Cannot change sector while a match is running.' }, { status: 400 });
          }
          if (!isValidLevelId(levelId)) {
            return Response.json({ error: 'Invalid sector selected.' }, { status: 400 });
          }
          room.levelId = levelId;
          room.levelDef = getLevelDef(levelId);
          room.timeRemaining = room.levelDef.timeLimit;
          const serialized = this.serializeRoom(room);
          batch.roomSummary = serialized;
          this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          return Response.json({ room: serialized, batch });
        }

        case 'register_invite': {
          const room = this.room;
          if (!room) return Response.json({ error: 'Room not found.' }, { status: 404 });
          if (room.status !== 'lobby') {
            return Response.json({ error: 'Cannot send invitations while a match is in progress.' }, { status: 400 });
          }
          if (room.players.size >= MAX_ROOM_PLAYERS) {
            return Response.json(
              { error: `Room is already full (${MAX_ROOM_PLAYERS}/${MAX_ROOM_PLAYERS}).` },
              { status: 400 }
            );
          }
          const invite = body.invite as RoomInviteInfo;
          const toUserId = String(body.toUserId || '');
          if (room.players.has(toUserId)) {
            return Response.json({ error: 'This friend is already in your room.' }, { status: 400 });
          }
          if (room.invites.has(toUserId)) {
            return Response.json({ error: 'Invitation already sent to this friend.' }, { status: 400 });
          }
          room.invites.set(toUserId, invite);
          const serialized = this.serializeRoom(room);
          batch.roomSummary = serialized;
          this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          return Response.json({ room: serialized, batch });
        }

        case 'clear_invite': {
          const room = this.room;
          if (!room) return Response.json({ batch });
          const toUserId = String(body.toUserId || '');
          if (room.invites.delete(toUserId)) {
            const serialized = this.serializeRoom(room);
            batch.roomSummary = serialized;
            this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          }
          return Response.json({ batch });
        }

        case 'start': {
          const room = this.room;
          const hostUserId = String(body.userId || '');
          if (!room) return Response.json({ error: 'Room not found.' }, { status: 404 });
          if (room.hostId !== hostUserId) {
            return Response.json({ error: 'Only the room host can start the match.' }, { status: 403 });
          }
          if (room.status === 'playing') {
            return Response.json({ error: 'Match is already in progress.' }, { status: 400 });
          }

          const realConnectedPlayers = Array.from(room.players.values()).filter(
            (p) => !p.isBot && p.connected
          );
          if (realConnectedPlayers.length < MIN_ROOM_PLAYERS) {
            return Response.json(
              {
                error: `At least ${MIN_ROOM_PLAYERS} real players are required to start (${realConnectedPlayers.length}/${MAX_ROOM_PLAYERS}).`,
              },
              { status: 400 }
            );
          }
          if (realConnectedPlayers.length > MAX_ROOM_PLAYERS) {
            return Response.json(
              { error: `Room exceeds maximum of ${MAX_ROOM_PLAYERS} players.` },
              { status: 400 }
            );
          }

          const def = getLevelDef(room.levelId);
          room.levelDef = def;
          room.status = 'playing';
          room.matchStartTime = Date.now();
          room.timeRemaining = def.timeLimit;
          room.matchResult = null;
          room.levelSims.clear();

          for (const [uid, inv] of room.invites.entries()) {
            batch.purgedInvites!.push({ userId: uid, inviteId: inv.id, reason: 'match_started' });
          }
          room.invites.clear();

          const playerList = Array.from(room.players.values());
          const now = Date.now();
          playerList.forEach((p, idx) => {
            p.levelId = room.levelId;
            p.transitioning = false;
            p.transitioningUntil = 0;
            p.respawnAt = 0;
            p.deaths = 0;
            p.levelTimeRemaining = def.timeLimit;
            this.placePlayerOnLevelSpawn(def, p, idx, playerList.length);
            p.alive = true;
            p.escaped = false;
            p.checkpointIndex = 0;
            p.lastUpdateAt = now;
            p.lastPushAt = 0;
            p.pushImmunityUntil = 0;
            p.beingEatenUntil = 0;
            p.beingEatenByMonsterId = null;
            p.moveAttemptWhileWatched = false;
            p.wasInWaterSession = false;
            p.waterEpoch = 0;
            p.pendingPlatformWatchEpoch = 0;
            p.watchHandledEpoch = 0;
          });

          const startSim = this.ensureLevelSim(room, room.levelId);
          room.monsters = startSim.monsters;

          if (this.tickTimer) clearInterval(this.tickTimer);
          room.lastTickMs = Date.now();
          this.lastWorldBroadcastMs = room.lastTickMs;
          this.tickTimer = setInterval(() => {
            void this.runTick();
          }, 50);
          this.scheduleNextAlarm(90);

          for (const p of room.players.values()) {
            batch.presenceChangedUserIds!.push(p.userId);
          }

          const serialized = this.serializeRoom(room);
          batch.roomSummary = serialized;
          this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          this.pushRoomBroadcast(batch, room, 'game:started', {
            roomId: room.id,
            levelId: room.levelId,
            timeLimit: def.timeLimit,
            elapsedSec: 0,
            players: this.serializeNetPlayers(room),
            monsters: this.serializeNetMonstersList(startSim.monsters),
          });
          return Response.json({ room: serialized, batch });
        }

        case 'return_lobby': {
          const room = this.room;
          const hostUserId = String(body.userId || '');
          if (!room) return Response.json({ error: 'Room not found.' }, { status: 404 });
          if (room.hostId !== hostUserId) {
            return Response.json(
              { error: 'Only the room host can return the room to lobby.' },
              { status: 403 }
            );
          }

          if (this.tickTimer) {
            clearInterval(this.tickTimer);
            this.tickTimer = null;
          }

          room.status = 'lobby';
          room.matchStartTime = null;
          room.matchResult = null;
          room.timeRemaining = room.levelDef.timeLimit;
          room.levelSims.clear();
          room.monsters = [];
          for (const p of room.players.values()) {
            p.levelId = room.levelId;
            p.transitioning = false;
            p.transitioningUntil = 0;
            p.respawnAt = 0;
            p.deaths = 0;
            p.levelTimeRemaining = room.levelDef.timeLimit;
            p.alive = true;
            p.escaped = false;
            p.checkpointIndex = 0;
            p.lastPushAt = 0;
            p.pushImmunityUntil = 0;
            p.beingEatenUntil = 0;
            p.beingEatenByMonsterId = null;
            p.moveAttemptWhileWatched = false;
            p.wasInWaterSession = false;
            p.waterEpoch = 0;
            p.pendingPlatformWatchEpoch = 0;
            p.watchHandledEpoch = 0;
          }
          for (const p of room.players.values()) {
            batch.presenceChangedUserIds!.push(p.userId);
          }
          const serialized = this.serializeRoom(room);
          batch.roomSummary = serialized;
          this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          return Response.json({ room: serialized, batch });
        }

        case 'level_ready': {
          const room = this.room;
          if (!room || room.status !== 'playing') return Response.json({ batch });
          const userId = String(body.userId || '');
          const levelId = typeof body.levelId === 'number' ? body.levelId : 1;
          const p = room.players.get(userId);
          if (!p || p.levelId !== levelId) return Response.json({ batch });
          p.transitioning = false;
          p.transitioningUntil = 0;
          p.lastUpdateAt = Date.now();
          this.dirtyPlayersSinceBroadcast = true;
          const serialized = this.serializeRoom(room);
          batch.roomSummary = serialized;
          this.pushRoomBroadcast(batch, room, 'room:state', serialized);
          return Response.json({ batch });
        }

        case 'portal_enter': {
          const room = this.room;
          if (!room || room.status !== 'playing' || room.matchResult) return Response.json({ batch });
          const userId = String(body.userId || '');
          const p = room.players.get(userId);
          const now = Date.now();
          if (!p || !p.connected || !p.alive || p.escaped || p.transitioning || (p.beingEatenUntil && now < p.beingEatenUntil)) {
            return Response.json({ batch });
          }
          const reportedLevelId = typeof body.levelId === 'number' ? body.levelId : undefined;
          if (typeof reportedLevelId === 'number' && Number.isFinite(reportedLevelId) && reportedLevelId !== room.levelId) {
            return Response.json({ batch });
          }
          p.levelId = room.levelId;
          const def = getLevelDef(room.levelId);
          const ex = def.exit.position;
          const dist = Math.hypot(p.x - ex[0], p.z - ex[2]);
          if (dist <= 4.5 && Math.abs(p.y - ex[1]) <= 3.5) {
            this.handlePlayerReachedPortal(room, p, def, now, batch);
          }
          return Response.json({ batch });
        }

        case 'input_state': {
          const room = this.room;
          if (!room || room.status !== 'playing') return Response.json({ batch });
          const userId = String(body.userId || '');
          const p = room.players.get(userId);
          const now = Date.now();
          if (!p || !p.alive || (p.beingEatenUntil && now < p.beingEatenUntil)) {
            return Response.json({ batch });
          }

          const packet = body.packet as ClientPlayerInputPacket;
          if (
            packet &&
            typeof packet.levelId === 'number' &&
            Number.isFinite(packet.levelId) &&
            packet.levelId !== room.levelId
          ) {
            return Response.json({ batch });
          }
          if (p.escaped) {
            if (packet?.characterId !== undefined) {
              p.characterId = normalizeCharacterId(packet.characterId);
            }
            p.lastUpdateAt = now;
            return Response.json({ batch });
          }
          if (!packet || !Number.isFinite(packet.x) || !Number.isFinite(packet.y) || !Number.isFinite(packet.z)) {
            return Response.json({ batch });
          }

          if (p.transitioning && !packet.escaped) {
            p.transitioning = false;
            p.transitioningUntil = 0;
          }

          const levelDef = getLevelDef(room.levelId);
          const prevX = p.x;
          const prevY = p.y;
          const prevZ = p.z;
          const prevYaw = p.yaw;
          const prevAnim = p.animState;
          const prevInWater = p.inWater;
          const prevMoveAttempt = Boolean(p.moveAttemptWhileWatched);
          const dtInput = Math.max(0.001, (now - p.lastUpdateAt) / 1000);

          const bounds = levelDef.bounds;
          p.x = clamp(packet.x, bounds.minX + 0.35, bounds.maxX - 0.35);
          p.y = clamp(packet.y, bounds.floorY, bounds.ceilingY - 0.5);
          p.z = clamp(packet.z, bounds.minZ + 0.35, bounds.maxZ - 0.35);
          if (Number.isFinite(packet.yaw)) p.yaw = packet.yaw;
          const validAnim =
            packet.animState === 'idle' ||
            packet.animState === 'walk' ||
            packet.animState === 'run' ||
            packet.animState === 'jump' ||
            packet.animState === 'fall' ||
            packet.animState === 'swim';
          p.animState = validAnim ? packet.animState : 'idle';
          p.horizSpeed = Number.isFinite(packet.horizSpeed) ? clamp(packet.horizSpeed, 0, 18) : 0;

          if (
            Number.isFinite(packet.velX) &&
            Number.isFinite(packet.velY) &&
            Number.isFinite(packet.velZ)
          ) {
            p.velX = clamp(packet.velX as number, -18, 18);
            p.velY = clamp(packet.velY as number, -18, 18);
            p.velZ = clamp(packet.velZ as number, -18, 18);
          } else if (dtInput >= 0.012 && dtInput <= 0.4) {
            p.velX = clamp((p.x - prevX) / dtInput, -18, 18);
            p.velY = clamp((p.y - prevY) / dtInput, -18, 18);
            p.velZ = clamp((p.z - prevZ) / dtInput, -18, 18);
          } else {
            p.velX = 0;
            p.velY = 0;
            p.velZ = 0;
          }

          const elapsedSec = room.matchStartTime ? Math.max(0, (now - room.matchStartTime) / 1000) : 0;
          p.inWater = this.isPlayerInWaterServer(
            levelDef,
            p.x,
            p.y,
            p.z,
            Boolean(packet.inWater),
            elapsedSec
          );
          const prevPendingWatch = p.pendingPlatformWatchEpoch || 0;
          if (p.inWater) {
            if (!p.wasInWaterSession) {
              p.wasInWaterSession = true;
              p.waterEpoch = (p.waterEpoch || 0) + 1;
            }
            p.pendingPlatformWatchEpoch = 0;
            p.moveAttemptWhileWatched = false;
          } else if (
            p.wasInWaterSession &&
            !packet.inWater &&
            this.isPlayerOnPlatform(levelDef, p.x, p.y, p.z, elapsedSec, false) &&
            p.y >= levelDef.bounds.waterY - 0.15
          ) {
            p.wasInWaterSession = false;
            p.pendingPlatformWatchEpoch = p.waterEpoch || 1;
            p.moveAttemptWhileWatched = Boolean(packet.moveAttemptWhileWatched);
          } else {
            p.moveAttemptWhileWatched = Boolean(packet.moveAttemptWhileWatched);
          }
          if (Number.isFinite(packet.checkpointIndex)) {
            p.checkpointIndex = Math.max(p.checkpointIndex, Math.floor(packet.checkpointIndex));
          }
          if (packet.characterId !== undefined) {
            p.characterId = normalizeCharacterId(packet.characterId);
          }

          const moved =
            Math.abs(p.x - prevX) + Math.abs(p.y - prevY) + Math.abs(p.z - prevZ) > 0.004 ||
            Math.abs(shortestAngleDiff(prevYaw, p.yaw)) > 0.008;
          let urgentSync =
            prevInWater !== p.inWater ||
            prevAnim !== p.animState ||
            prevMoveAttempt !== p.moveAttemptWhileWatched ||
            prevPendingWatch !== (p.pendingPlatformWatchEpoch || 0);

          if (moved || urgentSync) {
            this.dirtyPlayersSinceBroadcast = true;
          }

          if (packet.escaped && !p.escaped && !p.transitioning) {
            const ex = levelDef.exit.position;
            const dist = Math.hypot(p.x - ex[0], p.z - ex[2]);
            if (dist <= 4.5 && Math.abs(p.y - ex[1]) <= 3.5) {
              this.handlePlayerReachedPortal(room, p, levelDef, now, batch);
              urgentSync = true;
            }
          }

          p.lastUpdateAt = now;
          // Piggyback authoritative simulation & game:world directly on input_state responses so
          // real-time player movement & water chases NEVER stall even if Cloudflare pauses setInterval.
          if (urgentSync || now - this.lastWorldBroadcastMs >= 32) {
            this.advanceRoomSimulation(room, now, batch, true);
          }
          this.scheduleNextAlarm(90);
          return Response.json({ batch });
        }

        case 'push': {
          const room = this.room;
          if (!room || room.status !== 'playing') return Response.json({ batch });
          const userId = String(body.userId || '');
          const attacker = room.players.get(userId);
          const now = Date.now();
          if (
            !attacker ||
            !attacker.connected ||
            !attacker.alive ||
            attacker.escaped ||
            attacker.transitioning ||
            (attacker.beingEatenUntil && now < attacker.beingEatenUntil)
          ) {
            return Response.json({ batch });
          }

          if (now - attacker.lastPushAt < PUSH_COOLDOWN_MS) {
            return Response.json({ batch });
          }

          const levelDef = getLevelDef(attacker.levelId);
          const packet = body.packet as ClientPushPacket | undefined;
          const yaw = packet && Number.isFinite(packet.yaw) ? packet.yaw : attacker.yaw;
          let ax = attacker.x;
          let ay = attacker.y;
          let az = attacker.z;
          const b = levelDef.bounds;
          if (
            packet &&
            Number.isFinite(packet.x) &&
            Number.isFinite(packet.y) &&
            Number.isFinite(packet.z)
          ) {
            const px = clamp(packet.x as number, b.minX + 0.35, b.maxX - 0.35);
            const py = clamp(packet.y as number, b.floorY, b.ceilingY - 0.5);
            const pz = clamp(packet.z as number, b.minZ + 0.35, b.maxZ - 0.35);
            if (Math.hypot(px - attacker.x, pz - attacker.z) <= 3.2 && Math.abs(py - attacker.y) <= 2.2) {
              ax = px;
              ay = py;
              az = pz;
              attacker.x = ax;
              attacker.y = ay;
              attacker.z = az;
            }
          }
          attacker.yaw = yaw;

          const elapsedSec = room.matchStartTime ? Math.max(0, (now - room.matchStartTime) / 1000) : 0;
          const candidates = Array.from(room.players.values()).filter(
            (p) =>
              p.userId !== userId &&
              p.levelId === attacker.levelId &&
              !p.transitioning &&
              p.connected &&
              p.alive &&
              !p.escaped &&
              now >= p.pushImmunityUntil &&
              (!p.beingEatenUntil || now >= p.beingEatenUntil)
          );

          const selected = selectPushTarget(
            levelDef,
            { userId, x: ax, y: ay, z: az, yaw },
            candidates,
            elapsedSec,
            typeof packet?.targetUserId === 'string' ? packet.targetUserId : undefined
          );
          if (!selected) {
            return Response.json({ batch });
          }

          const target = room.players.get(selected.userId);
          if (!target || target.levelId !== attacker.levelId) {
            return Response.json({ batch });
          }

          attacker.lastPushAt = now;
          target.pushImmunityUntil = now + PUSH_RECOVERY_MS;

          // Apply immediate wall-safe impulse displacement & velocity on authoritative server state
          const nudged = computePushNudge(
            levelDef,
            target.x,
            target.y,
            target.z,
            selected.dirX,
            selected.dirZ,
            0.38,
            elapsedSec
          );
          target.x = nudged.x;
          target.z = nudged.z;
          target.velX = selected.dirX * PUSH_FORCE_HORIZ;
          target.velY = PUSH_FORCE_LIFT_Y;
          target.velZ = selected.dirZ * PUSH_FORCE_HORIZ;
          target.horizSpeed = PUSH_FORCE_HORIZ;
          target.animState = target.inWater ? 'swim' : 'fall';
          target.inWater = this.isPlayerInWaterServer(
            levelDef,
            target.x,
            target.y,
            target.z,
            target.inWater,
            elapsedSec
          );
          target.lastUpdateAt = now;
          this.dirtyPlayersSinceBroadcast = true;

          const ev: GamePlayerEvent = {
            kind: 'pushed',
            userId: target.userId,
            username: target.username,
            levelId: attacker.levelId,
            x: target.x,
            y: target.y,
            z: target.z,
            attackerUserId: attacker.userId,
            attackerUsername: attacker.username,
            dirX: selected.dirX,
            dirZ: selected.dirZ,
            force: PUSH_FORCE_HORIZ,
            liftY: PUSH_FORCE_LIFT_Y,
          };
          this.pushRoomBroadcast(batch, room, 'game:player_event', ev);
          this.advanceRoomSimulation(room, now, batch, true);
          return Response.json({ batch });
        }

        case 'voice_state': {
          const room = this.room;
          if (!room) return Response.json({ batch });
          const userId = String(body.userId || '');
          const player = room.players.get(userId);
          if (!player) return Response.json({ batch });

          const rawState = (body.voice || {}) as Partial<RoomVoicePeerState>;
          const voiceActive = Boolean(rawState.voiceActive);
          const micEnabled = voiceActive ? Boolean(rawState.micEnabled) : false;
          const selfMuted = Boolean(rawState.selfMuted);
          const speaking = voiceActive && micEnabled && !selfMuted ? Boolean(rawState.speaking) : false;

          const prev = player.voice;
          if (
            prev.voiceActive === voiceActive &&
            prev.micEnabled === micEnabled &&
            prev.selfMuted === selfMuted &&
            prev.speaking === speaking
          ) {
            return Response.json({ batch });
          }

          player.voice = { voiceActive, micEnabled, selfMuted, speaking };
          const msg: DeliveryMessage = {
            targetUserIds: this.roomPlayerIds(room),
            event: 'voice:peer_state',
            payload: {
              roomId: room.id,
              userId,
              voice: { ...player.voice },
            },
          };
          batch.deliveries.push(msg);
          return Response.json({ batch });
        }

        case 'voice_signal': {
          const room = this.room;
          const fromUserId = String(body.fromUserId || '');
          const targetUserId = String(body.targetUserId || '');
          const kind = body.kind as VoiceSignalKind;
          const data = body.data;

          if (!targetUserId || targetUserId === fromUserId) return Response.json({ batch });
          if (kind !== 'offer' && kind !== 'answer' && kind !== 'ice') return Response.json({ batch });
          if (!data || typeof data !== 'object') return Response.json({ batch });

          if (!room || !room.players.has(fromUserId)) {
            return Response.json({ error: 'Join a private room before using voice chat.' }, { status: 400 });
          }
          if (!room.players.has(targetUserId)) {
            return Response.json({ batch });
          }

          const encodedSize = JSON.stringify(data).length;
          if (encodedSize > 32768) {
            return Response.json({ error: 'Voice signaling payload exceeded limit.' }, { status: 400 });
          }

          batch.deliveries.push({
            targetUserId,
            event: 'voice:signal',
            payload: {
              roomId: room.id,
              fromUserId,
              kind,
              data,
            },
          });
          return Response.json({ batch });
        }

        case 'get_state': {
          return Response.json({
            room: this.room ? this.serializeRoom(this.room) : null,
            status: this.room?.status || null,
          });
        }

        default:
          return Response.json({ error: `Unknown room action: ${action}` }, { status: 400 });
      }
    } catch (err) {
      return Response.json(
        { error: err instanceof Error ? err.message : 'Room action failed.' },
        { status: 400 }
      );
    }
  }
}
