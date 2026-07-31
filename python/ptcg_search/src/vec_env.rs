/// Rust-backed parallel game environment (RL_SPEC §6.3).
///
/// Wraps libcg battle API via C bridge functions to avoid struct-return
/// ABI mismatch.  N battles, rayon parallel, direct FFI.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::os::raw::c_int;
use std::ptr;

use rayon::prelude::*;
use std::io::Write;

use crate::bridge;

fn _trace(msg: &str) {
    let _ = std::fs::write("/tmp/vecenv_trace.txt", msg);
}
fn _append(msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/vecenv_hist.txt") {
        let _ = f.write_all(msg.as_bytes());
    }
}
use crate::ffi::{cstr_to_string, CgLib, FnBattleFinish, FnBattleStart,
                  FnGetBattleData, FnSelect, SerialData, StartData};

// ── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VecEnvConfig {
    pub n_envs: usize,
    pub our_player: u8,
    pub max_decisions: usize,
    pub seed: u64,
}

impl Default for VecEnvConfig {
    fn default() -> Self {
        VecEnvConfig { n_envs: 12, our_player: 0, max_decisions: 400, seed: 42 }
    }
}

// ── Single battle ───────────────────────────────────────────────────────

struct Battle {
    ptr: *mut std::ffi::c_void,
    deck_self: Vec<i32>,
    deck_opp: Vec<i32>,
    n_decisions: usize,
    finished: bool,
    reward: f32,
    error: Option<String>,
    last_obs_json: String,
    last_sbi: String,
    needs_action: bool,
    select_player: i32,
}
// Each battle has its own libcg handle; concurrent battles are
// independent (RL_SPEC §2.1 evidence).
unsafe impl Send for Battle {}
unsafe impl Sync for Battle {}

impl Battle {
    fn new(deck_self: &[i32], deck_opp: &[i32]) -> Self {
        Battle {
            ptr: ptr::null_mut(), deck_self: deck_self.to_vec(),
            deck_opp: deck_opp.to_vec(), n_decisions: 0, finished: false,
            reward: 0.0, error: None, last_obs_json: String::new(),
            last_sbi: String::new(), needs_action: false, select_player: -1,
        }
    }

    fn start(
        &mut self,
        battle_start_fn: FnBattleStart,
        get_battle_data_fn: FnGetBattleData,
        our_player: u8,
    ) {
        let mut cards = [0i32; 120];
        cards[..60].copy_from_slice(&self.deck_self);
        cards[60..].copy_from_slice(&self.deck_opp);

        let mut sd = StartData { battle_ptr: ptr::null_mut(), error_player: 0, error_type: 0 };
        unsafe { bridge::bridge_battle_start(&mut sd, battle_start_fn, cards.as_ptr()) };

        if sd.battle_ptr.is_null() || sd.error_player != -1 {
            self.finished = true;
            self.reward = -1.0;
            self.error = Some(format!(
                "BattleStart: ptr={:?} player={} type={}",
                sd.battle_ptr, sd.error_player, sd.error_type
            ));
            return;
        }
        self.ptr = sd.battle_ptr;
        self.fetch_obs(get_battle_data_fn, our_player);
    }

    fn fetch_obs(&mut self, get_battle_data_fn: FnGetBattleData, our_player: u8) {
        if self.finished || self.ptr.is_null() { return; }

        let mut sd = SerialData {
            json: ptr::null(), data: ptr::null(), count: 0, select_player: -1,
        };
        unsafe { bridge::bridge_get_battle_data(&mut sd, get_battle_data_fn, self.ptr) };

        if sd.json.is_null() {
            self.finished = true; self.reward = -1.0;
            self.error = Some("GetBattleData: null json".into()); return;
        }
        self.last_obs_json = unsafe { cstr_to_string(sd.json) };
        self.last_sbi = if sd.data.is_null() || sd.count <= 0 {
            String::new()
        } else {
            unsafe {
                std::str::from_utf8(std::slice::from_raw_parts(sd.data, sd.count as usize))
                    .unwrap_or("").to_string()
            }
        };
        self.select_player = sd.select_player;

        if self.last_obs_json.is_empty() {
            self.finished = true; self.reward = 0.0;
            self.error = Some("GetBattleData: empty json".into()); return;
        }

        let v: serde_json::Value = match serde_json::from_str(&self.last_obs_json) {
            Ok(v) => v,
            Err(_) => {
                self.finished = true; self.reward = -1.0;
                self.error = Some(format!("invalid JSON: {}...", &self.last_obs_json[..80.min(self.last_obs_json.len())]));
                return;
            }
        };

        let result = v.get("current").and_then(|c| c.get("result"))
            .and_then(|r| r.as_i64()).unwrap_or(-1);
        if result != -1 {
            self.finished = true;
            let mut r = match result { 0 => 1.0, 1 => -1.0, 2 => 0.0, _ => 0.0 };
            if our_player == 1 { r = -r; }
            self.reward = r;
            return;
        }

        if v.get("select").and_then(|s| s.as_object()).is_some() {
            self.needs_action = true;
        } else {
            // Auto-advance: no select field, call Select with empty picks
            self.needs_action = true; // will send empty picks
        }
    }

    fn step(
        &mut self, select_fn: FnSelect, get_battle_data_fn: FnGetBattleData,
        picks: &[i32], our_player: u8, max_decisions: usize,
    ) {
        if self.finished || self.ptr.is_null() || !self.needs_action { return; }
        let err = unsafe {
            bridge::bridge_select(select_fn, self.ptr, picks.as_ptr(), picks.len() as c_int)
        };
        if err != 0 {
            self.finished = true; self.reward = -1.0;
            self.error = Some(format!("Select error {}", err)); return;
        }
        // Count every step as one decision (both sides in self-play).
        // select_player points to the NEXT player after fetch_obs, so
        // it's unreliable for counting "our" decisions.
        self.n_decisions += 1;
        // Timeout: force-terminate games that drag on
        if self.n_decisions > max_decisions {
            self.finished = true; self.reward = -1.0;
            self.error = Some(format!("timeout: {} decisions", self.n_decisions));
            return;
        }
        self.needs_action = false;
        self.fetch_obs(get_battle_data_fn, our_player);
    }

    fn finish(&mut self, battle_finish_fn: FnBattleFinish) {
        if !self.ptr.is_null() {
            unsafe { bridge::bridge_battle_finish(battle_finish_fn, self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

// ── VecEnv ──────────────────────────────────────────────────────────────

pub struct VecEnv {
    battles: Vec<Battle>,
    config: VecEnvConfig,
    finished: VecDeque<FinishedGame>,
    last_poll_indices: Vec<usize>,
    // Cached function pointers (resolved once via libloading)
    battle_start_fn: FnBattleStart,
    get_battle_data_fn: FnGetBattleData,
    select_fn: FnSelect,
    battle_finish_fn: FnBattleFinish,
}

#[derive(Debug, Clone)]
pub struct FinishedGame {
    pub battle_idx: usize,
    pub reward: f32,
    pub n_decisions: usize,
    pub error: Option<String>,
}

impl VecEnv {
    pub fn new(
        lib_path: &str,
        deck_self: &[i32],
        deck_opp: &[i32],
        config: VecEnvConfig,
    ) -> Result<Self, String> {
        let cg = unsafe { CgLib::load(lib_path) }
            .map_err(|e| format!("CgLib::load: {e}"))?;

        // Get raw function pointers (the bridge will call them safely)
        let battle_start_fn: FnBattleStart = unsafe { std::mem::transmute(*cg.battle_start) };
        let get_battle_data_fn: FnGetBattleData = unsafe { std::mem::transmute(*cg.get_battle_data) };
        let select_fn: FnSelect = unsafe { std::mem::transmute(*cg.select) };
        let battle_finish_fn: FnBattleFinish = unsafe { std::mem::transmute(*cg.battle_finish) };

        let mut battles = Vec::with_capacity(config.n_envs);
        for _ in 0..config.n_envs {
            let mut b = Battle::new(deck_self, deck_opp);
            b.start(battle_start_fn, get_battle_data_fn, config.our_player);
            battles.push(b);
        }

        let n = battles.len();
        let active = battles.iter().filter(|b| !b.finished).count();
        _trace(&format!("VecEnv::new: {} battles, {} active\n", n, active));
        Ok(VecEnv {
            battles, config,
            finished: VecDeque::new(),
            last_poll_indices: Vec::new(),
            battle_start_fn, get_battle_data_fn, select_fn, battle_finish_fn,
        })
    }

    pub fn poll(&mut self) -> Vec<PendingObs> {
        let bf = self.battle_finish_fn;
        let bs = self.battle_start_fn;
        let gb = self.get_battle_data_fn;
        let our = self.config.our_player;

        // ── Parallel: restart finished + advance all active battles ───
        let results: Vec<PollResult> = self.battles
            .par_iter_mut()
            .enumerate()
            .map(|(i, b)| {
                // If finished normally, queue for drain BEFORE restarting
                if b.finished && b.error.is_none() {
                    return PollResult { battle_idx: i,
                        finished_reward: Some(b.reward),
                        finished_n_dec: Some(b.n_decisions),
                        pending: None };
                }
                // Restart error battles
                if b.finished {
                    b.finish(bf);
                    let ds = b.deck_self.clone();
                    let dp = b.deck_opp.clone();
                    *b = Battle::new(&ds, &dp);
                    b.start(bs, gb, our);
                }
                // Collect observation if battle needs action
                if !b.finished && b.needs_action {
                    PollResult { battle_idx: i,
                        finished_reward: None, finished_n_dec: None,
                        pending: Some(PendingObs {
                            battle_idx: i,
                            obs_json: std::mem::take(&mut b.last_obs_json),
                            sbi: std::mem::take(&mut b.last_sbi),
                            select_player: b.select_player,
                        }) }
                } else {
                    PollResult { battle_idx: i,
                        finished_reward: None, finished_n_dec: None,
                        pending: None }
                }
            })
            .collect();

        // ── Drain finished games that completed this cycle ──────────
        for r in &results {
            if let Some(rew) = r.finished_reward {
                self.finished.push_back(FinishedGame {
                    battle_idx: r.battle_idx,
                    reward: rew, n_decisions: r.finished_n_dec.unwrap_or(0), error: None,
                });
                let b = &mut self.battles[r.battle_idx];
                b.finish(self.battle_finish_fn);
                let ds = b.deck_self.clone();
                let dp = b.deck_opp.clone();
                *b = Battle::new(&ds, &dp);
                b.start(self.battle_start_fn, self.get_battle_data_fn, self.config.our_player);
            }
        }

        // ── Sequential: drain errors, build ordered output ────────────
        for b in self.battles.iter_mut().enumerate() {
            if b.1.finished && b.1.error.is_some() {
                self.finished.push_back(FinishedGame {
                    battle_idx: b.0,
                    reward: b.1.reward, n_decisions: 0, error: b.1.error.take(),
                });
            }
        }

        let mut out = Vec::new();
        self.last_poll_indices.clear();
        for r in results {
            if let Some(obs) = r.pending {
                self.last_poll_indices.push(r.battle_idx);
                out.push(obs);
            }
        }
        let n_finished = self.battles.iter().filter(|b| b.finished).count();
        let n_needs = self.battles.iter().filter(|b| !b.finished && b.needs_action).count();
        _trace(&format!("poll: {} battles, {} fin, {} need\n",
            self.battles.len(), n_finished, n_needs));
        out
    }

    pub fn reply(&mut self, picks: &[Vec<i32>]) {
        _append(&format!("reply: {} picks, {} indices\n",
            picks.len(), self.last_poll_indices.len()));
        let sf = self.select_fn;
        let gb = self.get_battle_data_fn;
        let our = self.config.our_player;
        let indexed: Vec<(usize, &[i32])> = self.last_poll_indices
            .iter().zip(picks.iter()).map(|(&bi, p)| (bi, p.as_slice())).collect();

        // Parallel step: each battle's Select + GetBattleData is independent.
        // We iterate all battles in parallel and check if this battle has picks.
        self.battles.par_iter_mut().enumerate().for_each(|(bi, b)| {
            if let Some((_, pick_list)) = indexed.iter().find(|(idx, _)| *idx == bi) {
                b.step(sf, gb, pick_list, our, self.config.max_decisions);
            }
        });
    }

    pub fn drain(&mut self) -> Vec<FinishedGame> {
        let mut out: Vec<FinishedGame> = self.finished.drain(..).collect();
        _trace(&format!(
            "drain: {} finished games, {} more with errors\n",
            out.len(),
            self.battles.iter().filter(|b| b.finished && b.error.is_some()).count(),
        ));
        for (bi, b) in self.battles.iter_mut().enumerate() {
            if b.finished && b.error.is_some() {
                out.push(FinishedGame {
                    battle_idx: bi,
                    reward: b.reward, n_decisions: 0, error: b.error.take(),
                });
                b.finish(self.battle_finish_fn);
                let ds = b.deck_self.clone();
                let dp = b.deck_opp.clone();
                *b = Battle::new(&ds, &dp);
                b.start(self.battle_start_fn, self.get_battle_data_fn, self.config.our_player);
            }
        }
        out
    }
}

// ── Public types ────────────────────────────────────────────────────────

struct PollResult {
    battle_idx: usize,
    finished_reward: Option<f32>,
    finished_n_dec: Option<usize>,
    pending: Option<PendingObs>,
}

#[derive(Debug, Clone)]
pub struct PendingObs {
    pub battle_idx: usize,
    pub obs_json: String,
    pub sbi: String,
    pub select_player: i32,
}
