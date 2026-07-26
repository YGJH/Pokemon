/// Hidden-state guessing for search_begin parameters.
///
/// The engine's search API requires us to supply guesses for all hidden
/// information (opponent hand/deck/prize/active, and your own deck/prize).
/// This module provides heuristics to generate plausible guesses from the
/// current observation.

use rand::seq::SliceRandom;
use rand::Rng;
use serde_json::Value;

/// Guesses for all hidden-state parameters needed by `search_begin`.
#[derive(Debug, Clone)]
pub struct HiddenGuesses {
    pub your_deck: Vec<i32>,
    pub your_prize: Vec<i32>,
    pub opponent_deck: Vec<i32>,
    pub opponent_prize: Vec<i32>,
    pub opponent_hand: Vec<i32>,
    pub opponent_active: Vec<i32>,
    /// Whether the opponent's active Pokémon is face-down (needs guessing).
    #[allow(dead_code)]
    pub opp_active_face_down: bool,
}

/// Build guesses from the observation JSON and known deck lists.
///
/// # Arguments
///
/// * `obs_json` — The raw observation dict as a JSON string (from `agent()`).
/// * `fixed_deck` — Our 60-card FIXED_DECK (card IDs).
/// * `opponent_deck_template` — A plausible 60-card opponent deck list.
///   If empty, uses a copy of `fixed_deck` as a fallback heuristic.
/// * `rng` — Seeded random number generator for reproducible guesses.
pub fn build_guesses(
    obs_json: &str,
    fixed_deck: &[i32],
    opponent_deck_template: &[i32],
    rng: &mut impl Rng,
) -> Result<HiddenGuesses, String> {
    let obs: Value =
        serde_json::from_str(obs_json).map_err(|e| format!("obs parse: {e}"))?;

    let current = obs
        .get("current")
        .ok_or("obs has no 'current'")?;
    let your_index = current["yourIndex"].as_i64().unwrap_or(0) as usize;
    let opp_index = 1 - your_index;

    let your_player = &current["players"][your_index];
    let opp_player = &current["players"][opp_index];

    // ── Your deck: subtract known cards from FIXED_DECK ──────────────────
    let mut known_your_cards: Vec<i32> = Vec::new();

    // Cards in hand
    if let Some(hand) = your_player["hand"].as_array() {
        for card in hand {
            if let Some(id) = card["id"].as_i64() {
                known_your_cards.push(id as i32);
            }
        }
    }
    // Cards in play (active + bench) — include pre-evolutions
    for poke in cards_from_pokemon_list(&your_player["active"]) {
        known_your_cards.push(poke);
    }
    for poke in cards_from_pokemon_list(&your_player["bench"]) {
        known_your_cards.push(poke);
    }
    // Cards in discard
    if let Some(disc) = your_player["discard"].as_array() {
        for card in disc {
            if let Some(id) = card["id"].as_i64() {
                known_your_cards.push(id as i32);
            }
        }
    }
    // Revealed prize cards
    if let Some(prizes) = your_player["prize"].as_array() {
        for card in prizes {
            if let Some(c) = card.as_object() {
                if let Some(id) = c.get("id").and_then(|v| v.as_i64()) {
                    known_your_cards.push(id as i32);
                }
            }
        }
    }

    let deck_count = your_player["deckCount"].as_i64().unwrap_or(0) as usize;
    let unrevealed_prize_count = unrevealed_count(&your_player["prize"]);

    // Build your_deck: remaining unknown cards = FIXED_DECK minus known cards
    let mut deck_remaining = subtract_multiset(fixed_deck, &known_your_cards);
    // Shuffle
    deck_remaining.shuffle(rng);

    let your_deck: Vec<i32> = deck_remaining[..deck_remaining.len().min(deck_count)].to_vec();

    // Your prizes: the tail of the same shuffled pool the deck was dealt from.
    // This used to clone `deck_remaining`, which let one card land in both the
    // deck and the prizes -- the determinized world then held more copies of it
    // than the decklist allows, and every rollout that drew it explored a game
    // that cannot happen.  Taking the leftover keeps the two disjoint, and since
    // the pool was already shuffled the slice is still a uniform sample.
    let leftover = &deck_remaining[your_deck.len()..];
    let your_prize: Vec<i32> = if unrevealed_prize_count > 0 && !leftover.is_empty() {
        leftover[..leftover.len().min(unrevealed_prize_count)].to_vec()
    } else {
        vec![0i32; unrevealed_prize_count] // fallback: will be ignored if len matches
    };

    // ── Opponent guesses ──────────────────────────────────────────────────
    let opp_template: Vec<i32> = if opponent_deck_template.is_empty() {
        fixed_deck.to_vec() // fallback: assume mirror
    } else {
        opponent_deck_template.to_vec()
    };

    let opp_deck_count = opp_player["deckCount"].as_i64().unwrap_or(0) as usize;
    let opp_hand_count = opp_player["handCount"].as_i64().unwrap_or(0) as usize;
    let opp_prize_count = opp_player["prize"].as_array().map(|a| a.len()).unwrap_or(6);

    // Known opponent cards (revealed)
    let mut known_opp_cards: Vec<i32> = Vec::new();
    for poke in cards_from_pokemon_list(&opp_player["active"]) {
        known_opp_cards.push(poke);
    }
    for poke in cards_from_pokemon_list(&opp_player["bench"]) {
        known_opp_cards.push(poke);
    }
    if let Some(disc) = opp_player["discard"].as_array() {
        for card in disc {
            if let Some(id) = card["id"].as_i64() {
                known_opp_cards.push(id as i32);
            }
        }
    }
    if let Some(prizes) = opp_player["prize"].as_array() {
        for card in prizes {
            if let Some(c) = card.as_object() {
                if let Some(id) = c.get("id").and_then(|v| v.as_i64()) {
                    known_opp_cards.push(id as i32);
                }
            }
        }
    }

    let mut opp_unknown = subtract_multiset(&opp_template, &known_opp_cards);
    opp_unknown.shuffle(rng);

    let opponent_deck: Vec<i32> =
        opp_unknown[..opp_unknown.len().min(opp_deck_count)].to_vec();

    // Opponent hand: random sample from remaining unknown
    let opp_hand_start = opp_deck_count.min(opp_unknown.len());
    let opponent_hand: Vec<i32> = if opp_hand_count > 0 {
        let pool = &opp_unknown[opp_hand_start..];
        let mut hand_pool = pool.to_vec();
        hand_pool.shuffle(rng);
        hand_pool[..hand_pool.len().min(opp_hand_count)].to_vec()
    } else {
        Vec::new()
    };

    // Opponent prizes
    let opponent_prize: Vec<i32> = if opp_prize_count > 0 && opp_unknown.len() > opp_hand_start + opp_hand_count {
        let prize_pool = &opp_unknown[opp_hand_start + opp_hand_count..];
        let mut pp = prize_pool.to_vec();
        pp.shuffle(rng);
        pp[..pp.len().min(opp_prize_count)].to_vec()
    } else {
        vec![0i32; opp_prize_count]
    };

    // Opponent active: only needed if face-down (active[0] is null/None)
    let opp_active_face_down = opp_player["active"]
        .as_array()
        .and_then(|a| a.first())
        .map(|v| v.is_null())
        .unwrap_or(false);

    let opponent_active: Vec<i32> = if opp_active_face_down {
        // Pick a basic Pokémon from the opponent template
        basic_pokemon_from_deck(&opp_template, rng)
            .map(|id| vec![id])
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    Ok(HiddenGuesses {
        your_deck,
        your_prize,
        opponent_deck,
        opponent_prize,
        opponent_hand,
        opponent_active,
        opp_active_face_down,
    })
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Extract card IDs from a JSON array of Pokémon objects (including pre-evolutions,
/// energy cards, and tools attached to them).
fn cards_from_pokemon_list(arr: &Value) -> Vec<i32> {
    let mut ids = Vec::new();
    if let Some(list) = arr.as_array() {
        for poke in list {
            ids.extend(cards_from_one_pokemon(poke));
        }
    }
    ids
}

fn cards_from_one_pokemon(poke: &Value) -> Vec<i32> {
    let mut ids = Vec::new();
    if let Some(obj) = poke.as_object() {
        if let Some(id) = obj.get("id").and_then(|v| v.as_i64()) {
            ids.push(id as i32);
        }
        // Pre-evolution cards
        if let Some(pre) = obj.get("preEvolution").and_then(|v| v.as_array()) {
            for card in pre {
                if let Some(id) = card.get("id").and_then(|v| v.as_i64()) {
                    ids.push(id as i32);
                }
            }
        }
        // Attached tools
        if let Some(tools) = obj.get("tools").and_then(|v| v.as_array()) {
            for card in tools {
                if let Some(id) = card.get("id").and_then(|v| v.as_i64()) {
                    ids.push(id as i32);
                }
            }
        }
        // Attached energy cards
        if let Some(ec) = obj.get("energyCards").and_then(|v| v.as_array()) {
            for card in ec {
                if let Some(id) = card.get("id").and_then(|v| v.as_i64()) {
                    ids.push(id as i32);
                }
            }
        }
    }
    ids
}

/// Count how many prize slots are None (unrevealed).
fn unrevealed_count(prize_arr: &Value) -> usize {
    prize_arr
        .as_array()
        .map(|a| a.iter().filter(|v| v.is_null()).count())
        .unwrap_or(0)
}

/// Subtract `known` multiset from `deck` multiset, returning the remaining cards.
/// Preserves multiplicities: if deck has 4 copies of id X and known has 2,
/// the result has 2 copies.
fn subtract_multiset(deck: &[i32], known: &[i32]) -> Vec<i32> {
    use std::collections::HashMap;
    let mut counts: HashMap<i32, i32> = HashMap::new();
    for id in deck {
        *counts.entry(*id).or_insert(0) += 1;
    }
    for id in known {
        *counts.entry(*id).or_insert(0) -= 1;
    }
    let mut result = Vec::new();
    for (id, count) in counts {
        for _ in 0..count.max(0) {
            result.push(id);
        }
    }
    result
}

/// Pick a basic Pokémon card ID from the given deck template.
fn basic_pokemon_from_deck(deck: &[i32], rng: &mut impl Rng) -> Option<i32> {
    // We don't have the static card data here to check `basic` flag.
    // Common heuristic: return the first card ID from the deck (most decks
    // are Pokémon-heavy, and the first card is likely a Pokémon).
    // A better approach would be to accept a pre-computed list of basic Pokémon
    // IDs, but for v1 this is sufficient.
    let candidates: Vec<&i32> = deck.iter().collect();
    candidates.choose(rng).copied().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mid-game observation used by several tests: we are player 0 with a
    /// 45-card deck, six unrevealed prizes and one known card in play.
    const OBS_MIRROR: &str = r#"{
            "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1, "option": []},
            "current": {
                "turn": 5, "turnActionCount": 3, "yourIndex": 0, "firstPlayer": 0,
                "supporterPlayed": false, "stadiumPlayed": false,
                "energyAttached": true, "retreated": false, "result": -1,
                "stadium": [],
                "looking": null,
                "players": [{
                    "active": [{"id": 101, "serial": 1, "hp": 150, "maxHp": 150, "appearThisTurn": false, "energies": [], "energyCards": [], "tools": [], "preEvolution": []}],
                    "bench": [],
                    "benchMax": 5, "deckCount": 45, "discard": [],
                    "prize": [null, null, null, null, null, null],
                    "handCount": 5, "hand": [{"id": 201, "serial": 2}, {"id": 202, "serial": 3}],
                    "poisoned": false, "burned": false, "asleep": false, "paralyzed": false, "confused": false
                }, {
                    "active": [null],
                    "bench": [],
                    "benchMax": 5, "deckCount": 47, "discard": [],
                    "prize": [null, null, null, null, null, null],
                    "handCount": 7, "hand": null,
                    "poisoned": false, "burned": false, "asleep": false, "paralyzed": false, "confused": false
                }]
            },
            "search_begin_input": "dummy123"
        }"#;

    #[test]
    fn test_subtract_multiset() {
        let mut result = subtract_multiset(&[1, 1, 2, 3], &[1, 2]);
        result.sort();
        assert_eq!(result, vec![1, 3]);
    }

    #[test]
    fn test_guesses_with_mirror_deck() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let obs = OBS_MIRROR;

        let fixed = (1..=60).collect::<Vec<i32>>();
        let result = build_guesses(obs, &fixed, &[], &mut rng).unwrap();

        assert!(result.opp_active_face_down);
        assert_eq!(result.opponent_active.len(), 1);
        assert_eq!(result.opponent_hand.len(), 7);
        assert_eq!(result.opponent_prize.len(), 6);
    }

    #[test]
    fn test_your_deck_and_prizes_are_disjoint() {
        // Prizes used to be drawn from the whole unknown pool rather than what
        // was left after the deck was dealt, so a card could sit in both at
        // once and the determinized world held more copies of it than the
        // decklist allows.  Every id here is distinct, so any card appearing
        // twice across deck+prizes is that overlap.
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let obs = OBS_MIRROR;
        let mut fixed = vec![101, 201, 202];
        fixed.extend(1..=57);
        assert_eq!(fixed.len(), 60);

        // Seed-swept: one seed could get lucky, the bug is probabilistic.
        for seed in 0..32u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let result = build_guesses(obs, &fixed, &[], &mut rng).unwrap();

            let mut all = result.your_deck.clone();
            all.extend(result.your_prize.iter().copied());
            let n = all.len();
            all.sort_unstable();
            all.dedup();
            assert_eq!(
                all.len(),
                n,
                "seed {seed}: a card is in both the deck and the prizes"
            );
        }
    }
}
