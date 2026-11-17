//! Tokenizers mapping text and melodies to dense symbol ids and back.

use std::collections::BTreeMap;

/// Character-level tokenizer; the vocabulary is the sorted set of distinct
/// characters in the training text, so ids are deterministic.
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    chars: Vec<char>,
    index: BTreeMap<char, usize>,
}

impl CharTokenizer {
    pub fn fit(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        let index = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharTokenizer { chars, index }
    }

    pub fn vocab(&self) -> usize {
        self.chars.len()
    }

    /// Characters outside the vocabulary are skipped (`None` for them would
    /// complicate streaming use); use [`CharTokenizer::encode_strict`] to
    /// detect them.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.index.get(&c).copied())
            .collect()
    }

    pub fn encode_strict(&self, text: &str) -> Option<Vec<usize>> {
        text.chars().map(|c| self.index.get(&c).copied()).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> Option<String> {
        ids.iter().map(|&i| self.chars.get(i).copied()).collect()
    }
}

const NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Melody tokenizer: note names like `C4`, `F#3` <-> MIDI-style pitch ids.
/// The id is `octave * 12 + semitone` for octaves 0..=8 (C0 = 0), so the
/// vocabulary is 108 and ordering follows pitch.
pub struct MelodyTokenizer;

impl MelodyTokenizer {
    pub const VOCAB: usize = 9 * 12;

    pub fn parse_note(note: &str) -> Option<usize> {
        let split = note.find(|c: char| c.is_ascii_digit())?;
        let (name, oct) = note.split_at(split);
        let semitone = NAMES.iter().position(|&n| n == name)?;
        let octave: usize = oct.parse().ok()?;
        (octave <= 8).then_some(octave * 12 + semitone)
    }

    pub fn note_name(id: usize) -> Option<String> {
        (id < Self::VOCAB).then(|| format!("{}{}", NAMES[id % 12], id / 12))
    }

    /// Whitespace-separated note names -> ids; `None` if any is invalid.
    pub fn encode(melody: &str) -> Option<Vec<usize>> {
        melody.split_whitespace().map(Self::parse_note).collect()
    }

    pub fn decode(ids: &[usize]) -> Option<String> {
        let names: Option<Vec<String>> = ids.iter().map(|&i| Self::note_name(i)).collect();
        names.map(|n| n.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn char_round_trip_random_strings() {
        let alphabet: Vec<char> = "abc xyz,.é漢".chars().collect();
        let mut rng = Rng::new(11);
        for _ in 0..200 {
            let len = (rng.next_u64() % 40) as usize;
            let s: String = (0..len)
                .map(|_| alphabet[(rng.next_u64() % alphabet.len() as u64) as usize])
                .collect();
            let tok = CharTokenizer::fit(&s);
            let ids = tok.encode(&s);
            assert_eq!(ids.len(), s.chars().count());
            assert!(ids.iter().all(|&i| i < tok.vocab()));
            assert_eq!(tok.decode(&ids).unwrap(), s);
        }
    }

    #[test]
    fn char_unknown_handling() {
        let tok = CharTokenizer::fit("abc");
        assert_eq!(tok.encode("abz"), vec![0, 1]);
        assert_eq!(tok.encode_strict("abz"), None);
        assert_eq!(tok.decode(&[0, 9]), None);
    }

    #[test]
    fn melody_round_trip_all_ids() {
        for id in 0..MelodyTokenizer::VOCAB {
            let name = MelodyTokenizer::note_name(id).unwrap();
            assert_eq!(MelodyTokenizer::parse_note(&name), Some(id), "{name}");
        }
        let ids = MelodyTokenizer::encode("C4 E4 G4 F#3").unwrap();
        assert_eq!(ids, vec![48, 52, 55, 42]);
        assert_eq!(MelodyTokenizer::decode(&ids).unwrap(), "C4 E4 G4 F#3");
    }

    #[test]
    fn melody_rejects_garbage() {
        for bad in ["H4", "C", "C9", "C#x", "4", "c4", "C-1"] {
            assert_eq!(MelodyTokenizer::parse_note(bad), None, "{bad}");
        }
        assert_eq!(MelodyTokenizer::encode("C4 nope"), None);
        assert_eq!(MelodyTokenizer::note_name(108), None);
    }
}
