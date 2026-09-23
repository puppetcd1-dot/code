use std::io::Write;

use hashbrown::HashMap;
use icicle_vm::VmExit;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, WeightedAliasIndex};

use crate::{
    calculate_energy, config,
    contract::{Action, Contract},
    input::{MultiStream, StreamKey},
    mutations::{self, Mutation, ALL_MUTATIONS},
    queue::CorpusStore,
    utils::{get_non_empty_streams, get_stream_weights, random_bytes},
    DictionaryRef, Fuzzer, Snapshot, StageData, StageExit,
};

/// Positional, length-preserving mutations.
///
/// Used on a stream whose bits a confirmed gate fixes: an insertion or a removal
/// moves every byte after it, and the position the gate was measured at would stop
/// naming the byte it describes.
const POSITIONAL_MUTATIONS: &[Mutation] = &[
    Mutation::BitFlip,
    Mutation::IncDec,
    Mutation::ReplaceByte,
    Mutation::InterestingValue,
    Mutation::DictReplace,
];

/// How often a stream with a semantic field gets an evidence-driven action instead
/// of a random one.
///
/// Deliberately not 1.0: filling every field in every round is the "wall of
/// delimiters" failure mode, and the random mutations are what explore around what
/// the evidence states.
const SEMANTIC_PROBABILITY: f32 = 0.35;

/// A fuzzing stage that applies random mutations to the input.
pub(crate) struct HavocStage {
    attempts: u32,
    mutator: HavocMutator,

    streams: Vec<(StreamKey, usize)>,
    stream_distr: WeightedAliasIndex<f64>,
    streams_to_extend: HashMap<StreamKey, usize>,

    log2_max_mutations: u32,
    max_mutations: u32,
    saved: bool,

    /// The role map, as the mutator asks it questions.
    contract: Contract,
    /// The input this round is mutating, for the record.
    input_id: usize,
    /// Which contract the round ran against.
    ///
    /// In the log rather than only in the start-up banner: the file is appended to
    /// across runs, so two runs with different role maps end up in one document and
    /// the banner of the second says nothing about the first.  (Two runs of this
    /// pipeline were once compared while each had loaded the *other* seed's role
    /// map -- the per-stream payload bytes were the only fingerprint, and nothing in
    /// the log said so.)
    contract_label: String,
    /// One line per evidence-driven action, one summary line per round.
    log: Option<std::io::BufWriter<std::fs::File>>,
    counts: RoundCounts,
}

/// What a round spent its budget on.
///
/// A summary per round rather than a line per mutation: the per-mutation detail is
/// already in the corpus metadata (`MutationKind::Evidence`), and this is the number
/// that says whether the contract changed the split at all.
#[derive(Default)]
struct RoundCounts {
    random: u32,
    magic_refill: u32,
    length_boundary: u32,
    length_joint: u32,
    protected_bits: u32,
    skipped_register: u32,
    /// Mutations per stream, so the report can say where the budget went next to
    /// the byte budget (`channel` vs `register`).
    per_stream: HashMap<StreamKey, u32>,
}

impl RoundCounts {
    fn note(&mut self, key: StreamKey) {
        *self.per_stream.entry(key).or_default() += 1;
    }
}

impl Drop for HavocStage {
    fn drop(&mut self) {
        let Some(log) = self.log.as_mut() else { return };
        let counts = &self.counts;
        let _ = writeln!(
            log,
            "{{\"input\": {}, \"contract\": {}, \"random\": {}, \"magic_refill\": {}, \
             \"length_boundary\": {}, \"length_joint\": {}, \"protected_bits\": {}, \
             \"skipped_register\": {}}}",
            self.input_id,
            serde_json::to_string(&self.contract_label).unwrap_or_else(|_| "\"?\"".into()),
            counts.random,
            counts.magic_refill,
            counts.length_boundary,
            counts.length_joint,
            counts.protected_bits,
            counts.skipped_register
        );
        // Per stream, with the class and the payload bytes the analysis marked: the
        // share the contract produced, next to the share the budget predicted.
        let mut streams: Vec<(&StreamKey, &u32)> = counts.per_stream.iter().collect();
        streams.sort_unstable();
        for (key, mutations) in streams {
            // The length the round started with: the input may have grown since, and
            // "how much of this stream could a mutation change" is a question about
            // what was there when the budget was spent.
            let len = self
                .streams
                .iter()
                .find(|(stream, _)| *stream == *key)
                .map(|(_, len)| *len)
                .unwrap_or(0);
            // The protected (stream, offset span, mask) triples: the span is what the
            // mutator pins, the mask is the bit the firmware branches on.
            let gates: Vec<String> = self
                .contract
                .protected_spans(*key)
                .into_iter()
                .map(|(start, end, mask)| format!("\"{start}-{end}:{mask:#x}\""))
                .collect();
            let _ = writeln!(
                log,
                "{{\"input\": {}, \"stream\": \"{:#x}\", \"class\": \"{}\", \
                 \"mutations\": {}, \"payload_bytes\": {}, \"bytes\": {}, \"pinned\": {}, \
                 \"gates\": [{}]}}",
                self.input_id,
                key,
                self.contract.class(*key).name(),
                mutations,
                self.contract.payload_bytes(*key),
                len,
                self.contract.pinned_bytes(*key, len),
                gates.join(", ")
            );
        }
    }
}

impl StageData for HavocStage {
    fn start(fuzzer: &mut Fuzzer) -> Result<Self, StageExit> {
        fuzzer.copy_current_input();

        let (Some(id), data) = (fuzzer.input_id, &fuzzer.state.input)
        else {
            return Err(StageExit::Unsupported);
        };

        let streams = get_non_empty_streams(data);
        if streams.is_empty() {
            // Must have at least one non-empty stream.
            return Err(StageExit::Skip);
        }

        let stream_distr = get_stream_weights(fuzzer, id, &streams);
        let mutator = HavocMutator::new();
        let attempts = calculate_energy(fuzzer) as u32;
        let log2_max_mutations = log2_max_mutations(fuzzer);
        let max_mutations = max_mutations(fuzzer);

        fuzzer.corpus[id].metadata.havoc_rounds += 1;

        // The contract: the workdir's role map unless `TAINT_CONTRACT` says
        // otherwise.  `TAINT_CONTRACT=0` is the baseline arm -- the same binary with
        // no role map at all.
        let contract_path = contract_path(fuzzer);
        let contract = match &contract_path {
            Some(path) => Contract::load(path),
            None => Contract::empty(),
        };
        let contract_label = contract_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "baseline".to_string());
        let log = open_action_log(fuzzer, &contract);
        announce_first_round(fuzzer, contract_path.as_deref(), &contract);

        tracing::trace!(
            "[{id}] havoc for {attempts} attempts with {} max mutations",
            2_u64.pow(log2_max_mutations)
        );
        Ok(Self {
            attempts,
            streams,
            stream_distr,
            mutator,
            log2_max_mutations,
            max_mutations,
            streams_to_extend: HashMap::new(),
            saved: false,
            contract,
            input_id: id,
            contract_label,
            log,
            counts: RoundCounts::default(),
        })
    }

    fn fuzz_one(&mut self, fuzzer: &mut Fuzzer) -> Option<VmExit> {
        self.attempts = self.attempts.checked_sub(1)?;

        Snapshot::restore_initial(fuzzer);
        fuzzer.copy_current_input();
        fuzzer.reset_input_cursor().unwrap();

        self.havoc_v1(fuzzer);

        // Also extend any streams that have caused us to exit because there were too small, these
        // streams will be trimmed back to the correct length as part of `auto_trim_input` if the
        // extension was unnecessary
        let data = &mut fuzzer.state.input;
        for (key, count) in &self.streams_to_extend {
            let bytes = &mut data.streams.entry(*key).or_default().bytes;
            if bytes.len() >= config::MAX_STREAM_LEN {
                continue;
            }

            let local_dict = fuzzer.dict.entry(*key).or_default();
            local_dict.compute_weights();
            let dict = DictionaryRef { local: local_dict, global: &fuzzer.global_dict };
            mutations::extend_input_by(&mut fuzzer.rng, dict, bytes, 4 * count);
        }

        fuzzer.write_input_to_target().unwrap();
        let exit = fuzzer.execute()?;

        // Keep track of the streams that cause us to exit because they are too small.
        if let Some(key) = fuzzer.state.input.last_read {
            *self.streams_to_extend.entry(key).or_default() += 1;
        }

        fuzzer.auto_trim_input().ok()?;

        if fuzzer.debug.havoc && !self.saved {
            let _ = std::fs::write(
                fuzzer.workdir.join(format!("queue/{}.havoc.bin", fuzzer.input_id.unwrap_or(0))),
                fuzzer.state.input.to_bytes(),
            );
            self.saved = true;
        }

        Some(exit)
    }
}

impl HavocStage {
    #[allow(unused)]
    fn havoc_v1(&mut self, fuzzer: &mut Fuzzer) {
        let data = &mut fuzzer.state.input;

        let mut mutations = crate::utils::rand_pow2(&mut fuzzer.rng, self.log2_max_mutations);
        let mut mutations = fuzzer.rng.gen_range(1..=self.max_mutations);
        while mutations > 0 {
            // Select a stream to mutate.  The distribution already encodes which
            // streams reached new code; the contract adds what the stream *is*, and
            // the two stay separable by taking this decision as a rejection.
            let mut key = None;
            for _ in 0..8 {
                let (candidate, _) = self.streams[self.stream_distr.sample(&mut fuzzer.rng)];
                let len = data.streams.get(&candidate).map(|s| s.bytes.len()).unwrap_or(0);
                if fuzzer.rng.gen_bool(self.contract.stream_factor(candidate, len)) {
                    key = Some(candidate);
                    break;
                }
                self.counts.skipped_register += 1;
            }
            let Some(key) = key else { break };
            let bytes = &mut data.streams.get_mut(&key).unwrap().bytes;

            let local_dict = fuzzer.dict.entry(key).or_default();
            local_dict.compute_weights();
            let dict = DictionaryRef { local: local_dict, global: &fuzzer.global_dict };

            // Avoid excess mutations for small streams.
            // let max_mutations_for_stream = match bytes.len() {
            //     ..=8 => 8,
            //     ..=128 => 32,
            //     _ => 64,
            // };
            let max_mutations_for_stream = mutations;
            // Consume some propotion of the total number of mutations on the current stream.
            let num_mutations = fuzzer.rng.gen_range(1..=mutations.min(max_mutations_for_stream));
            mutations -= num_mutations;
            for _ in 0..num_mutations {
                // Layer 1: the bits a confirmed gate fixes are not the mutator's to
                // set.  The values are taken before the mutation and put back after
                // it, so a mutation that also changed the rest of the byte survives:
                // the gate is about the bit, not about the byte.
                let guard = self.contract.guard(key, bytes);

                // Layer 2: with probability, act on the evidence instead of at
                // random -- a random byte mutation cannot express "write the constant
                // this comparison expects, at this position".
                if let Some((action, offset, value)) =
                    self.apply_evidence(&mut fuzzer.rng, key, bytes)
                {
                    fuzzer.state.mutation_kinds.push(crate::MutationKind::Evidence {
                        stream: key,
                        action,
                        offset,
                        value,
                    });
                    self.counts.note(key);
                }
                else {
                    // Layer 3/4: a random mutation -- positional when a bit of this
                    // stream is protected, free otherwise.
                    let mutation = if guard.is_empty() {
                        self.mutator.havoc_bytes(
                            &mut fuzzer.rng,
                            dict,
                            bytes,
                            key,
                            &fuzzer.corpus,
                            0,
                        )
                    }
                    else {
                        let kind = *POSITIONAL_MUTATIONS
                            .choose(&mut fuzzer.rng)
                            .unwrap_or(&Mutation::BitFlip);
                        mutations::apply_mutation(
                            kind,
                            &mut fuzzer.rng,
                            dict,
                            bytes,
                            key,
                            &fuzzer.corpus,
                            0,
                        );
                        Some(kind)
                    };
                    if let Some(mutation) = mutation {
                        fuzzer.state.mutation_kinds.push((key, mutation).into());
                        self.counts.random += 1;
                        self.counts.note(key);
                    }
                }

                // Layer 1 has the last word, whichever layer acted: a refill that
                // landed on a protected bit (a constant sharing a position with a
                // gate) is put back, which is the same precedence the report uses
                // when a gate and a magic collide.
                if !guard.is_empty() {
                    self.counts.protected_bits +=
                        Contract::restore_protected(bytes, &guard) as u32;
                }
            }
        }
    }

    #[allow(unused)]
    fn havoc_v2(&mut self, fuzzer: &mut Fuzzer) {
        let Some(input_id) = fuzzer.input_id
        else {
            return;
        };
        let max_find_gap = fuzzer.corpus[input_id].metadata.max_find_gap;

        let data = &mut fuzzer.state.input;
        for &(key, _) in &self.streams {
            // Decided whether this stream should be mutated.
            if fuzzer.rng.gen_bool(0.5) {
                continue;
            }

            let bytes = &mut data.streams.get_mut(&key).unwrap().bytes;
            let local_dict = fuzzer.dict.entry(key).or_default();
            local_dict.compute_weights();
            let dict = DictionaryRef { local: local_dict, global: &fuzzer.global_dict };

            let mutations = fuzzer.rng.gen_range(1..=self.log2_max_mutations);
            for _ in 0..mutations {
                self.mutator.havoc_bytes(&mut fuzzer.rng, dict, bytes, key, &fuzzer.corpus, 0);
            }
        }
    }

    /// Ask the contract for an evidence-driven action and apply it.
    ///
    /// Returns what to record in the corpus metadata, or `None` to fall through to a
    /// random mutation.  Which layer acts is decided by what the stream has: a
    /// channel with a constant at a position gets a refill, a stream with a length
    /// field gets a boundary, and a stream with neither gets nothing.
    fn apply_evidence<R: Rng>(
        &mut self,
        rng: &mut R,
        key: StreamKey,
        bytes: &mut Vec<u8>,
    ) -> Option<(crate::contract::ActionKind, u32, u64)> {
        if self.contract.is_empty() || !rng.gen_bool(f64::from(SEMANTIC_PROBABILITY)) {
            return None;
        }
        let action = if rng.gen_bool(0.5) {
            self.contract.pick_magic(rng, key, bytes.len()).map(Action::Magic)
        }
        else {
            self.contract.pick_length(rng, key, bytes).map(Action::Length)
        };
        let applied = self.contract.apply(rng, action?, bytes)?;
        match applied.0 {
            crate::contract::ActionKind::MagicRefill => self.counts.magic_refill += 1,
            crate::contract::ActionKind::LengthBoundary => self.counts.length_boundary += 1,
            crate::contract::ActionKind::LengthJoint => self.counts.length_joint += 1,
        }
        Some(applied)
    }
}

/// Where the contract lives.
///
/// Searched in order, and the one that was used is logged at info level, because
/// "the contract silently did not load" is indistinguishable from "the contract
/// changed nothing":
///
///   1. `TAINT_CONTRACT=<path>` -- a file, or a directory holding `role_map.json`;
///      `TAINT_CONTRACT=0` means no contract at all, which is the baseline arm (the
///      same binary with no role map, so the ablation is a runtime choice);
///   2. `$WORKDIR/role_map.json` -- what an analysis run under this workdir writes;
///   3. `./role_map.json` -- where a `REPLAY=` run actually writes it, which is the
///      directory the analysis was run from rather than the fuzzer's workdir.
///
/// Returns `None` when the baseline was asked for or nothing was found.
fn contract_path(fuzzer: &Fuzzer) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    match std::env::var("TAINT_CONTRACT") {
        Ok(value) if value == "0" => return None,
        Ok(value) => {
            let path = std::path::PathBuf::from(value);
            if path.is_dir() {
                candidates.push(path.join("role_map.json"));
            }
            else {
                candidates.push(path);
            }
        }
        Err(_) => {}
    }
    candidates.push(fuzzer.workdir.join("role_map.json"));
    candidates.push(std::path::PathBuf::from("role_map.json"));
    for path in &candidates {
        if path.is_file() {
            tracing::info!("using the contract at {}", path.display());
            return Some(path.clone());
        }
    }
    tracing::debug!(
        "no contract in {:?}: baseline havoc (set TAINT_CONTRACT to a role_map.json)",
        candidates
    );
    None
}

/// Append-mode log of what the contract decided, next to the role map it read.
fn open_action_log(fuzzer: &Fuzzer, contract: &Contract) -> Option<std::io::BufWriter<std::fs::File>> {
    if contract.is_empty() {
        return None;
    }
    let path = fuzzer.workdir.join("contract_mutations.jsonl");
    match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(file) => Some(std::io::BufWriter::new(file)),
        Err(err) => {
            tracing::warn!("cannot write the contract log at {}: {}", path.display(), err);
            None
        }
    }
}

/// Say, once per process, whether this run is under a contract and where its log is.
///
/// Printed unconditionally rather than through `tracing`, because the whole failure
/// mode this answers -- "the log file is not there" -- is indistinguishable from
/// "the mutator never ran": the file is only created when a havoc round starts with
/// a contract loaded, and neither of those two events is visible in the artifacts
/// afterwards.  If this line is missing, havoc never started (a `REPLAY=` run, an
/// empty corpus, or an earlier stage); if it says `contract=none`, the path search
/// failed and the candidates are listed; if it names a file, the log is at the path
/// it prints.
fn announce_first_round(fuzzer: &Fuzzer, path: Option<&std::path::Path>, contract: &Contract) {
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    ANNOUNCED.call_once(|| {
        match path {
            Some(path) => eprintln!(
                "[contract] first havoc round: workdir={} contract={} ({} stream(s))",
                fuzzer.workdir.display(),
                path.display(),
                contract.stream_count()
            ),
            None => eprintln!(
                "[contract] first havoc round: workdir={} contract=none -- baseline havoc \
                 (searched TAINT_CONTRACT, {}/role_map.json, ./role_map.json)",
                fuzzer.workdir.display(),
                fuzzer.workdir.display()
            ),
        }
        eprintln!(
            "[contract] actions go to {}/contract_mutations.jsonl (empty contract: no file)",
            fuzzer.workdir.display()
        );
    });
}

/// Determine the maximum number of mutations to try. This number increases the longer it takes to
/// find new inputs.
///
/// @todo: these ranges were selected to be similar the havoc stacking factor used by AFL++, but it
/// is possible that there are better values.
fn log2_max_mutations(fuzzer: &Fuzzer) -> u32 {
    let max_find_gap =
        fuzzer.input_id.map(|id| fuzzer.corpus[id].metadata.max_find_gap).unwrap_or(0);
    match max_find_gap {
        ..=1000 => 3,
        ..=10000 => 4,
        ..=100000 => 5,
        ..=1000000 => 6,
        ..=10000000 => 7,
        _ => 8,
    }
}

fn max_mutations(fuzzer: &Fuzzer) -> u32 {
    let max_find_gap =
        fuzzer.input_id.map(|id| fuzzer.corpus[id].metadata.max_find_gap).unwrap_or(0);
    match max_find_gap {
        ..=100 => 4,
        ..=1000 => 8,
        ..=10000 => 16,
        ..=100000 => 32,
        _ => 64,
    }
}

fn mutation_weight(mutation: &Mutation) -> u32 {
    return match mutation {
        Mutation::BitFlip => 20,
        Mutation::ReplaceByte => 40,
        Mutation::IncDec => 10,
        Mutation::InsertByte => 10,
        Mutation::Insert4 => 10,
        Mutation::RemoveByte => 5,
        Mutation::Remove4 => 5,
        Mutation::InterestingValue => 10,
        Mutation::DictReplace => 20,
        Mutation::DictInsert => 20,
        Mutation::StreamSplice => 5,
        Mutation::InnerSplice => 5,
        Mutation::RandomSplice => 5,
        Mutation::RemoveRegion => 1,
    };
    // match mutation {
    //     Mutation::BitFlip => 1,
    //     Mutation::IncDec => 1,

    //     Mutation::ReplaceByte => 4,
    //     Mutation::InsertByte => 4,
    //     Mutation::Insert4 => 4,

    //     Mutation::InterestingValue => 2,
    //     Mutation::DictReplace => 4,
    //     Mutation::DictInsert => 4,

    //     Mutation::StreamSplice => 4,
    //     Mutation::InnerSplice => 4,
    //     Mutation::RandomSplice => 1,

    //     Mutation::RemoveByte => 2,
    //     Mutation::Remove4 => 1,
    //     Mutation::RemoveRegion => 1,
    // }
}

pub struct HavocMutator {
    mutation_distr: WeightedAliasIndex<u32>,
}

impl HavocMutator {
    pub(crate) fn new() -> Self {
        let weights = ALL_MUTATIONS.iter().map(mutation_weight).collect();
        Self { mutation_distr: WeightedAliasIndex::new(weights).unwrap() }
    }

    #[allow(unused)]
    fn random_weights<R: Rng>(rng: &mut R) -> Self {
        let weights = ALL_MUTATIONS.iter().map(|_| rng.gen_range(0..100)).collect();
        let Ok(mutation_distr) = WeightedAliasIndex::new(weights)
        else {
            return HavocMutator::new();
        };
        Self { mutation_distr }
    }

    pub(crate) fn havoc_bytes<R>(
        &self,
        rng: &mut R,
        dict: DictionaryRef,
        input: &mut Vec<u8>,
        key: StreamKey,
        corpus: &CorpusStore<MultiStream>,
        offset: usize,
    ) -> Option<Mutation>
    where
        R: Rng,
    {
        if input.is_empty() {
            random_bytes(rng, input);
            return None;
        }

        let mutation = ALL_MUTATIONS[self.mutation_distr.sample(rng)];
        mutations::apply_mutation(mutation, rng, dict, input, key, corpus, offset);

        if input.is_empty() {
            random_bytes(rng, input);
            return None;
        }

        Some(mutation)
    }
}
