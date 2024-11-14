use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use solana_bpf_loader_program::syscalls::create_program_runtime_environment_v1;
use solana_compute_budget::compute_budget::ComputeBudget;
use solana_program_runtime::loaded_programs::{BlockRelation, ForkGraph, ProgramCacheEntry};
use solana_sdk::{
    account::{AccountSharedData, ReadableAccount},
    clock::Slot,
    feature_set::FeatureSet,
    fee::FeeStructure,
    hash::Hash,
    pubkey::Pubkey,
    rent_collector::RentCollector,
    transaction,
};
use solana_svm::{
    account_loader::CheckedTransactionDetails,
    transaction_processing_callback::{AccountState, TransactionProcessingCallback},
    transaction_processor::{
        LoadAndExecuteSanitizedTransactionsOutput, TransactionBatchProcessor,
        TransactionProcessingConfig, TransactionProcessingEnvironment,
    },
};
use solana_svm_transaction::svm_transaction::SVMTransaction;
use solana_system_program::system_processor;

pub struct ProgramSvmTest<'a> {
    accounts: ProgramSvmTestAccountLoader,
    fork_graph: Arc<RwLock<ProgramSvmTestForkGraph>>,
    compute_budget: ComputeBudget,
    feature_set: FeatureSet,
    fee_structure: FeeStructure,
    rent_collector: RentCollector,
    client: Arc<Option<ProgramSvmTestClient<'a>>>,
}

#[derive(Default)]
pub struct ProgramSvmTestClient<'a> {
    processing_environment: TransactionProcessingEnvironment<'a>,
    processor: TransactionBatchProcessor<ProgramSvmTestForkGraph>,
}

impl<'a> ProgramSvmTestClient<'a> {
    pub fn process_transaction_batch(
        &self,
        sanitized_txs: &[impl SVMTransaction],
        compute_budget: ComputeBudget,
        callbacks: &ProgramSvmTestAccountLoader,
        lamports_per_signature: u64,
    ) -> LoadAndExecuteSanitizedTransactionsOutput {
        let config = TransactionProcessingConfig {
            compute_budget: Some(compute_budget),
            ..Default::default()
        };
        self.processor.load_and_execute_sanitized_transactions(
            callbacks,
            sanitized_txs,
            get_transaction_check_results(sanitized_txs.len(), lamports_per_signature),
            &self.processing_environment,
            &config,
        )
    }
}

impl<'a> Default for ProgramSvmTest<'a> {
    fn default() -> Self {
        let compute_budget = ComputeBudget::default();
        let feature_set = FeatureSet::all_enabled();
        let fee_structure = FeeStructure::default();
        let rent_collector = RentCollector::default();
        let accounts = ProgramSvmTestAccountLoader::default();
        let fork_graph = Arc::new(RwLock::new(ProgramSvmTestForkGraph {}));
        Self {
            accounts,
            fork_graph,
            compute_budget,
            feature_set,
            fee_structure,
            rent_collector,
            client: Arc::new(None),
        }
    }
}
impl<'a> ProgramSvmTest<'a> {
    pub fn new() -> ProgramSvmTest<'a> {
        ProgramSvmTest::default()
    }

    pub fn add_account(&mut self, address: Pubkey, account: AccountSharedData) {
        self.accounts.add_account(address, account);
    }

    pub fn add_program() {
        todo!();
    }

    pub async fn start_with_context(&'a mut self) {
        let processor = create_transaction_batch_processor(
            &self.accounts,
            &self.feature_set,
            &self.compute_budget,
            Arc::clone(&self.fork_graph),
        );
        let processing_environment = TransactionProcessingEnvironment {
            blockhash: Hash::default(),
            epoch_total_stake: None,
            epoch_vote_accounts: None,
            feature_set: Arc::new(self.feature_set.clone()),
            fee_structure: Some(&self.fee_structure),
            lamports_per_signature: self.fee_structure.lamports_per_signature,
            rent_collector: Some(&self.rent_collector),
        };
        self.client = Arc::new(Some(ProgramSvmTestClient {
            processing_environment,
            processor,
        }));
    }
    pub fn process_transaction_batch(
        &self,
        sanitized_txs: &[impl SVMTransaction],
    ) -> LoadAndExecuteSanitizedTransactionsOutput {
        let config = TransactionProcessingConfig {
            compute_budget: Some(self.compute_budget),
            ..Default::default()
        };
        let x = self.client.as_ref();
        if let Some(ref processor) = self.client.as_ref() {
            processor.process_transaction_batch(
                sanitized_txs,
                self.compute_budget,
                &self.accounts,
                self.fee_structure.lamports_per_signature,
            )
        } else {
            panic!("The test framework is not initialized. Did you run `start_with_context`?")
        }
    }
}

/// This function is also a mock. In the Agave validator, the bank pre-checks
/// transactions before providing them to the SVM API. We mock this step in
/// PayTube, since we don't need to perform such pre-checks.
pub(crate) fn get_transaction_check_results(
    len: usize,
    lamports_per_signature: u64,
) -> Vec<transaction::Result<CheckedTransactionDetails>> {
    vec![
        transaction::Result::Ok(CheckedTransactionDetails {
            nonce: None,
            lamports_per_signature,
        });
        len
    ]
}

pub(crate) fn create_transaction_batch_processor<CB: TransactionProcessingCallback>(
    callbacks: &CB,
    feature_set: &FeatureSet,
    compute_budget: &ComputeBudget,
    fork_graph: Arc<RwLock<ProgramSvmTestForkGraph>>,
) -> TransactionBatchProcessor<ProgramSvmTestForkGraph> {
    let processor = TransactionBatchProcessor::<ProgramSvmTestForkGraph>::new_uninitialized(
        /* slot */ 1, /* epoch */ 1,
    );

    {
        let mut cache = processor.program_cache.write().unwrap();

        // Initialize the mocked fork graph.
        cache.fork_graph = Some(Arc::downgrade(&fork_graph));

        // Initialize a proper cache environment.
        // (Use Loader v4 program to initialize runtime v2 if desired)
        cache.environments.program_runtime_v1 = Arc::new(
            create_program_runtime_environment_v1(feature_set, compute_budget, false, false)
                .unwrap(),
        );
    }

    // Add the system program builtin.
    processor.add_builtin(
        callbacks,
        solana_system_program::id(),
        "system_program",
        ProgramCacheEntry::new_builtin(
            0,
            b"system_program".len(),
            system_processor::Entrypoint::vm,
        ),
    );

    // Add the BPF Loader v2 builtin, for the SPL Token program.
    processor.add_builtin(
        callbacks,
        solana_sdk::bpf_loader::id(),
        "solana_bpf_loader_program",
        ProgramCacheEntry::new_builtin(
            0,
            b"solana_bpf_loader_program".len(),
            solana_bpf_loader_program::Entrypoint::vm,
        ),
    );

    processor
}
#[derive(Default, Clone)]
pub struct ProgramSvmTestAccountLoader {
    pub account_shared_data: Arc<RwLock<HashMap<Pubkey, AccountSharedData>>>,
    #[allow(clippy::type_complexity)]
    pub inspected_accounts:
        Arc<RwLock<HashMap<Pubkey, Vec<(Option<AccountSharedData>, /* is_writable */ bool)>>>>,
}

impl ProgramSvmTestAccountLoader {
    pub fn new(accounts: Arc<RwLock<HashMap<Pubkey, AccountSharedData>>>) -> Self {
        Self {
            account_shared_data: accounts,
            inspected_accounts: Default::default(),
        }
    }

    pub fn add_account(&mut self, address: Pubkey, account: AccountSharedData) {
        let mut accounts = self.account_shared_data.write().unwrap();
        accounts.insert(address, account);
    }
}

impl TransactionProcessingCallback for ProgramSvmTestAccountLoader {
    fn account_matches_owners(&self, account: &Pubkey, owners: &[Pubkey]) -> Option<usize> {
        if let Some(data) = self.account_shared_data.read().unwrap().get(account) {
            if data.lamports() == 0 {
                None
            } else {
                owners.iter().position(|entry| data.owner() == entry)
            }
        } else {
            None
        }
    }

    fn get_account_shared_data(&self, pubkey: &Pubkey) -> Option<AccountSharedData> {
        self.account_shared_data
            .read()
            .unwrap()
            .get(pubkey)
            .cloned()
    }

    fn add_builtin_account(&self, name: &str, program_id: &Pubkey) {
        let mut account_data = AccountSharedData::default();
        account_data.set_data_from_slice(name.as_bytes());
        self.account_shared_data
            .write()
            .unwrap()
            .insert(*program_id, account_data);
    }

    fn inspect_account(&self, address: &Pubkey, account_state: AccountState, is_writable: bool) {
        let account = match account_state {
            AccountState::Dead => None,
            AccountState::Alive(account) => Some(account.clone()),
        };
        self.inspected_accounts
            .write()
            .unwrap()
            .entry(*address)
            .or_default()
            .push((account, is_writable));
    }
}

pub(crate) struct ProgramSvmTestForkGraph {}

impl ForkGraph for ProgramSvmTestForkGraph {
    fn relationship(&self, _a: Slot, _b: Slot) -> BlockRelation {
        BlockRelation::Unknown
    }
}
// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = 4;
//         assert_eq!(result, 4);
//     }
// }
