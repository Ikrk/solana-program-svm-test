use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use serde::{Deserialize, Serialize};
use solana_bpf_loader_program::syscalls::create_program_runtime_environment_v1;
use solana_compute_budget::compute_budget::ComputeBudget;
use solana_program_runtime::{
    invoke_context::BuiltinFunctionWithContext,
    loaded_programs::{BlockRelation, ForkGraph, LoadProgramMetrics, ProgramCacheEntry},
};
use solana_sdk::{
    account::{AccountSharedData, ReadableAccount, WritableAccount},
    bpf_loader, bpf_loader_deprecated,
    bpf_loader_upgradeable::{self, UpgradeableLoaderState},
    clock::Slot,
    compute_budget,
    feature_set::FeatureSet,
    fee::FeeStructure,
    hash::Hash,
    native_loader,
    pubkey::Pubkey,
    rent::Rent,
    rent_collector::RentCollector,
    signature::Keypair,
    signer::Signer,
    slot_hashes::SlotHashes,
    transaction::{self, SanitizedTransaction, Transaction, TransactionError},
};
use solana_svm::{
    account_loader::CheckedTransactionDetails,
    rollback_accounts::RollbackAccounts,
    transaction_processing_callback::{AccountState, TransactionProcessingCallback},
    transaction_processing_result::ProcessedTransaction,
    transaction_processor::{
        TransactionBatchProcessor, TransactionProcessingConfig, TransactionProcessingEnvironment,
    },
};
use solana_system_program::system_processor;

use solana_svm_transaction::svm_message::SVMMessage;
use sysvars::Sysvars;

pub mod sysvars;

pub const TOKEN_2022_PROGRAM_ID: Pubkey =
    solana_sdk::pubkey!("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb");
pub const TOKEN_PROGRAM_ID: Pubkey =
    solana_sdk::pubkey!("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA");
pub const ASSOCIATED_TOKEN_PROGRAM_ID: Pubkey =
    solana_sdk::pubkey!("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL");
pub const MEMO_PROGRAM_ID: Pubkey =
    solana_sdk::pubkey!("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr");
pub const MEMO_V1_PROGRAM_ID: Pubkey =
    solana_sdk::pubkey!("Memo1UhkJRfHyvLMcVucJwxXeuD728EqVDDwQDxFMNo");

pub type TransactionProcessingResult = Result<(), TransactionError>;
pub struct ProgramSvmTestProgramEntry {
    program_id: Pubkey,
    name: String,
    loader: Loader,
}

pub enum Loader {
    V1,
    V2,
    UPGRADABLE,
}

impl Loader {
    pub fn to_pubkey(&self) -> Pubkey {
        match self {
            Loader::V1 => bpf_loader_deprecated::id(),
            Loader::V2 => bpf_loader::id(),
            Loader::UPGRADABLE => bpf_loader_upgradeable::id(),
        }
    }
}
pub struct ProgramSvmTest {
    accounts: ProgramSvmTestAccountLoader,
    fork_graph: Arc<RwLock<ProgramSvmTestForkGraph>>,
    compute_budget: ComputeBudget,
    feature_set: FeatureSet,
    fee_structure: FeeStructure,
    rent_collector: RentCollector,
    programs: Vec<ProgramSvmTestProgramEntry>,
}

impl Default for ProgramSvmTest {
    /// Creates new barebone testing enviroment
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
            programs: Vec::default(),
        }
    }
}
impl ProgramSvmTest {
    /// Creates new testing enviroment and adds commonly used programs (currently only Token, Token2022 and Associated Token programs)
    pub fn new() -> ProgramSvmTest {
        solana_logger::setup_with_default(
            "solana_rbpf::vm=debug,\
                     solana_runtime::message_processor=debug,\
                     solana_runtime::system_instruction_processor=trace",
        );

        let mut program_test = ProgramSvmTest::default();
        program_test.add_upgradable_program("token", TOKEN_PROGRAM_ID, 0, None, None);
        program_test.add_upgradable_program("token-2022", TOKEN_2022_PROGRAM_ID, 0, None, None);
        program_test.add_upgradable_program(
            "associated-token",
            ASSOCIATED_TOKEN_PROGRAM_ID,
            0,
            None,
            None,
        );
        program_test.add_program("memo", MEMO_PROGRAM_ID, None, Loader::V2);
        program_test.add_program("memo-v1", MEMO_V1_PROGRAM_ID, None, Loader::V1);

        program_test
    }

    /// Adds new account to the testing environment
    pub fn add_account(&mut self, address: Pubkey, account: AccountSharedData) {
        self.accounts.add_account(address, account);
    }

    fn add_sysvars(&mut self, processor: &TransactionBatchProcessor<ProgramSvmTestForkGraph>) {
        let sysvars = Sysvars::default();
        let (key, account) = sysvars.keyed_account_for_rent_sysvar();
        self.add_account(key, account);

        let (key, account) = sysvars.keyed_account_for_clock_sysvar();
        self.add_account(key, account);

        let (key, account) = sysvars.keyed_account_for_epoch_rewards_sysvar();
        self.add_account(key, account);

        let (key, account) = sysvars.keyed_account_for_epoch_schedule_sysvar();
        self.add_account(key, account);

        let (key, account) = sysvars.keyed_account_for_last_restart_slot_sysvar();
        self.add_account(key, account);

        let (key, account) = sysvars.keyed_account_for_slot_hashes_sysvar();
        self.add_account(key, account);

        let (key, account) = sysvars.keyed_account_for_stake_history_sysvar();
        self.add_account(key, account);

        processor.fill_missing_sysvar_cache_entries(&self.accounts);
    }

    fn add_programs(&self, processor: &TransactionBatchProcessor<ProgramSvmTestForkGraph>) {
        let mut cache = processor.program_cache.write().unwrap();
        for program in self.programs.iter() {
            let program_data = load_program(&program.name);
            let entry = Arc::new(
                ProgramCacheEntry::new(
                    &program.loader.to_pubkey(),
                    cache.environments.program_runtime_v1.clone(),
                    0,
                    0,
                    &program_data,
                    program_data.len(),
                    &mut LoadProgramMetrics::default(),
                )
                .unwrap(),
            );
            cache.assign_program(program.program_id, entry);
        }
    }

    /// Adds new non-upgradable program to the testing environment
    ///
    /// Natively compiled programs are currently not supported.
    pub fn add_program(
        &mut self,
        program_name: &'static str,
        program_id: Pubkey,
        _builtin_function: Option<BuiltinFunctionWithContext>,
        loader: Loader,
    ) -> Pubkey {
        self.programs.push(ProgramSvmTestProgramEntry {
            program_id,
            name: program_name.to_string(),
            loader,
        });
        let rent = Rent::default();
        let program_account = program_id;
        let program_data = load_program(program_name);

        let mut account_data = AccountSharedData::new_data(
            rent.minimum_balance(program_data.len()),
            &program_data,
            &bpf_loader::id(),
        )
        .unwrap();
        account_data.set_executable(true);
        self.accounts
            .account_shared_data
            .write()
            .unwrap()
            .insert(program_account, account_data);

        program_account
    }

    /// Adds new upgradable program to the testing environment
    ///
    /// Natively compiled programs are currently not supported.
    pub fn add_upgradable_program(
        &mut self,
        program_name: &'static str,
        program_id: Pubkey,
        deployment_slot: Slot,
        _builtin_function: Option<BuiltinFunctionWithContext>, // TODO add support for natively compiled programs
        upgrade_authority_address: Option<Pubkey>,
    ) -> Pubkey {
        self.programs.push(ProgramSvmTestProgramEntry {
            program_id,
            name: program_name.to_string(),
            loader: Loader::UPGRADABLE,
        });

        let rent = Rent::default();
        let program_account = program_id;
        let program_data_account =
            bpf_loader_upgradeable::get_program_data_address(&program_account);

        let state = UpgradeableLoaderState::Program {
            programdata_address: program_data_account,
        };

        let mut account_data = AccountSharedData::new_data(
            rent.minimum_balance(UpgradeableLoaderState::size_of_program()),
            &state,
            &bpf_loader_upgradeable::id(),
        )
        .unwrap();
        account_data.set_executable(true);
        self.accounts
            .account_shared_data
            .write()
            .unwrap()
            .insert(program_account, account_data);

        let program_data = load_program(program_name);
        let state = ProgramSvmTestUpgradableLoaderState::new(
            deployment_slot,
            upgrade_authority_address,
            program_data,
        );
        let account_data = AccountSharedData::new_data(
            rent.minimum_balance(state.len()),
            &state,
            &bpf_loader_upgradeable::id(),
        )
        .unwrap();
        self.accounts
            .account_shared_data
            .write()
            .unwrap()
            .insert(program_data_account, account_data);

        program_account
    }

    // TODO add additional methods to set/override the internal state (compute_budget, feature_set, fee_structure, rent_collector)

    /// Initialize the processing environment
    pub fn start(&mut self) -> (ProgramSvmTestClient, Keypair) {
        let payer = Keypair::new();

        self.add_account(
            payer.pubkey(),
            AccountSharedData::new(5_000_000_000_000_000, 0, &solana_system_program::id()),
        );

        let processor = create_transaction_batch_processor(
            &self.accounts,
            &self.feature_set,
            &self.compute_budget,
            Arc::clone(&self.fork_graph),
        );
        self.add_sysvars(&processor);
        let processing_environment = TransactionProcessingEnvironment {
            blockhash: Hash::default(),
            epoch_total_stake: None,
            epoch_vote_accounts: None,
            feature_set: Arc::new(self.feature_set.clone()),
            fee_structure: Some(&self.fee_structure),
            lamports_per_signature: self.fee_structure.lamports_per_signature,
            rent_collector: Some(&self.rent_collector),
        };
        self.add_programs(&processor);
        let client = ProgramSvmTestClient {
            processing_environment,
            processor,
            accounts: &mut self.accounts,
        };

        (client, payer)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
struct ProgramSvmTestUpgradableLoaderState {
    loader_state: UpgradeableLoaderState,
    program_data: Vec<u8>,
}

impl ProgramSvmTestUpgradableLoaderState {
    fn new(
        deployment_slot: u64,
        upgrade_authority_address: Option<Pubkey>,
        program_data: Vec<u8>,
    ) -> Self {
        let loader_state = UpgradeableLoaderState::ProgramData {
            slot: deployment_slot,
            upgrade_authority_address,
        };
        Self {
            loader_state,
            program_data,
        }
    }
    fn len(&self) -> usize {
        UpgradeableLoaderState::size_of_programdata(self.program_data.len())
    }
}
pub struct ProgramSvmTestClient<'a> {
    processing_environment: TransactionProcessingEnvironment<'a>,
    processor: TransactionBatchProcessor<ProgramSvmTestForkGraph>,
    accounts: &'a mut ProgramSvmTestAccountLoader,
}

impl<'a> ProgramSvmTestClient<'a> {
    /// Processes a transaction
    pub fn process_transaction(&self, transaction: Transaction) -> TransactionProcessingResult {
        // TODO add custom ProgramSvmTestClient errors
        // TODO for now only default tx config, it would be good to have optional config parameter
        let config = TransactionProcessingConfig {
            // compute_budget: Some(compute_budget),
            compute_budget: Some(ComputeBudget::default()),
            ..Default::default()
        };

        let sanitized_txs = &[SanitizedTransaction::from_transaction_for_tests(
            transaction,
        )];
        let batch_output = self.processor.load_and_execute_sanitized_transactions(
            self.accounts,
            sanitized_txs,
            get_transaction_check_results(
                sanitized_txs.len(),
                self.processing_environment.lamports_per_signature,
            ),
            &self.processing_environment,
            &config,
        );

        let mut final_accounts_actual = self.accounts.account_shared_data.write().unwrap();

        assert_eq!(batch_output.processing_results.len(), 1);

        let processed_transaction = batch_output.processing_results.first().unwrap();
        dbg!(processed_transaction);
        match processed_transaction {
            Ok(ProcessedTransaction::Executed(executed_transaction)) => {
                for (pubkey, account_data) in
                    executed_transaction.loaded_transaction.accounts.clone()
                {
                    final_accounts_actual.insert(pubkey, account_data);
                }
                executed_transaction.execution_details.status.clone()
            }
            Ok(ProcessedTransaction::FeesOnly(fees_only_transaction)) => {
                let fee_payer = sanitized_txs[0].fee_payer();

                match fees_only_transaction.rollback_accounts.clone() {
                    RollbackAccounts::FeePayerOnly { fee_payer_account } => {
                        final_accounts_actual.insert(*fee_payer, fee_payer_account);
                    }
                    RollbackAccounts::SameNonceAndFeePayer { nonce } => {
                        final_accounts_actual.insert(*nonce.address(), nonce.account().clone());
                    }
                    RollbackAccounts::SeparateNonceAndFeePayer {
                        nonce,
                        fee_payer_account,
                    } => {
                        final_accounts_actual.insert(*fee_payer, fee_payer_account);
                        final_accounts_actual.insert(*nonce.address(), nonce.account().clone());
                    }
                }
                Err(fees_only_transaction.load_error.clone())
            }
            Err(e) => Err(e.clone()),
        }
    }

    pub fn get_last_blockhash(&self) -> Hash {
        self.processing_environment.blockhash
    }

    pub fn get_account(&self, address: &Pubkey) -> Option<AccountSharedData> {
        self.accounts.get_account_shared_data(address)
    }

    /// Warp the test environment to a slot by updating sysvars.
    pub fn warp_to_slot(&mut self, slot: u64) {
        let old_sysvars = self.processor.sysvar_cache();
        let mut new_sysvars = Sysvars {
            clock: (*old_sysvars.get_clock().unwrap_or_default()).clone(),
            epoch_rewards: (*old_sysvars.get_epoch_rewards().unwrap_or_default()).clone(),
            epoch_schedule: (*old_sysvars.get_epoch_schedule().unwrap_or_default()).clone(),
            last_restart_slot: (*old_sysvars.get_last_restart_slot().unwrap_or_default()).clone(),
            rent: (*old_sysvars.get_rent().unwrap_or_default()).clone(),
            slot_hashes: SlotHashes::new(
                &(*old_sysvars.get_slot_hashes().unwrap_or_default()).clone(),
            ),
            stake_history: (*old_sysvars.get_stake_history().unwrap_or_default()).clone(),
        };
        new_sysvars.warp_to_slot(slot);

        // reset_sysvar_cache needs the write-lock to the old sysvars, so we need to release the previous read lock
        drop(old_sysvars);

        self.processor.reset_sysvar_cache();

        let (key, account) = new_sysvars.keyed_account_for_rent_sysvar();
        self.accounts.add_account(key, account);

        let (key, account) = new_sysvars.keyed_account_for_clock_sysvar();
        self.accounts.add_account(key, account);

        let (key, account) = new_sysvars.keyed_account_for_epoch_rewards_sysvar();
        self.accounts.add_account(key, account);

        let (key, account) = new_sysvars.keyed_account_for_epoch_schedule_sysvar();
        self.accounts.add_account(key, account);

        let (key, account) = new_sysvars.keyed_account_for_last_restart_slot_sysvar();
        self.accounts.add_account(key, account);

        let (key, account) = new_sysvars.keyed_account_for_slot_hashes_sysvar();
        self.accounts.add_account(key, account);

        let (key, account) = new_sysvars.keyed_account_for_stake_history_sysvar();
        self.accounts.add_account(key, account);

        self.processor
            .fill_missing_sysvar_cache_entries(self.accounts);
    }
}

/// This function is also a mock. In the Agave validator, the bank pre-checks
/// transactions before providing them to the SVM API. We mock this step
/// since we don't need to perform such pre-checks.
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

fn default_shared_object_dirs() -> Vec<PathBuf> {
    let mut search_path = vec![PathBuf::from("programs")];

    if let Ok(bpf_out_dir) = std::env::var("BPF_OUT_DIR") {
        search_path.push(PathBuf::from(bpf_out_dir));
    }

    if let Ok(bpf_out_dir) = std::env::var("SBF_OUT_DIR") {
        search_path.push(PathBuf::from(bpf_out_dir));
    }

    if let Ok(dir) = std::env::current_dir() {
        search_path.push(dir);
    }

    search_path
}

fn find_file(filename: &str) -> Option<PathBuf> {
    for dir in default_shared_object_dirs() {
        let candidate = dir.join(filename);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

pub fn read_file<P: AsRef<Path>>(path: P) -> Vec<u8> {
    let path = path.as_ref();
    let mut file = File::open(path).expect("Could not open the file");

    let mut file_data = Vec::new();
    file.read_to_end(&mut file_data)
        .expect("could not read the file");
    file_data
}

fn load_program(name: &str) -> Vec<u8> {
    std::env::set_var("SBF_OUT_DIR", "./target/deploy");
    let binary_name = format!("{}.so", name.replace('-', "_"));

    let file = find_file(&binary_name).expect("File not found");
    read_file(file)
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

    // Add the BPF Loaders
    processor.add_builtin(
        callbacks,
        solana_sdk::bpf_loader_deprecated::id(),
        "solana_bpf_loader_program",
        ProgramCacheEntry::new_builtin(
            0,
            b"solana_bpf_loader_program".len(),
            solana_bpf_loader_program::Entrypoint::vm,
        ),
    );
    processor.add_builtin(
        callbacks,
        solana_sdk::loader_v4::id(),
        "solana_bpf_loader_program",
        ProgramCacheEntry::new_builtin(
            0,
            b"solana_bpf_loader_program".len(),
            solana_bpf_loader_program::Entrypoint::vm,
        ),
    );
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
    processor.add_builtin(
        callbacks,
        solana_sdk::bpf_loader_upgradeable::id(),
        "solana_bpf_loader_program",
        ProgramCacheEntry::new_builtin(
            0,
            b"solana_bpf_loader_program".len(),
            solana_bpf_loader_program::Entrypoint::vm,
        ),
    );
    processor.add_builtin(
        callbacks,
        compute_budget::id(),
        "compute_budget_program",
        ProgramCacheEntry::new_builtin(
            0,
            "compute_budget_program".len(),
            solana_compute_budget_program::Entrypoint::vm,
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

    fn add_account(&mut self, address: Pubkey, account: AccountSharedData) {
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
        account_data.set_executable(true);
        account_data.set_owner(native_loader::id());
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
#[cfg(test)]
mod tests {
    use solana_sdk::{
        clock::Clock,
        epoch_schedule::EpochSchedule,
        instruction::{AccountMeta, Instruction},
        program_pack::Pack,
        signature::Keypair,
        signer::Signer,
        system_instruction, system_program,
        sysvar::SysvarId,
    };

    use super::*;

    #[test]
    fn test_create_account() {
        let user = Keypair::new();

        let mut test_program = ProgramSvmTest::new();

        let (client, payer) = test_program.start();
        let instructions = vec![system_instruction::create_account(
            &payer.pubkey(),
            &user.pubkey(),
            500_000_000,
            0,
            &solana_system_program::id(),
        )];

        let transaction = Transaction::new_signed_with_payer(
            &instructions,
            Some(&payer.pubkey()),
            &[&payer, &user],
            client.get_last_blockhash(),
        );
        client.process_transaction(transaction).unwrap();
        let acc = client.get_account(&user.pubkey());
        assert!(acc.is_some());
        assert_eq!(acc.unwrap().lamports(), 500_000_000);
    }

    #[test]
    fn test_hello_solana_program() {
        let program_id = Pubkey::new_unique();

        let mut test_program = ProgramSvmTest::new();
        test_program.add_upgradable_program("hello-solana-program", program_id, 0, None, None);
        let (client, payer) = test_program.start();

        let instructions = vec![Instruction::new_with_bytes(program_id, &[], vec![])];
        let transaction = Transaction::new_signed_with_payer(
            &instructions,
            Some(&payer.pubkey()),
            &[&payer],
            client.get_last_blockhash(),
        );
        client.process_transaction(transaction).unwrap();
    }

    #[test]
    fn test_simple_transfer_program() {
        let recipient = Keypair::new();
        let program_id = Pubkey::new_unique();

        let mut test_program = ProgramSvmTest::new();
        test_program.add_upgradable_program("simple-transfer-program", program_id, 0, None, None);
        let (client, payer) = test_program.start();
        let payer_amount_before = client.get_account(&payer.pubkey()).unwrap().lamports();
        let amount = 600_000_000u64;

        let instructions = vec![Instruction::new_with_bytes(
            program_id,
            amount.to_be_bytes().as_ref(),
            vec![
                AccountMeta::new(payer.pubkey(), true),
                AccountMeta::new(recipient.pubkey(), false),
                AccountMeta::new_readonly(system_program::id(), false),
            ],
        )];

        let transaction = Transaction::new_signed_with_payer(
            &instructions,
            Some(&payer.pubkey()),
            &[&payer],
            client.get_last_blockhash(),
        );
        client.process_transaction(transaction).unwrap();

        let acc = client.get_account(&recipient.pubkey()).unwrap();
        let payer_amount_after = client.get_account(&payer.pubkey()).unwrap().lamports();
        assert_eq!(acc.lamports(), amount);
        assert!(payer_amount_after < payer_amount_before - amount);
    }

    #[test]
    fn test_token_legacy() {
        let mut test_program = ProgramSvmTest::new();
        let user = Keypair::new();

        test_program.add_account(
            user.pubkey(),
            AccountSharedData::new(5_000_000_000_000, 0, &solana_system_program::id()),
        );
        let (client, payer) = test_program.start();
        let mint = Keypair::new();
        let minimum_balance = Rent::default().minimum_balance(spl_token::state::Mint::LEN);
        let transaction = Transaction::new_signed_with_payer(
            &[
                system_instruction::create_account(
                    &payer.pubkey(),
                    &mint.pubkey(),
                    minimum_balance,
                    spl_token::state::Mint::LEN as u64,
                    &spl_token::id(),
                ),
                spl_token::instruction::initialize_mint2(
                    &spl_token::id(),
                    &mint.pubkey(),
                    &payer.pubkey(),
                    None,
                    6,
                )
                .unwrap(),
            ],
            Some(&payer.pubkey()),
            &[&payer, &mint],
            client.get_last_blockhash(),
        );
        client.process_transaction(transaction).unwrap();
        let mint_data = client.get_account(&mint.pubkey()).unwrap();
        let mint_deserialized = spl_token::state::Mint::unpack(mint_data.data()).unwrap();
        assert_eq!(mint_deserialized.decimals, 6);

        let token_account = Keypair::new();
        let minimum_balance = Rent::default().minimum_balance(spl_token::state::Account::LEN);
        let transaction = Transaction::new_signed_with_payer(
            &[
                system_instruction::create_account(
                    &payer.pubkey(),
                    &token_account.pubkey(),
                    minimum_balance,
                    spl_token::state::Account::LEN as u64,
                    &spl_token::id(),
                ),
                spl_token::instruction::initialize_account3(
                    &spl_token::id(),
                    &token_account.pubkey(),
                    &mint.pubkey(),
                    &payer.pubkey(),
                )
                .unwrap(),
            ],
            Some(&payer.pubkey()),
            &[&payer, &token_account],
            client.get_last_blockhash(),
        );
        client.process_transaction(transaction).unwrap();
        let ix = spl_token::instruction::mint_to(
            &spl_token::id(),
            &mint.pubkey(),
            &token_account.pubkey(),
            &payer.pubkey(),
            &[&payer.pubkey()],
            100000,
        )
        .unwrap();
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&payer.pubkey()),
            &[&payer],
            client.get_last_blockhash(),
        );
        client.process_transaction(tx).unwrap();
        let token_data = client.get_account(&token_account.pubkey()).unwrap();
        let token_deserialized = spl_token::state::Account::unpack(token_data.data()).unwrap();
        assert_eq!(token_deserialized.amount, 100000);
    }

    #[test]
    fn test_token_2022() {
        let mut test_program = ProgramSvmTest::new();
        let user = Keypair::new();

        test_program.add_account(
            user.pubkey(),
            AccountSharedData::new(5_000_000_000_000, 0, &solana_system_program::id()),
        );
        let (client, payer) = test_program.start();
        let mint = Keypair::new();
        let minimum_balance = Rent::default().minimum_balance(spl_token_2022::state::Mint::LEN);
        let transaction = Transaction::new_signed_with_payer(
            &[
                system_instruction::create_account(
                    &payer.pubkey(),
                    &mint.pubkey(),
                    minimum_balance,
                    spl_token_2022::state::Mint::LEN as u64,
                    &spl_token_2022::id(),
                ),
                spl_token_2022::instruction::initialize_mint2(
                    &spl_token_2022::id(),
                    &mint.pubkey(),
                    &payer.pubkey(),
                    None,
                    6,
                )
                .unwrap(),
            ],
            Some(&payer.pubkey()),
            &[&payer, &mint],
            client.get_last_blockhash(),
        );
        client.process_transaction(transaction).unwrap();
        let mint_data = client.get_account(&mint.pubkey()).unwrap();
        let mint_deserialized = spl_token_2022::state::Mint::unpack(mint_data.data()).unwrap();
        assert_eq!(mint_deserialized.decimals, 6);

        let token_account = Keypair::new();
        let minimum_balance = Rent::default().minimum_balance(spl_token_2022::state::Account::LEN);
        let transaction = Transaction::new_signed_with_payer(
            &[
                system_instruction::create_account(
                    &payer.pubkey(),
                    &token_account.pubkey(),
                    minimum_balance,
                    spl_token_2022::state::Account::LEN as u64,
                    &spl_token_2022::id(),
                ),
                spl_token_2022::instruction::initialize_account3(
                    &spl_token_2022::id(),
                    &token_account.pubkey(),
                    &mint.pubkey(),
                    &payer.pubkey(),
                )
                .unwrap(),
            ],
            Some(&payer.pubkey()),
            &[&payer, &token_account],
            client.get_last_blockhash(),
        );
        client.process_transaction(transaction).unwrap();
        let ix = spl_token_2022::instruction::mint_to(
            &spl_token_2022::id(),
            &mint.pubkey(),
            &token_account.pubkey(),
            &payer.pubkey(),
            &[&payer.pubkey()],
            100000,
        )
        .unwrap();
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&payer.pubkey()),
            &[&payer],
            client.get_last_blockhash(),
        );
        client.process_transaction(tx).unwrap();
        let token_data = client.get_account(&token_account.pubkey()).unwrap();
        let token_deserialized = spl_token_2022::state::Account::unpack(token_data.data()).unwrap();
        assert_eq!(token_deserialized.amount, 100000);
    }

    #[test]
    fn test_warp_to_slot() {
        let mut test_program = ProgramSvmTest::new();
        let (mut client, _) = test_program.start();

        let clock: Clock = client
            .get_account(&Clock::id())
            .unwrap()
            .deserialize_data()
            .unwrap();

        assert_eq!(clock.slot, 0);

        let new_slot = 1000;
        client.warp_to_slot(new_slot);

        let epoch_schedule: EpochSchedule = client
            .get_account(&EpochSchedule::id())
            .unwrap()
            .deserialize_data()
            .unwrap();

        let clock: Clock = client
            .get_account(&Clock::id())
            .unwrap()
            .deserialize_data()
            .unwrap();

        assert_eq!(clock.slot, new_slot);
        assert_eq!(clock.epoch, epoch_schedule.get_epoch(new_slot));
    }
}
