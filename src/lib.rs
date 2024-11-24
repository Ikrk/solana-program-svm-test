use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::Read,
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

pub const TOKEN_2022_ID: Pubkey =
    solana_sdk::pubkey!("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb");
pub const TOKEN_ID: Pubkey = solana_sdk::pubkey!("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA");
pub const ASSOCIATED_TOKEN_ID: Pubkey =
    solana_sdk::pubkey!("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL");

pub type TransactionProcessingResult = Result<(), TransactionError>;
pub struct ProgramSvmTestProgramEntry {
    program_id: Pubkey,
    name: String,
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
    pub fn new() -> ProgramSvmTest {
        // TODO also add a default payer account
        let mut program_test = ProgramSvmTest::default();
        program_test.add_program("token", TOKEN_ID, 0, None);
        program_test.add_program("token-2022", TOKEN_2022_ID, 0, None);
        program_test.add_program("associated-token", ASSOCIATED_TOKEN_ID, 0, None);
        program_test
    }

    pub fn add_account(&mut self, address: Pubkey, account: AccountSharedData) {
        self.accounts.add_account(address, account);
    }

    pub fn add_program2(
        &mut self,
        program_name: &'static str,
        program_id: Pubkey,
        // deployment_slot: Slot,
        // builtin_function: Option<BuiltinFunctionWithContext>,
    ) {
        self.programs.push(ProgramSvmTestProgramEntry {
            program_id,
            name: program_name.to_string(),
        });
    }

    fn add_programs(&self, processor: &TransactionBatchProcessor<ProgramSvmTestForkGraph>) {
        let mut cache = processor.program_cache.write().unwrap();
        for program in self.programs.iter() {
            let program_data = load_program(&program.name);
            let entry = Arc::new(
                ProgramCacheEntry::new(
                    &bpf_loader_upgradeable::id(),
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

    pub fn add_program(
        &mut self,
        program_name: &'static str,
        program_id: Pubkey,
        deployment_slot: Slot,
        builtin_function: Option<BuiltinFunctionWithContext>,
    ) -> Pubkey {
        self.programs.push(ProgramSvmTestProgramEntry {
            program_id,
            name: program_name.to_string(),
        });
        self.add_upgradable_program(
            program_name,
            program_id,
            deployment_slot,
            builtin_function,
            None,
        )
    }

    pub fn add_upgradable_program(
        &mut self,
        program_name: &'static str,
        program_id: Pubkey,
        deployment_slot: Slot,
        _builtin_function: Option<BuiltinFunctionWithContext>, // TODO add support for natively compiled programs
        upgrade_authority_address: Option<Pubkey>,
    ) -> Pubkey {
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

    // TODO this method in the original ProgramTest crate consumes self so that self cannot be reused once the bank was initialized
    // It would make sense to do it here too.
    pub fn start(&mut self) -> ProgramSvmTestClient {
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
        self.add_programs(&processor);
        ProgramSvmTestClient {
            processing_environment,
            processor,
            accounts: &self.accounts,
        }
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
    accounts: &'a ProgramSvmTestAccountLoader, // TODO this should be probably in Arc<RwLock<_>>
}

impl<'a> ProgramSvmTestClient<'a> {
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
            }
            Err(e) => return Err(e.clone()),
        }
        Ok(())
    }

    pub fn get_last_blockhash(&self) -> Hash {
        self.processing_environment.blockhash
    }

    pub fn get_account(&self, address: &Pubkey) -> Option<AccountSharedData> {
        self.accounts.get_account_shared_data(address)
    }

    fn _prepare_transactions(transactions: &[Transaction]) -> Vec<SanitizedTransaction> {
        transactions
            .iter()
            .cloned()
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect()
    }
}

/// This function is also a mock. In the Agave validator, the bank pre-checks
/// transactions before providing them to the SVM API. We mock this step in
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

fn load_program(name: &str) -> Vec<u8> {
    // Loading the program file, first look in the programs folder
    let mut dir = env::current_dir().unwrap();
    dir.push("programs");
    let binary_name = name.replace('-', "_");
    dir.push(binary_name.clone() + ".so");

    let mut file = if let Ok(file) = File::open(dir.clone()) {
        file
    } else {
        // if not found check other locations
        // TODO check target folder
        dir = env::current_dir().unwrap();
        dir.push("tests");
        dir.push("example-programs");
        dir.push(name);
        dir.push(binary_name + "_program.so");
        File::open(dir.clone()).expect("File not found")
    };

    let metadata = fs::metadata(dir).expect("Unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    file.read_exact(&mut buffer).expect("Buffer overflow");
    buffer
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
        instruction::{AccountMeta, Instruction},
        signature::Keypair,
        signer::Signer,
        system_instruction, system_program,
    };

    use super::*;

    #[test]
    fn test_create_account() {
        let payer = Keypair::new();
        let user = Keypair::new();

        let mut test_program = ProgramSvmTest::new();

        test_program.add_account(
            payer.pubkey(),
            AccountSharedData::new(5_000_000_000_000, 0, &solana_system_program::id()),
        );
        let client = test_program.start();
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
        let payer = Keypair::new();
        let program_id = Pubkey::new_unique();

        let mut test_program = ProgramSvmTest::new();
        test_program.add_program("hello-solana", program_id, 0, None);
        test_program.add_account(
            payer.pubkey(),
            AccountSharedData::new(5_000_000_000_000, 0, &solana_system_program::id()),
        );
        let client = test_program.start();

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
        let payer = Keypair::new();
        let recipient = Keypair::new();
        let program_id = Pubkey::new_unique();

        let mut test_program = ProgramSvmTest::new();
        test_program.add_program("simple-transfer", program_id, 0, None);
        test_program.add_account(
            payer.pubkey(),
            AccountSharedData::new(5_000_000_000_000, 0, &solana_system_program::id()),
        );
        let client = test_program.start();
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
}
