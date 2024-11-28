# Solana ProgramSvmTest crate

This is a WIP re-implementation of the original Solana ProgramTest but using the stand-alone SVM provided by the solana-svm crate.

------------------
## Basic usage
```rust
#[test]
fn test_hello_solana_program() {
    let payer = Keypair::new();
    let program_id = Pubkey::new_unique();

    // Create a new testing enviroment
    let mut test_program = ProgramSvmTest::new();
    // Add custom programs to the execution environment
    test_program.add_program("hello-solana-program", program_id, 0, None);
    // Add accounts
    test_program.add_account(
        payer.pubkey(),
        AccountSharedData::new(5_000_000_000_000, 0, &solana_system_program::id()),
    );
    // Initialize the processing environment
    let client = test_program.start();

    let instructions = vec![Instruction::new_with_bytes(program_id, &[], vec![])];
    let transaction = Transaction::new_signed_with_payer(
        &instructions,
        Some(&payer.pubkey()),
        &[&payer],
        client.get_last_blockhash(),
    );
    // Process your transaction
    client.process_transaction(transaction).unwrap();
}
```

-------------------
## ToDos
- [ ] Add support for sysvars.
- [ ] Add additional common programs (currently only Token, Token2022 and Associated Token programs are added).
- [ ] Add default funded payer account.
- [ ] Add support for natively compiled programs (currenlty only BPF/SBF binaries supported).
- [ ] Add methods to modify processing environment state (compute_budget, feature_set, fee_structure, rent_collector).
- [ ] Split the code to multiple files.
- [ ] Many more.
