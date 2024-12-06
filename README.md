# Solana ProgramSvmTest crate

This crate is a WIP re-implementation of the original Solana ProgramTest crate but using the stand-alone SVM provided by the solana-svm crate.

## Key Goals

- Offer a similar interface to the original ProgramTest crate, making it easier for developers to transition to ProgramSvmTest.
- Enable testing of Solana programs in a lightweight, standalone SVM environment.


## Basic usage

To write a test,
developers can add multiple programs and accounts and process transactions in similar way as with ProgramTest crate.

```rust
#[test]
fn test_hello_solana_program() {
    let program_id = Pubkey::new_unique();
    let user = Keypair::new();

    // Create a new testing enviroment
    let mut test_program = ProgramSvmTest::new();
    // Add custom programs to the execution environment
    test_program.add_upgradable_program("hello-solana-program", program_id, 0, None, None);
    // Add accounts
    test_program.add_account(
        user.pubkey(),
        AccountSharedData::new(5_000_000_000_000, 0, &solana_system_program::id()),
    );
    // Initialize the processing environment
    let (client, payer) = test_program.start();

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


## ToDos
- [ ] ~~Add support for sysvars.~~ Add sysvars tests.
- [x] Add additional common programs (currently only Token, Token2022 and Associated Token programs are added).
- [x] Add default funded payer account.
- [ ] Add support for natively compiled programs (currenlty only BPF/SBF binaries supported).
- [ ] Add methods to modify processing environment state (compute_budget, feature_set, fee_structure, rent_collector).
- [ ] Align with Solana's ProgramTest crate interface.
- [ ] Split the code to multiple files.
- [ ] ...


## Credits
- This crate was created as part of the [Turbin3](https://turbin3.com) advanced SVM program.
- Inspired by Solana's [program-test](https://github.com/solana-labs/solana/tree/master/program-test) crate.
- Uses parts of code from [mollusk](https://github.com/buffalojoec/mollusk).
