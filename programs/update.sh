solana program dump MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr ./memo.so -u mainnet-beta
solana program dump Memo1UhkJRfHyvLMcVucJwxXeuD728EqVDDwQDxFMNo ./memo-v1.so -u mainnet-beta
solana program dump TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA ./token.so -u mainnet-beta
solana program dump TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb ./token_2022.so -u mainnet-beta
solana program dump ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL ./associated_token.so -u mainnet-beta
echo "Last update on slot: $(solana slot -u mainnet-beta)" > last_update.log
