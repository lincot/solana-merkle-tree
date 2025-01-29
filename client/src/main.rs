use basic_merkle_tree::{processor::InsertLeaf, state::MerkleTree};
use solana_client::{rpc_client::RpcClient, rpc_config::RpcTransactionConfig};
use solana_sdk::{
    commitment_config::CommitmentConfig,
    hash::Hash,
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    signature::{read_keypair_file, Signer},
    system_program,
    transaction::Transaction,
};
use solana_transaction_status_client_types::UiTransactionEncoding;
use std::env;

fn main() {
    let rpc_url = env::var("SOLANA_RPC").expect("SOLANA_RPC is not defined");
    let keypair_path =
        env::var("SOLANA_KEYPAIR").unwrap_or_else(|_| "~/.config/solana/id.json".into());
    let client = RpcClient::new_with_commitment(rpc_url, CommitmentConfig::processed());

    let leaf: Pubkey = env::args()
        .nth(1)
        .expect("expected the first argument to be the pubkey leaf")
        .parse()
        .expect("invalid pubkey argument");

    let payer = read_keypair_file(shellexpand::tilde(&keypair_path).to_string()).unwrap();

    let owner = &payer;

    println!(
        "Adding {leaf} to the Merkle tree owned by {}...",
        owner.pubkey()
    );

    let merkle_tree_pda = MerkleTree::find_pda(owner.pubkey()).0;

    let ix_data = borsh::to_vec(&InsertLeaf {
        leaf: leaf.to_bytes(),
    })
    .expect("Failed to serialize instruction data");

    let merkle_ix = Instruction {
        program_id: basic_merkle_tree::ID,
        accounts: vec![
            AccountMeta::new(merkle_tree_pda, false),
            AccountMeta::new_readonly(owner.pubkey(), true),
            AccountMeta::new(payer.pubkey(), true),
            AccountMeta::new_readonly(system_program::ID, false),
        ],
        data: ix_data,
    };

    let latest_blockhash: Hash = client.get_latest_blockhash().unwrap();

    let tx = Transaction::new_signed_with_payer(
        &[merkle_ix],
        Some(&payer.pubkey()),
        &[&payer, owner],
        latest_blockhash,
    );

    let signature = client
        .send_and_confirm_transaction_with_spinner(&tx)
        .unwrap();
    println!("Transaction Signature: {}", signature);

    let confirmed_tx = client
        .get_transaction_with_config(
            &signature,
            RpcTransactionConfig {
                encoding: Some(UiTransactionEncoding::Json),
                commitment: Some(CommitmentConfig::confirmed()),
                max_supported_transaction_version: Some(0),
            },
        )
        .unwrap();

    let logs = confirmed_tx.transaction.meta.unwrap().log_messages.unwrap();

    println!("--- Transaction Logs ---");
    for log in logs {
        println!("{}", log);
    }
}
