use anchor_lang::{prelude::CpiContext, Id};
use borsh::{BorshDeserialize, BorshSerialize};
use core::mem::size_of;
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
    system_program,
};
use spl_account_compression::{state::CONCURRENT_MERKLE_TREE_HEADER_SIZE_V1, ConcurrentMerkleTree};

/// Represents a program instruction. Could be an enum, but we only have
/// a single instruction.
#[derive(BorshSerialize, BorshDeserialize)]
pub struct InsertLeaf {
    pub leaf: [u8; 32],
}

const MAX_DEPTH: u32 = 5;
const MAX_BUFFER_SIZE: u32 = 8;

/// Processes an instruction.
///
/// Accounts expected:
/// 0. `[writable]`         The Merkle tree PDA.
/// 1. `[signer]`           The Merkle tree owner.
/// 2. `[]`                 The Merkle tree authority PDA.
/// 4. `[]`                 System program.
/// 5. `[]`                 SPL compression program.
/// 6. `[]`                 Noop program.
pub(crate) fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = InsertLeaf::try_from_slice(instruction_data)
        .map_err(|_| ProgramError::InvalidInstructionData)?;

    let account_info_iter = &mut accounts.iter();
    let merkle_tree_info = next_account_info(account_info_iter)?;
    let owner_info = next_account_info(account_info_iter)?;
    let authority_info = next_account_info(account_info_iter)?;
    let system_program_info = next_account_info(account_info_iter)?;
    let compression_program_info = next_account_info(account_info_iter)?;
    let noop_program_info = next_account_info(account_info_iter)?;

    if !owner_info.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    let (authority_pda, authority_bump) = Pubkey::find_program_address(
        &[b"merkle_tree_authority", &owner_info.key.to_bytes()],
        &crate::ID,
    );
    if authority_info.key != &authority_pda {
        return Err(ProgramError::InvalidArgument);
    }
    let authority_seeds = &[
        &b"merkle_tree_authority"[..],
        &owner_info.key.to_bytes(),
        &[authority_bump],
    ];

    if system_program_info.key != &system_program::ID {
        return Err(ProgramError::InvalidArgument);
    }
    if compression_program_info.key.to_bytes() != spl_account_compression::ID.to_bytes() {
        return Err(ProgramError::InvalidArgument);
    }
    if noop_program_info.key.to_bytes() != spl_account_compression::Noop::id().to_bytes() {
        return Err(ProgramError::InvalidArgument);
    }

    let signer_seeds: &[&[&[u8]]] = &[authority_seeds];

    if merkle_tree_info.try_borrow_data()?[0] == 0 {
        let cpi_ctx = CpiContext::new_with_signer(
            compression_program_info.clone(),
            spl_account_compression::cpi::accounts::Initialize {
                authority: authority_info.clone(),
                merkle_tree: merkle_tree_info.clone(),
                noop: noop_program_info.clone(),
            },
            signer_seeds,
        );
        spl_account_compression::cpi::init_empty_merkle_tree(cpi_ctx, MAX_DEPTH, MAX_BUFFER_SIZE)?;
    }

    let cpi_ctx = CpiContext::new_with_signer(
        compression_program_info.clone(),
        spl_account_compression::cpi::accounts::Modify {
            authority: authority_info.clone(),
            merkle_tree: merkle_tree_info.clone(),
            noop: noop_program_info.clone(),
        },
        signer_seeds,
    );
    spl_account_compression::cpi::append(cpi_ctx, instruction.leaf)?;

    msg!(
        "New Merkle root: {:?}",
        get_root(&merkle_tree_info.try_borrow_data()?)
    );

    Ok(())
}

fn get_root(merkle_tree_bytes: &[u8]) -> [u8; 32] {
    let (_header_bytes, rest) = merkle_tree_bytes.split_at(CONCURRENT_MERKLE_TREE_HEADER_SIZE_V1);

    let merkle_tree_size =
        size_of::<ConcurrentMerkleTree<{ MAX_DEPTH as usize }, { MAX_BUFFER_SIZE as usize }>>();
    let tree_bytes = &rest[..merkle_tree_size];

    let tree = bytemuck::try_from_bytes::<
        ConcurrentMerkleTree<{ MAX_DEPTH as usize }, { MAX_BUFFER_SIZE as usize }>,
    >(tree_bytes)
    .unwrap();

    tree.get_root()
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::keccak;
    use solana_program_test::*;
    use solana_sdk::{
        instruction::{AccountMeta, Instruction},
        signature::{Keypair, Signer},
        system_program,
        transaction::Transaction,
    };

    #[tokio::test]
    async fn test_merkle_insert() {
        let mut test = ProgramTest::new(
            "basic_merkle_tree",
            crate::ID,
            processor!(process_instruction),
        );

        test.add_program("spl_account_compression", spl_account_compression::ID, None);
        test.add_program("spl_noop", spl_account_compression::Noop::id(), None);

        let owner = Keypair::new();

        let (mut banks_client, payer, _recent_blockhash) = test.start().await;

        let merkle_tree_pda = init_tree(&mut banks_client, &payer).await;

        let mut expected_leaves = [[0; 32]; 1 << MAX_DEPTH];

        let leaf1 = [1; 32];
        expected_leaves[0] = leaf1;
        let tx = add_leaf_tx(&mut banks_client, &payer, &owner, merkle_tree_pda, leaf1).await;
        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_root_and_leaves(&mut banks_client, merkle_tree_pda).await;
        assert_eq!(expected_leaves, leaves.as_slice());
        assert_eq!(root, compute_merkle_root(&expected_leaves));

        let leaf2 = [2; 32];
        expected_leaves[1] = leaf2;
        let tx = add_leaf_tx(&mut banks_client, &payer, &owner, merkle_tree_pda, leaf2).await;
        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_root_and_leaves(&mut banks_client, merkle_tree_pda).await;

        assert_eq!(expected_leaves, leaves.as_slice());
        assert_eq!(root, compute_merkle_root(&expected_leaves));

        let leaf3 = [3; 32];
        expected_leaves[2] = leaf3;
        let tx = add_leaf_tx(&mut banks_client, &payer, &owner, merkle_tree_pda, leaf3).await;

        let last_root = compute_merkle_root(&expected_leaves);

        let sim_result = banks_client.simulate_transaction(tx.clone()).await.unwrap();
        let logs = sim_result.simulation_details.unwrap().logs;
        let expected_log = format!("New Merkle root: {:?}", last_root);
        assert!(
            logs.iter().any(|log| log.contains(&expected_log)),
            "Merkle tree log not found among\n{:?}",
            logs
        );

        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_root_and_leaves(&mut banks_client, merkle_tree_pda).await;
        assert_eq!(expected_leaves, leaves.as_slice());
        assert_eq!(root, last_root);
    }

    async fn init_tree(banks_client: &mut BanksClient, payer: &Keypair) -> Pubkey {
        let to = Keypair::new();

        let space = CONCURRENT_MERKLE_TREE_HEADER_SIZE_V1
            + size_of::<ConcurrentMerkleTree<{ MAX_DEPTH as usize }, { MAX_BUFFER_SIZE as usize }>>(
            )
            + ((1 << (MAX_DEPTH + 1)) - 2) * 32;
        let lamports = banks_client
            .get_rent()
            .await
            .unwrap()
            .minimum_balance(space);
        let ix = solana_program::system_instruction::create_account(
            &payer.pubkey(),
            &to.pubkey(),
            lamports,
            space as _,
            &spl_account_compression::ID,
        );

        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&payer.pubkey()),
            &[&payer, &to],
            banks_client.get_latest_blockhash().await.unwrap(),
        );
        banks_client.process_transaction(tx).await.unwrap();

        to.pubkey()
    }

    async fn add_leaf_tx(
        banks_client: &mut BanksClient,
        payer: &Keypair,
        owner: &Keypair,
        merkle_tree_pda: Pubkey,
        leaf: [u8; 32],
    ) -> Transaction {
        let ix_data = borsh::to_vec(&InsertLeaf { leaf }).expect("Failed to serialize instruction");

        let (authority, _) = Pubkey::find_program_address(
            &[b"merkle_tree_authority", &owner.pubkey().to_bytes()],
            &crate::ID,
        );

        let insert_ix = Instruction {
            program_id: crate::ID,
            accounts: vec![
                AccountMeta::new(merkle_tree_pda, false),
                AccountMeta::new_readonly(owner.pubkey(), true),
                AccountMeta::new_readonly(authority, false),
                AccountMeta::new_readonly(system_program::ID, false),
                AccountMeta::new_readonly(spl_account_compression::ID, false),
                AccountMeta::new_readonly(spl_account_compression::Noop::id(), false),
            ],
            data: ix_data,
        };

        Transaction::new_signed_with_payer(
            &[insert_ix],
            Some(&payer.pubkey()),
            &[&payer, &owner],
            banks_client.get_latest_blockhash().await.unwrap(),
        )
    }

    async fn fetch_root_and_leaves(
        banks_client: &mut BanksClient,
        merkle_tree_pda: Pubkey,
    ) -> ([u8; 32], Vec<[u8; 32]>) {
        let account = banks_client
            .get_account(merkle_tree_pda)
            .await
            .unwrap()
            .expect("PDA account should now exist");
        (get_root(&account.data), get_leaves(&account.data))
    }

    macro_rules! merkle_tree_hash_iteration {
        ($src:expr, $dst:expr) => {{
            let mut src_i = 0;
            let mut dst_i = 0;
            while src_i < $src.len() {
                let left = $src[src_i];
                let right = $src.get(src_i + 1).copied().unwrap_or([0; 32]);

                $dst[dst_i] = keccak::hashv(&[&left, &right]).0;

                src_i += 2;
                dst_i += 1;
            }

            $dst.truncate(dst_i);
        }};
    }

    /// Compute the Merkle root from the given list of leaves. The leaves are not
    /// hashed. If there's an odd number of nodes in a level, duplicates the last
    /// node.
    fn compute_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
        assert!(leaves.len().is_power_of_two());

        let mut row = vec![[0; 32]; leaves.len() / 2];

        // Perform two types of iteration to avoid cloning the entire array of leaves.
        merkle_tree_hash_iteration!(leaves, row);
        while row.len() > 1 {
            merkle_tree_hash_iteration!(row, row);
        }

        row[0]
    }

    fn get_leaves(merkle_tree_bytes: &[u8]) -> Vec<[u8; 32]> {
        let merkle_tree_size =
            size_of::<ConcurrentMerkleTree<{ MAX_DEPTH as usize }, { MAX_BUFFER_SIZE as usize }>>();

        let canopy_bytes =
            &merkle_tree_bytes[CONCURRENT_MERKLE_TREE_HEADER_SIZE_V1 + merkle_tree_size..];
        (0..1 << MAX_DEPTH)
            .map(|i| {
                let start = canopy_bytes.len() - (1 << MAX_DEPTH) * 32 + i * 32;
                canopy_bytes[start..start + 32].try_into().unwrap()
            })
            .collect()
    }
}
