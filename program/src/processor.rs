use crate::state::MerkleTree;
use borsh::{BorshDeserialize, BorshSerialize};
use solana_invoke::invoke_signed;
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint::ProgramResult,
    keccak, msg,
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
    system_instruction, system_program,
    sysvar::Sysvar,
};

/// Represents a program instruction. Could be an enum, but we only have
/// a single instruction.
#[derive(BorshSerialize, BorshDeserialize)]
pub struct InsertLeaf {
    pub leaf: [u8; 32],
}

/// Processes an instruction.
///
/// Accounts Expected:
/// 0. `[writable]` The Merkle tree PDA.
/// 1. `[signer]`   The Merkle tree "owner" (used in seeds).
/// 2. `[writable, signer]` The payer.
/// 3. `[]`         System program.
pub(crate) fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = InsertLeaf::try_from_slice(instruction_data)
        .map_err(|_| ProgramError::InvalidInstructionData)?;

    let account_info_iter = &mut accounts.iter();
    let merkle_tree_info = next_account_info(account_info_iter)?;
    let owner_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_program_info = next_account_info(account_info_iter)?;

    let rent = Rent::get()?;

    if !owner_info.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    let (merkle_tree_pda, merkle_tree_bump) = MerkleTree::find_pda(*owner_info.key);
    if merkle_tree_info.key != &merkle_tree_pda {
        return Err(ProgramError::InvalidArgument);
    }
    let merkle_tre_seeds = &[
        &b"merkle_tree"[..],
        &owner_info.key.to_bytes(),
        &[merkle_tree_bump],
    ];

    if system_program_info.key != &system_program::ID {
        return Err(ProgramError::InvalidArgument);
    }

    let (ix, mut merkle_tree) = if merkle_tree_info.data_is_empty() {
        let size = MerkleTree::space(1);
        let lamports_required = rent.minimum_balance(size);

        let ix = system_instruction::create_account(
            payer_info.key,
            merkle_tree_info.key,
            lamports_required,
            size as u64,
            program_id,
        );

        (ix, Default::default())
    } else {
        let initial_size = merkle_tree_info.data_len();
        let size = initial_size + 32;
        let lamports_required = rent.minimum_balance(size);

        let ix = system_instruction::transfer(
            payer_info.key,
            merkle_tree_info.key,
            lamports_required - merkle_tree_info.lamports(),
        );

        merkle_tree_info.realloc(size, false)?;

        (
            ix,
            MerkleTree::try_from_slice(&merkle_tree_info.data.borrow()[..initial_size])?,
        )
    };
    invoke_signed(
        &ix,
        &[payer_info.clone(), merkle_tree_info.clone()],
        &[merkle_tre_seeds],
    )?;

    merkle_tree.leaves.push(instruction.leaf);
    let new_root = compute_merkle_root(&merkle_tree.leaves);
    merkle_tree.root = new_root;

    msg!("New Merkle root: {:?}", new_root);

    merkle_tree.serialize(&mut &mut merkle_tree_info.data.borrow_mut()[..])?;

    Ok(())
}

macro_rules! merkle_tree_hash_iteration {
    ($src:expr, $dst:expr) => {{
        let mut src_i = 0;
        let mut dst_i = 0;
        while src_i < $src.len() {
            let left = $src[src_i];
            let right = $src.get(src_i + 1).copied().unwrap_or(left);

            let (left, right) = if left <= right {
                (left, right)
            } else {
                (right, left)
            };

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
    assert!(!leaves.is_empty());

    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut row = vec![[0; 32]; (leaves.len() + 1) / 2];

    // Perform two types of iteration to avoid cloning the entire array of leaves.
    merkle_tree_hash_iteration!(leaves, row);
    while row.len() > 1 {
        merkle_tree_hash_iteration!(row, row);
    }

    row[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use borsh::BorshDeserialize;
    use solana_program_test::*;
    use solana_sdk::{
        instruction::{AccountMeta, Instruction},
        signature::{Keypair, Signer},
        system_program,
        transaction::Transaction,
    };

    #[tokio::test]
    async fn test_merkle_insert() {
        let test = ProgramTest::new(
            "basic_merkle_tree",
            crate::ID,
            processor!(process_instruction),
        );

        let owner = Keypair::new();

        let (banks_client, payer, _recent_blockhash) = test.start().await;

        let (merkle_tree_pda, _bump) =
            Pubkey::find_program_address(&[b"merkle_tree", &owner.pubkey().to_bytes()], &crate::ID);

        let leaf1 = [1; 32];
        let tx = add_leaf_tx(&banks_client, &payer, &owner, merkle_tree_pda, leaf1).await;
        banks_client.process_transaction(tx).await.unwrap();
        let merkle_tree = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(merkle_tree.leaves, [leaf1]);
        assert_eq!(merkle_tree.root, leaf1);

        let leaf2 = [2; 32];
        let tx = add_leaf_tx(&banks_client, &payer, &owner, merkle_tree_pda, leaf2).await;
        banks_client.process_transaction(tx).await.unwrap();
        let merkle_tree = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(merkle_tree.leaves.len(), 2);
        assert_eq!(merkle_tree.leaves, [leaf1, leaf2]);
        assert_eq!(merkle_tree.root, keccak::hashv(&[&leaf1, &leaf2]).0);

        let leaf3 = [3; 32];
        let tx = add_leaf_tx(&banks_client, &payer, &owner, merkle_tree_pda, leaf3).await;

        let node0 = keccak::hashv(&[&leaf1, &leaf2]).0;
        let node1 = keccak::hashv(&[&leaf3, &leaf3]).0;
        let last_root = keccak::hashv(&[&node0, &node1]).0;

        let sim_result = banks_client.simulate_transaction(tx.clone()).await.unwrap();
        let logs = sim_result.simulation_details.unwrap().logs;
        let expected_log = format!("New Merkle root: {:?}", last_root);
        assert!(
            logs.iter().any(|log| log.contains(&expected_log)),
            "Merkle tree log not found among\n{:?}",
            logs
        );

        banks_client.process_transaction(tx).await.unwrap();
        let merkle_tree = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(merkle_tree.leaves, [leaf1, leaf2, leaf3]);
        assert_eq!(merkle_tree.root, last_root);
    }

    async fn add_leaf_tx(
        banks_client: &BanksClient,
        payer: &Keypair,
        owner: &Keypair,
        merkle_tree_pda: Pubkey,
        leaf: [u8; 32],
    ) -> Transaction {
        let ix_data = borsh::to_vec(&InsertLeaf { leaf }).expect("Failed to serialize instruction");

        let insert_ix = Instruction {
            program_id: crate::ID,
            accounts: vec![
                AccountMeta::new(merkle_tree_pda, false),
                AccountMeta::new_readonly(owner.pubkey(), true),
                AccountMeta::new(payer.pubkey(), true),
                AccountMeta::new_readonly(system_program::ID, false),
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

    async fn fetch_tree(banks_client: &BanksClient, merkle_tree_pda: Pubkey) -> MerkleTree {
        let pda_account_data = banks_client
            .get_account(merkle_tree_pda)
            .await
            .unwrap()
            .expect("PDA account should now exist");
        MerkleTree::try_from_slice(&pda_account_data.data).expect("deserialize")
    }
}
