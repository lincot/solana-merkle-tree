use crate::{default_nodes::DEFAULT_NODES, state::MerkleTreeHeader};
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
pub enum MerkleTreeInstruction {
    InitTree { depth: u8 },
    AppendLeaf { leaf: [u8; 32] },
    ModifyLeaf { leaf: [u8; 32], index: u32 },
    RemoveLeaf { index: u32 },
}

// NOTE: Accounts are the same for each ix for simplicity.
/// Processes an instruction.
///
/// Accounts expected:
/// 0. `[writable]`         The Merkle tree PDA.
/// 1. `[signer]`           The Merkle tree owner.
/// 2. `[writable, signer]` The payer.
/// 3. `[]`                 System program.
pub(crate) fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = MerkleTreeInstruction::try_from_slice(instruction_data)
        .map_err(|_| ProgramError::InvalidInstructionData)?;

    let account_info_iter = &mut accounts.iter();
    let merkle_tree_info = next_account_info(account_info_iter)?;
    let owner_info = next_account_info(account_info_iter)?;
    let payer_info = next_account_info(account_info_iter)?;
    let system_program_info = next_account_info(account_info_iter)?;

    if !owner_info.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    let (merkle_tree_pda, merkle_tree_bump) =
        Pubkey::find_program_address(&[b"merkle_tree", &owner_info.key.to_bytes()], &crate::ID);
    if merkle_tree_info.key != &merkle_tree_pda {
        return Err(ProgramError::InvalidArgument);
    }
    let merkle_tree_seeds = &[
        &b"merkle_tree"[..],
        &owner_info.key.to_bytes(),
        &[merkle_tree_bump],
    ];

    if system_program_info.key != &system_program::ID {
        return Err(ProgramError::InvalidArgument);
    }

    match instruction {
        MerkleTreeInstruction::InitTree { depth } => init_tree(
            merkle_tree_info,
            payer_info,
            program_id,
            merkle_tree_seeds,
            depth,
        )?,
        MerkleTreeInstruction::AppendLeaf { leaf } => {
            append_leaf(merkle_tree_info, payer_info, merkle_tree_seeds, leaf)?
        }
        MerkleTreeInstruction::ModifyLeaf { leaf, index } => {
            modify_leaf(merkle_tree_info, index, leaf)?
        }
        MerkleTreeInstruction::RemoveLeaf { index } => {
            remove_leaf(merkle_tree_info, payer_info, index)?
        }
    }

    Ok(())
}

fn init_tree<'info>(
    merkle_tree_info: &AccountInfo<'info>,
    payer_info: &AccountInfo<'info>,
    program_id: &Pubkey,
    merkle_tree_seeds: &[&[u8]],
    depth: u8,
) -> Result<(), ProgramError> {
    let size = MerkleTreeHeader::SPACE;
    let rent = Rent::get()?;
    let lamports_required = rent.minimum_balance(size);

    let ix = system_instruction::create_account(
        payer_info.key,
        merkle_tree_info.key,
        lamports_required,
        size as u64,
        program_id,
    );

    invoke_signed(
        &ix,
        &[payer_info.clone(), merkle_tree_info.clone()],
        &[merkle_tree_seeds],
    )?;

    let merkle_tree = MerkleTreeHeader { depth, len: 0 };

    merkle_tree.serialize(&mut &mut merkle_tree_info.try_borrow_mut_data()?[..])?;

    Ok(())
}

fn append_leaf<'info>(
    merkle_tree_info: &AccountInfo<'info>,
    payer_info: &AccountInfo<'info>,
    merkle_tree_seeds: &[&[u8]],
    leaf: [u8; 32],
) -> Result<(), ProgramError> {
    if merkle_tree_info.data_is_empty() {
        return Err(ProgramError::InvalidAccountData);
    }

    let (depth, leaf_n) = {
        let merkle_tree_data = &mut merkle_tree_info.try_borrow_mut_data()?;
        let (mut header_data, _) = merkle_tree_data.split_at_mut(MerkleTreeHeader::SPACE);
        let mut tree_header = MerkleTreeHeader::try_from_slice(header_data)?;
        tree_header.len += 1;
        tree_header.serialize(&mut header_data)?;
        (tree_header.depth as u32, tree_header.len - 1)
    };

    let initial_size = merkle_tree_info.data_len();
    let size = initial_size
        + 32 * (1 + if leaf_n == 0 {
            depth
        } else {
            leaf_n.trailing_zeros()
        } as usize);
    let rent = Rent::get()?;
    let lamports_required = rent.minimum_balance(size);

    let ix = system_instruction::transfer(
        payer_info.key,
        merkle_tree_info.key,
        lamports_required - merkle_tree_info.lamports(),
    );
    invoke_signed(
        &ix,
        &[payer_info.clone(), merkle_tree_info.clone()],
        &[merkle_tree_seeds],
    )?;
    merkle_tree_info.realloc(size, true)?;

    modify_leaf_(merkle_tree_info, depth, leaf_n, leaf)?;

    Ok(())
}

fn remove_leaf<'info>(
    merkle_tree_info: &AccountInfo<'info>,
    payer_info: &AccountInfo<'info>,
    leaf_n: u32,
) -> Result<(), ProgramError> {
    let (depth, last_leaf_n) = {
        let merkle_tree_data = &mut merkle_tree_info.try_borrow_mut_data()?;
        let (mut header_data, _) = merkle_tree_data.split_at_mut(MerkleTreeHeader::SPACE);
        let mut tree_header = MerkleTreeHeader::try_from_slice(header_data)?;

        if leaf_n >= tree_header.len {
            return Err(ProgramError::InvalidArgument);
        }

        tree_header.len -= 1;
        tree_header.serialize(&mut header_data)?;
        (tree_header.depth as u32, tree_header.len)
    };

    let initial_size = merkle_tree_info.data_len();
    let size = initial_size
        - 32 * (1 + if last_leaf_n == 0 {
            depth
        } else {
            last_leaf_n.trailing_zeros()
        } as usize);
    let rent = Rent::get()?;
    let lamports_required = rent.minimum_balance(size);
    let extra_lamports = merkle_tree_info.lamports() - lamports_required;

    **merkle_tree_info.try_borrow_mut_lamports()? -= extra_lamports;
    **payer_info.try_borrow_mut_lamports()? += extra_lamports;

    let last_leaf = {
        let merkle_tree_data = &mut merkle_tree_info.try_borrow_mut_data()?;
        let (_, node_data) = merkle_tree_data.split_at_mut(MerkleTreeHeader::SPACE);
        *get_node(node_data, get_leaf_index(depth, last_leaf_n))
    };

    modify_leaf_(merkle_tree_info, depth, leaf_n, last_leaf)?;
    modify_leaf_(merkle_tree_info, depth, last_leaf_n, [0; 32])?;

    merkle_tree_info.realloc(size, false)?;

    Ok(())
}

fn modify_leaf(
    merkle_tree_info: &AccountInfo,
    leaf_n: u32,
    leaf: [u8; 32],
) -> Result<(), ProgramError> {
    let depth = {
        let merkle_tree_data = &merkle_tree_info.try_borrow_data()?;
        let (header_data, _) = merkle_tree_data.split_at(MerkleTreeHeader::SPACE);
        let tree_header = MerkleTreeHeader::try_from_slice(header_data)?;
        if leaf_n >= tree_header.len {
            return Err(ProgramError::InvalidArgument);
        }
        tree_header.depth as u32
    };

    modify_leaf_(merkle_tree_info, depth, leaf_n, leaf)
}

fn modify_leaf_(
    merkle_tree_info: &AccountInfo,
    depth: u32,
    leaf_n: u32,
    leaf: [u8; 32],
) -> Result<(), ProgramError> {
    let merkle_tree_data = &mut merkle_tree_info.try_borrow_mut_data()?;
    let (_, node_data) = merkle_tree_data.split_at_mut(MerkleTreeHeader::SPACE);

    let nodes_len = (node_data.len() / 32) as u32;
    let init_index = get_leaf_index(depth, leaf_n);
    let mut curr_index = init_index;
    let mut curr_n = leaf_n;
    let mut curr_node = leaf;

    write_node(node_data, init_index, &leaf);

    for level in 0..depth {
        let step = get_step(curr_n, curr_index, depth, level);

        let sibling_node = if step.sibling < nodes_len {
            get_node(node_data, step.sibling)
        } else {
            &DEFAULT_NODES[level as usize]
        };

        let parent_node = if step.sibling_is_left {
            keccak::hashv(&[sibling_node, &curr_node]).0
        } else {
            keccak::hashv(&[&curr_node, sibling_node]).0
        };

        write_node(node_data, step.parent, &parent_node);

        curr_index = step.parent;
        curr_node = parent_node;
        curr_n /= 2;
    }

    let root = get_root(node_data, depth);
    msg!("New Merkle root: {:?}", root);

    Ok(())
}

fn write_node(node_data: &mut [u8], index: u32, node: &[u8; 32]) {
    get_node_mut(node_data, index).copy_from_slice(node)
}

fn get_root(node_data: &[u8], depth: u32) -> &[u8; 32] {
    get_node(node_data, depth)
}

fn get_node(node_data: &[u8], index: u32) -> &[u8; 32] {
    let i = 32 * index as usize;
    (&node_data[i..i + 32]).try_into().unwrap()
}

fn get_node_mut(node_data: &mut [u8], index: u32) -> &mut [u8; 32] {
    let i = 32 * index as usize;
    (&mut node_data[i..i + 32]).try_into().unwrap()
}

#[derive(Debug)]
struct Step {
    sibling: u32,
    parent: u32,
    sibling_is_left: bool,
}

fn get_step(col: u32, curr_index: u32, depth: u32, level: u32) -> Step {
    if col % 2 == 0 {
        Step {
            sibling: curr_index + get_sibling_distance(depth, level, col / 2),
            parent: curr_index + 1,
            sibling_is_left: col % 2 == 1,
        }
    } else {
        let sibling = curr_index - get_sibling_distance(depth, level, col / 2);
        Step {
            sibling,
            parent: sibling + 1,
            sibling_is_left: col % 2 == 1,
        }
    }
}

fn get_leaf_index(depth: u32, n: u32) -> u32 {
    if n == 0 {
        return 0;
    }

    let m = (n + 1) / 2;
    n / 2 + 2 * m + (depth - 1) + (m - 1 - (m - 1).count_ones())
}

fn get_sibling_distance(depth: u32, level: u32, pair_n: u32) -> u32 {
    (1 << (level + 1))
        + if pair_n == 0 {
            depth - level - 1
        } else {
            pair_n.trailing_zeros()
        }
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

        let depth = 5;

        let mut expected_leaves = [[0; 32]; 1 << 5];

        let tx = get_tx(
            &banks_client,
            &payer,
            &owner,
            merkle_tree_pda,
            MerkleTreeInstruction::InitTree { depth },
        )
        .await;
        banks_client.process_transaction(tx).await.unwrap();

        let leaf1 = [1; 32];
        expected_leaves[0] = leaf1;
        let tx = get_tx(
            &banks_client,
            &payer,
            &owner,
            merkle_tree_pda,
            MerkleTreeInstruction::AppendLeaf { leaf: leaf1 },
        )
        .await;
        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(leaves, expected_leaves[..1]);
        assert_eq!(root, compute_merkle_root(&expected_leaves));

        let leaf2 = [2; 32];
        expected_leaves[1] = leaf2;
        let tx = get_tx(
            &banks_client,
            &payer,
            &owner,
            merkle_tree_pda,
            MerkleTreeInstruction::AppendLeaf { leaf: leaf2 },
        )
        .await;
        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(leaves, expected_leaves[..2]);
        assert_eq!(root, compute_merkle_root(&expected_leaves));

        let leaf3 = [3; 32];
        expected_leaves[2] = leaf3;
        let tx = get_tx(
            &banks_client,
            &payer,
            &owner,
            merkle_tree_pda,
            MerkleTreeInstruction::AppendLeaf { leaf: leaf3 },
        )
        .await;

        let expected_root = compute_merkle_root(&expected_leaves);

        let sim_result = banks_client.simulate_transaction(tx.clone()).await.unwrap();
        let logs = sim_result.simulation_details.unwrap().logs;
        let expected_log = format!("New Merkle root: {:?}", expected_root);
        assert!(
            logs.iter().any(|log| log.contains(&expected_log)),
            "Merkle tree log not found among\n{:?}",
            logs
        );

        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(leaves, expected_leaves[..3]);
        assert_eq!(root, expected_root);

        let leaf2 = [42; 32];
        expected_leaves[1] = leaf2;
        let tx = get_tx(
            &banks_client,
            &payer,
            &owner,
            merkle_tree_pda,
            MerkleTreeInstruction::ModifyLeaf {
                leaf: leaf2,
                index: 1,
            },
        )
        .await;
        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(leaves, expected_leaves[..3]);
        assert_eq!(root, compute_merkle_root(&expected_leaves));

        expected_leaves[1] = expected_leaves[2];
        expected_leaves[2] = [0; 32];
        let tx = get_tx(
            &banks_client,
            &payer,
            &owner,
            merkle_tree_pda,
            MerkleTreeInstruction::RemoveLeaf { index: 1 },
        )
        .await;
        banks_client.process_transaction(tx).await.unwrap();
        let (root, leaves) = fetch_tree(&banks_client, merkle_tree_pda).await;
        assert_eq!(leaves, expected_leaves[..2]);
        assert_eq!(root, compute_merkle_root(&expected_leaves));
    }

    async fn get_tx(
        banks_client: &BanksClient,
        payer: &Keypair,
        owner: &Keypair,
        merkle_tree_pda: Pubkey,
        ix: MerkleTreeInstruction,
    ) -> Transaction {
        let ix_data = borsh::to_vec(&ix).expect("Failed to serialize instruction");

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

    async fn fetch_tree(
        banks_client: &BanksClient,
        merkle_tree_pda: Pubkey,
    ) -> ([u8; 32], Vec<[u8; 32]>) {
        let pda_account_data = banks_client
            .get_account(merkle_tree_pda)
            .await
            .unwrap()
            .expect("PDA account should now exist");
        let (header_data, node_data) = pda_account_data.data.split_at(MerkleTreeHeader::SPACE);
        let header = MerkleTreeHeader::try_from_slice(header_data).expect("deserialize error");

        (
            *get_root(node_data, header.depth as _),
            get_leaves(node_data, header.depth as _, header.len),
        )
    }

    fn get_leaves(node_data: &[u8], depth: u32, len: u32) -> Vec<[u8; 32]> {
        let mut index = 0;
        (0..len)
            .map(|n| {
                let res = *get_node(node_data, index);

                if n % 2 == 0 {
                    index += get_sibling_distance(depth, 0, n / 2);
                } else {
                    index += 1;
                }

                res
            })
            .collect()
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
}
