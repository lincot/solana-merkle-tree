use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::pubkey::Pubkey;

#[derive(BorshSerialize, BorshDeserialize, Debug, Default)]
pub struct MerkleTree {
    pub root: [u8; 32],
    pub leaves: Vec<[u8; 32]>,
}

impl MerkleTree {
    pub const fn space(leaves_len: usize) -> usize {
        let space_root = 32;
        let space_leaves = 4 + leaves_len * 32;
        space_root + space_leaves
    }

    pub fn find_pda(owner: Pubkey) -> (Pubkey, u8) {
        let seeds: &[&[u8]] = &[b"merkle_tree", &owner.to_bytes()];
        Pubkey::find_program_address(seeds, &crate::ID)
    }
}
