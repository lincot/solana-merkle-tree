use borsh::{BorshDeserialize, BorshSerialize};

#[derive(BorshSerialize, BorshDeserialize, Debug, Default)]
pub struct MerkleTreeHeader {
    pub depth: u8,
    pub len: u32,
}

impl MerkleTreeHeader {
    pub const SPACE: usize = 1 + 4;
}
