use crate::processor::process_instruction;
use solana_program::{declare_id, entrypoint};

pub mod processor;
pub mod state;

declare_id!("merKipqVdFnEjxHxJRauruM2f4qs1cSNAcYoe27d6Sb");
entrypoint!(process_instruction);
