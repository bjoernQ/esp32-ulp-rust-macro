use bitfield::bitfield;
use std::collections::HashMap;

use crate::parser::{Condition, Instruction, Reg, Value};

#[derive(Debug)]
pub(crate) enum CodeGenError {
    GenericError(String),
}

#[derive(Debug, Clone)]
pub(crate) struct Label {
    pub(crate) name: String,
    pub(crate) address: u32,
}

pub(crate) trait CodeGen {
    fn generate(&self, ast: Vec<Instruction>) -> Result<(Vec<u8>, Vec<Label>), CodeGenError>;
}

pub(crate) fn create_codegen() -> impl CodeGen {
    Esp32CodeGen::new()
}

pub(crate) struct Esp32CodeGen {}

impl Esp32CodeGen {
    pub(crate) fn new() -> Esp32CodeGen {
        Esp32CodeGen {}
    }
}
bitfield! {
    #[derive(Default, Debug)]
    pub struct AluOp(u32);
    opcode, set_opcode: 31, 28;
    immediate, set_immediate: 27, 25;
    alu_sel, set_alu_sel: 24, 21;
    rsrc2, set_rsrc2: 5, 4;
    rsrc1, set_rsrc1: 3, 2;
    rdst, set_rdst: 1, 0;
    imm, set_imm: 19, 4;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct LdStOp(u32);
    opcode, set_opcode: 31, 28;
    is_store, set_is_store: 27,25;
    rsrc, set_rsrc: 3, 2;
    rdst, set_rdst: 1, 0;
    offset, set_offset: 20, 10;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct JumpOp(u32);
    opcode, set_opcode: 31, 28;
    typ, set_typ: 24, 22;
    sel, set_sel: 21, 21;
    is_store, set_is_store: 27,25;
    rdst, set_rdst: 1, 0;
    imm, set_imm: 12, 2;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct JumpROp(u32);
    opcode, set_opcode: 31, 28;
    typ, set_typ: 27, 25;
    step, set_step: 24, 17;
    condition, set_condition: 16, 16;
    threshold, set_threshold: 15, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct JumpSOp(u32);
    opcode, set_opcode: 31, 28;
    typ, set_typ: 27, 25;
    step, set_step: 24, 17;
    condition, set_condition: 16, 15;
    threshold, set_threshold: 7, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct HaltOp(u32);
    opcode, set_opcode: 31, 28;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct WakeOp(u32);
    opcode, set_opcode: 31, 28;
    zero, set_zero: 27, 25;
    one, set_one: 0, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct SleepOp(u32);
    opcode, set_opcode: 31, 28;
    one, set_one: 27, 25;
    sleep_reg, set_sleep_reg: 3, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct WaitOp(u32);
    opcode, set_opcode: 31, 28;
    cycles, set_cycles: 15, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct AdcOp(u32);
    opcode, set_opcode: 31, 28;
    sel, set_sel: 6, 6;
    sar_mux, set_sar_mux: 5, 2;
    dst_reg, set_dst_reg: 1, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct I2cOp(u32);
    opcode, set_opcode: 31, 28;
    rw, set_rw: 27, 27;
    i2c_sel, set_i2c_sel: 25, 22;
    high, set_high: 21, 19;
    low, set_low: 18, 16;
    data, set_data: 15, 8;
    sub_addr, set_sub_addr: 7, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct RegRdOp(u32);
    opcode, set_opcode: 31, 28;
    high, set_high: 27, 23;
    low, set_low: 22, 18;
    addr, set_addr: 9, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct RegWrOp(u32);
    opcode, set_opcode: 31, 28;
    high, set_high: 27, 23;
    low, set_low: 22, 18;
    data, set_data: 17, 10;
    addr, set_addr: 9, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct TsensOp(u32);
    opcode, set_opcode: 31, 28;
    delay, set_delay: 16, 2;
    dreg, set_dreg: 1, 0;
}

bitfield! {
    #[derive(Default, Debug)]
    pub struct NopOp(u32);
    opcode, set_opcode: 31, 28;
}

#[derive(Debug)]
enum AluSel {
    Add = 0,
    Sub = 1,
    And = 2,
    Or = 3,
    Move = 4,
    Lsh = 5,
    Rsh = 6,
}

#[derive(Debug)]
enum StageAluSel {
    Inc = 0,
    Dec = 1,
    Rst = 2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ImmediateValue {
    Constant(u32),
    FromLabel(u32),
}

impl ImmediateValue {
    fn value(&self) -> u32 {
        match self {
            ImmediateValue::Constant(v) => *v,
            ImmediateValue::FromLabel(v) => *v,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ResolvedInstruction {
    Move(Reg, Reg),
    MoveImmediate(Reg, ImmediateValue),
    Store(Reg, Reg, ImmediateValue),
    Load(Reg, Reg, ImmediateValue),
    Add(Reg, Reg, Reg),
    AddImmediate(Reg, Reg, ImmediateValue),
    Nop,
    Sub(Reg, Reg, Reg),
    SubImmediate(Reg, Reg, ImmediateValue),
    And(Reg, Reg, Reg),
    AndImmediate(Reg, Reg, ImmediateValue),
    Or(Reg, Reg, Reg),
    OrImmediate(Reg, Reg, ImmediateValue),
    Lsh(Reg, Reg, Reg),
    LshImmediate(Reg, Reg, ImmediateValue),
    Rsh(Reg, Reg, Reg),
    RshImmediate(Reg, Reg, ImmediateValue),
    Jump(Reg),
    JumpImmediate(ImmediateValue),
    JumpConditional(Reg, Condition),
    JumpConditionalImmediate(ImmediateValue, Condition),
    JumpR(ImmediateValue, ImmediateValue, Condition),
    JumpS(ImmediateValue, ImmediateValue, Condition),
    StageRst,
    StageInc(ImmediateValue),
    StageDec(ImmediateValue),
    Halt,
    Wake,
    Sleep(ImmediateValue),
    Wait(ImmediateValue),
    Tsens(Reg, ImmediateValue),
    Adc(Reg, ImmediateValue, ImmediateValue),
    I2cRd(
        ImmediateValue,
        ImmediateValue,
        ImmediateValue,
        ImmediateValue,
    ),
    I2cWr(
        ImmediateValue,
        ImmediateValue,
        ImmediateValue,
        ImmediateValue,
        ImmediateValue,
    ),
    RegRd(ImmediateValue, ImmediateValue, ImmediateValue),
    RegWr(
        ImmediateValue,
        ImmediateValue,
        ImmediateValue,
        ImmediateValue,
    ),
    Long(ImmediateValue),
    NoCode,
}

impl CodeGen for Esp32CodeGen {
    fn generate(&self, ast: Vec<Instruction>) -> Result<(Vec<u8>, Vec<Label>), CodeGenError> {
        let mut code = Vec::new();
        let mut symbol2value = HashMap::<String, ImmediateValue>::new();
        let mut offset = 0;

        for instr in &ast {
            match instr {
                Instruction::Comment => (),
                Instruction::Label(label) => {
                    symbol2value.insert(label.clone(), ImmediateValue::FromLabel(offset));
                }
                Instruction::Set(symbol, Value::Number(value)) => {
                    symbol2value.insert(symbol.clone(), ImmediateValue::Constant(*value));
                }
                Instruction::Global(_) => (),
                Instruction::JumpR(_, _, c) => {
                    offset += match c {
                        Condition::Eq => 8, // emulated by two JUMPR
                        _ => 4,
                    };
                }
                Instruction::JumpS(_, _, c) => {
                    offset += match c {
                        Condition::Eq => 8, // emulated by two JUMPR
                        Condition::Gt => 8, // emulated by two JUMPR
                        _ => 4,
                    };
                }
                _ => offset += 4,
            }
        }

        let ast = resolve(&ast, &symbol2value)?;

        for instr in &ast {
            match instr {
                ResolvedInstruction::Move(_, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::MoveImmediate(_, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::Store(_, _, _) => load_store_operation(&mut code, instr),
                ResolvedInstruction::Load(_, _, _) => load_store_operation(&mut code, instr),
                ResolvedInstruction::Add(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::AddImmediate(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::Nop => nop_operation(&mut code, instr),
                ResolvedInstruction::Sub(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::SubImmediate(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::And(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::AndImmediate(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::Or(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::OrImmediate(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::Lsh(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::LshImmediate(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::Rsh(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::RshImmediate(_, _, _) => alu_operation(&mut code, instr),
                ResolvedInstruction::Jump(_) => jump_operation(&mut code, instr),
                ResolvedInstruction::JumpImmediate(_) => jump_operation(&mut code, instr),
                ResolvedInstruction::JumpConditional(_, _) => jump_operation(&mut code, instr),
                ResolvedInstruction::JumpConditionalImmediate(_, _) => {
                    jump_operation(&mut code, instr)
                }
                ResolvedInstruction::JumpR(_, _, _) => {
                    jumpr_operation(code.len(), &mut code, instr)
                }
                ResolvedInstruction::JumpS(_, _, _) => {
                    jumps_operation(code.len(), &mut code, instr)
                }
                ResolvedInstruction::StageRst => alu_operation(&mut code, instr),
                ResolvedInstruction::StageInc(_) => alu_operation(&mut code, instr),
                ResolvedInstruction::StageDec(_) => alu_operation(&mut code, instr),
                ResolvedInstruction::Halt => halt_operation(&mut code, instr),
                ResolvedInstruction::Wake => wake_operation(&mut code, instr),
                ResolvedInstruction::Sleep(_) => sleep_operation(&mut code, instr),
                ResolvedInstruction::Wait(_) => wait_operation(&mut code, instr),
                ResolvedInstruction::Tsens(_, _) => tsens_operation(&mut code, instr),
                ResolvedInstruction::Adc(_, _, _) => adc_operation(&mut code, instr),
                ResolvedInstruction::I2cRd(_, _, _, _) => i2c_rw_operation(&mut code, instr),
                ResolvedInstruction::I2cWr(_, _, _, _, _) => i2c_rw_operation(&mut code, instr),
                ResolvedInstruction::RegRd(_, _, _) => reg_rd_operation(&mut code, instr),
                ResolvedInstruction::RegWr(_, _, _, _) => reg_wr_operation(&mut code, instr),
                ResolvedInstruction::Long(_) => long_pseudo_instr(&mut code, instr),
                _ => Ok(()),
            }?
        }

        let mut labels = Vec::new();
        for (key, value) in symbol2value {
            match value {
                ImmediateValue::Constant(_) => (),
                ImmediateValue::FromLabel(addr) => {
                    labels.push(Label {
                        name: key,
                        address: addr,
                    });
                }
            }
        }

        Ok((code, labels))
    }
}

fn resolve(
    ast: &Vec<Instruction>,
    symbol2value: &HashMap<String, ImmediateValue>,
) -> Result<Vec<ResolvedInstruction>, CodeGenError> {
    let mut resolved = Vec::new();

    for instr in ast {
        let resolved_instr = match instr {
            Instruction::MoveImmediate(reg, value) => {
                ResolvedInstruction::MoveImmediate(*reg, resolve_symbol(value, symbol2value)?)
            }
            Instruction::Store(reg1, reg2, value) => {
                ResolvedInstruction::Store(*reg1, *reg2, resolve_symbol(value, symbol2value)?)
            }
            Instruction::Load(reg1, reg2, value) => {
                ResolvedInstruction::Load(*reg1, *reg2, resolve_symbol(value, symbol2value)?)
            }
            Instruction::AddImmediate(reg1, reg2, value) => ResolvedInstruction::AddImmediate(
                *reg1,
                *reg2,
                resolve_symbol(value, symbol2value)?,
            ),
            Instruction::SubImmediate(reg1, reg2, value) => ResolvedInstruction::SubImmediate(
                *reg1,
                *reg2,
                resolve_symbol(value, symbol2value)?,
            ),
            Instruction::AndImmediate(reg1, reg2, value) => ResolvedInstruction::AndImmediate(
                *reg1,
                *reg2,
                resolve_symbol(value, symbol2value)?,
            ),
            Instruction::OrImmediate(reg1, reg2, value) => {
                ResolvedInstruction::OrImmediate(*reg1, *reg2, resolve_symbol(value, symbol2value)?)
            }
            Instruction::LshImmediate(reg1, reg2, value) => ResolvedInstruction::LshImmediate(
                *reg1,
                *reg2,
                resolve_symbol(value, symbol2value)?,
            ),
            Instruction::RshImmediate(reg1, reg2, value) => ResolvedInstruction::RshImmediate(
                *reg1,
                *reg2,
                resolve_symbol(value, symbol2value)?,
            ),
            Instruction::JumpImmediate(value) => {
                ResolvedInstruction::JumpImmediate(resolve_symbol(value, symbol2value)?)
            }
            Instruction::JumpConditionalImmediate(value, condition) => {
                ResolvedInstruction::JumpConditionalImmediate(
                    resolve_symbol(value, symbol2value)?,
                    *condition,
                )
            }
            Instruction::JumpR(value1, value2, condition) => ResolvedInstruction::JumpR(
                resolve_symbol(value1, symbol2value)?,
                resolve_symbol(value2, symbol2value)?,
                *condition,
            ),
            Instruction::JumpS(value1, value2, condition) => ResolvedInstruction::JumpS(
                resolve_symbol(value1, symbol2value)?,
                resolve_symbol(value2, symbol2value)?,
                *condition,
            ),
            Instruction::StageInc(value) => {
                ResolvedInstruction::StageInc(resolve_symbol(value, symbol2value)?)
            }
            Instruction::StageDec(value) => {
                ResolvedInstruction::StageDec(resolve_symbol(value, symbol2value)?)
            }
            Instruction::Sleep(value) => {
                ResolvedInstruction::Sleep(resolve_symbol(value, symbol2value)?)
            }
            Instruction::Wait(value) => {
                ResolvedInstruction::Wait(resolve_symbol(value, symbol2value)?)
            }
            Instruction::Tsens(reg, value) => {
                ResolvedInstruction::Tsens(*reg, resolve_symbol(value, symbol2value)?)
            }
            Instruction::Adc(reg, value1, value2) => ResolvedInstruction::Adc(
                *reg,
                resolve_symbol(value1, symbol2value)?,
                resolve_symbol(value2, symbol2value)?,
            ),
            Instruction::I2cRd(value1, value2, value3, value4) => ResolvedInstruction::I2cRd(
                resolve_symbol(value1, symbol2value)?,
                resolve_symbol(value2, symbol2value)?,
                resolve_symbol(value3, symbol2value)?,
                resolve_symbol(value4, symbol2value)?,
            ),
            Instruction::I2cWr(value1, value2, value3, value4, value5) => {
                ResolvedInstruction::I2cWr(
                    resolve_symbol(value1, symbol2value)?,
                    resolve_symbol(value2, symbol2value)?,
                    resolve_symbol(value3, symbol2value)?,
                    resolve_symbol(value4, symbol2value)?,
                    resolve_symbol(value5, symbol2value)?,
                )
            }
            Instruction::RegRd(value1, value2, value3) => ResolvedInstruction::RegRd(
                resolve_symbol(value1, symbol2value)?,
                resolve_symbol(value2, symbol2value)?,
                resolve_symbol(value3, symbol2value)?,
            ),
            Instruction::RegWr(value1, value2, value3, value4) => ResolvedInstruction::RegWr(
                resolve_symbol(value1, symbol2value)?,
                resolve_symbol(value2, symbol2value)?,
                resolve_symbol(value3, symbol2value)?,
                resolve_symbol(value4, symbol2value)?,
            ),
            Instruction::Long(value) => {
                ResolvedInstruction::Long(resolve_symbol(value, symbol2value)?)
            }
            Instruction::Move(r1, r2) => ResolvedInstruction::Move(*r1, *r2),
            Instruction::Add(r1, r2, r3) => ResolvedInstruction::Add(*r1, *r2, *r3),
            Instruction::Nop => ResolvedInstruction::Nop,
            Instruction::Sub(r1, r2, r3) => ResolvedInstruction::Sub(*r1, *r2, *r3),
            Instruction::And(r1, r2, r3) => ResolvedInstruction::And(*r1, *r2, *r3),
            Instruction::Or(r1, r2, r3) => ResolvedInstruction::Or(*r1, *r2, *r3),
            Instruction::Lsh(r1, r2, r3) => ResolvedInstruction::Lsh(*r1, *r2, *r3),
            Instruction::Rsh(r1, r2, r3) => ResolvedInstruction::Rsh(*r1, *r2, *r3),
            Instruction::Jump(r1) => ResolvedInstruction::Jump(*r1),
            Instruction::JumpConditional(r1, c) => ResolvedInstruction::JumpConditional(*r1, *c),
            Instruction::StageRst => ResolvedInstruction::StageRst,
            Instruction::Halt => ResolvedInstruction::Halt,
            Instruction::Wake => ResolvedInstruction::Wake,
            _ => ResolvedInstruction::NoCode,
        };

        if resolved_instr != ResolvedInstruction::NoCode {
            resolved.push(resolved_instr);
        }
    }

    Ok(resolved)
}

fn resolve_symbol(
    value: &Value,
    symbol2value: &HashMap<String, ImmediateValue>,
) -> Result<ImmediateValue, CodeGenError> {
    match value {
        Value::Number(n) => Ok(ImmediateValue::Constant(*n)),
        Value::Identifier(identifier) => {
            let resolved = symbol2value.get(identifier);

            match resolved {
                Some(resolved) => Ok(resolved.clone()),
                None => Err(CodeGenError::GenericError(format!(
                    "No symbol {}",
                    identifier
                ))),
            }
        }
    }
}

fn alu_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let word = {
        let mut res = AluOp::default();
        res.set_opcode(7);

        match instr {
            ResolvedInstruction::Move(rdst, rsrc) => {
                res.set_alu_sel(AluSel::Move as u32);
                res.set_immediate(0);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc as u32);
                res.set_rsrc2(*rsrc as u32);
                res.0
            }
            ResolvedInstruction::MoveImmediate(rdst, value) => {
                let imm = match value {
                    ImmediateValue::Constant(c) => *c,
                    ImmediateValue::FromLabel(a) => *a / 4,
                };

                res.set_alu_sel(AluSel::Move as u32);
                res.set_immediate(1);
                res.set_rdst(*rdst as u32);
                res.set_imm(imm);
                res.0
            }
            ResolvedInstruction::Add(rdst, rsrc1, rsrc2) => {
                res.set_alu_sel(AluSel::Add as u32);
                res.set_immediate(0);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_rsrc2(*rsrc2 as u32);
                res.0
            }
            ResolvedInstruction::AddImmediate(rdst, rsrc1, value) => {
                res.set_alu_sel(AluSel::Add as u32);
                res.set_immediate(1);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_imm(value.value());
                res.0
            }
            ResolvedInstruction::Sub(rdst, rsrc1, rsrc2) => {
                res.set_alu_sel(AluSel::Sub as u32);
                res.set_immediate(0);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_rsrc2(*rsrc2 as u32);
                res.0
            }
            ResolvedInstruction::SubImmediate(rdst, rsrc1, value) => {
                res.set_alu_sel(AluSel::Sub as u32);
                res.set_immediate(1);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_imm(value.value());
                res.0
            }
            ResolvedInstruction::And(rdst, rsrc1, rsrc2) => {
                res.set_alu_sel(AluSel::And as u32);
                res.set_immediate(0);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_rsrc2(*rsrc2 as u32);
                res.0
            }
            ResolvedInstruction::AndImmediate(rdst, rsrc1, value) => {
                res.set_alu_sel(AluSel::And as u32);
                res.set_immediate(1);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_imm(value.value());
                res.0
            }
            ResolvedInstruction::Or(rdst, rsrc1, rsrc2) => {
                res.set_alu_sel(AluSel::Or as u32);
                res.set_immediate(0);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_rsrc2(*rsrc2 as u32);
                res.0
            }
            ResolvedInstruction::OrImmediate(rdst, rsrc1, value) => {
                res.set_alu_sel(AluSel::Or as u32);
                res.set_immediate(1);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_imm(value.value());
                res.0
            }
            ResolvedInstruction::Lsh(rdst, rsrc1, rsrc2) => {
                res.set_alu_sel(AluSel::Lsh as u32);
                res.set_immediate(0);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_rsrc2(*rsrc2 as u32);
                res.0
            }
            ResolvedInstruction::LshImmediate(rdst, rsrc1, value) => {
                res.set_alu_sel(AluSel::Lsh as u32);
                res.set_immediate(1);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_imm(value.value());
                res.0
            }
            ResolvedInstruction::Rsh(rdst, rsrc1, rsrc2) => {
                res.set_alu_sel(AluSel::Rsh as u32);
                res.set_immediate(0);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_rsrc2(*rsrc2 as u32);
                res.0
            }
            ResolvedInstruction::RshImmediate(rdst, rsrc1, value) => {
                res.set_alu_sel(AluSel::Rsh as u32);
                res.set_immediate(1);
                res.set_rdst(*rdst as u32);
                res.set_rsrc1(*rsrc1 as u32);
                res.set_imm(value.value());
                res.0
            }
            ResolvedInstruction::StageRst => {
                res.set_alu_sel(StageAluSel::Rst as u32);
                res.set_immediate(2);
                res.0
            }
            ResolvedInstruction::StageInc(value) => {
                res.set_alu_sel(StageAluSel::Inc as u32);
                res.set_immediate(2);
                res.set_imm(value.value());
                res.0
            }
            ResolvedInstruction::StageDec(value) => {
                res.set_alu_sel(StageAluSel::Dec as u32);
                res.set_immediate(2);
                res.set_imm(value.value());
                res.0
            }
            _ => {
                return Err(CodeGenError::GenericError(format!(
                    "Not an ALU operation {:?}",
                    instr
                )))
            }
        }
    };

    push_word(word, code);
    Ok(())
}

fn load_store_operation(
    code: &mut Vec<u8>,
    instr: &ResolvedInstruction,
) -> Result<(), CodeGenError> {
    let word = {
        let mut res = LdStOp::default();

        match instr {
            ResolvedInstruction::Store(rdst, rsrc, value) => {
                res.set_opcode(6);
                res.set_is_store(0b100);
                res.set_rdst(*rdst as u32);
                res.set_rsrc(*rsrc as u32);
                res.set_offset(value.value() / 4);
                res.0
            }
            ResolvedInstruction::Load(rdst, rsrc, value) => {
                res.set_opcode(13);
                res.set_rdst(*rdst as u32);
                res.set_rsrc(*rsrc as u32);
                res.set_offset(value.value() / 4);
                res.0
            }
            _ => {
                return Err(CodeGenError::GenericError(format!(
                    "Not an LD/ST operation {:?}",
                    instr
                )))
            }
        }
    };

    push_word(word, code);
    Ok(())
}

fn jump_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let word = {
        let mut res = JumpOp::default();
        res.set_opcode(8);

        match instr {
            ResolvedInstruction::Jump(rdst) => {
                res.set_typ(0);
                res.set_sel(1);
                res.set_rdst(*rdst as u32);
                res.0
            }
            ResolvedInstruction::JumpImmediate(value) => {
                res.set_typ(0);
                res.set_sel(0);
                res.set_imm(value.value() / 4);
                res.0
            }
            ResolvedInstruction::JumpConditional(rdst, condition)
                if *condition == Condition::Eq || *condition == Condition::Ov =>
            {
                res.set_typ(match *condition {
                    Condition::Eq => 1,
                    Condition::Ov => 2,
                    _ => {
                        return Err(CodeGenError::GenericError(format!(
                            "Condition not allowed here {:?}",
                            condition
                        )))
                    }
                });
                res.set_sel(1);
                res.set_rdst(*rdst as u32);
                res.0
            }
            ResolvedInstruction::JumpConditionalImmediate(value, condition)
                if *condition == Condition::Eq || *condition == Condition::Ov =>
            {
                res.set_typ(match *condition {
                    Condition::Eq => 1,
                    Condition::Ov => 2,
                    _ => {
                        return Err(CodeGenError::GenericError(format!(
                            "Condition not allowed here {:?}",
                            condition
                        )))
                    }
                });
                res.set_sel(0);
                res.set_imm(value.value() / 4);
                res.0
            }
            _ => {
                return Err(CodeGenError::GenericError(format!(
                    "Not an JUMP operation {:?}",
                    instr
                )))
            }
        }
    };

    push_word(word, code);
    Ok(())
}

fn jumpr_operation(
    pc: usize,
    code: &mut Vec<u8>,
    instr: &ResolvedInstruction,
) -> Result<(), CodeGenError> {
    let word = {
        let mut res = JumpROp::default();
        res.set_opcode(8);
        res.set_typ(1);

        match instr {
            ResolvedInstruction::JumpR(target, threshold, condition) => {
                match condition {
                    Condition::Eq => {
                        jumpr_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpR(
                                ImmediateValue::FromLabel(pc as u32 + 8),
                                ImmediateValue::Constant(threshold.value() + 1),
                                Condition::Ge,
                            ),
                        )?;
                        return jumpr_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpR(*target, *threshold, Condition::Ge),
                        );
                    }
                    Condition::Le => {
                        return jumpr_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpR(
                                *target,
                                ImmediateValue::Constant(threshold.value() + 1),
                                Condition::Lt,
                            ),
                        );
                    }
                    Condition::Gt => {
                        return jumpr_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpR(
                                *target,
                                ImmediateValue::Constant(threshold.value() + 1),
                                Condition::Ge,
                            ),
                        );
                    }
                    _ => (),
                };

                let step = (target.value() as i32 - pc as i32) / 4;
                let step = if step < 0 {
                    (((step * -1) as u32) & 0b111_1111) | 0b1000_0000
                } else {
                    step as u32
                };

                res.set_step(step);
                res.set_threshold(threshold.value());
                res.set_condition(match condition {
                    Condition::Lt => 0,
                    Condition::Ge => 1,
                    _ => {
                        return Err(CodeGenError::GenericError(format!(
                            "Condition not allowed here {:?}",
                            condition
                        )))
                    }
                });
                res.0
            }
            _ => {
                return Err(CodeGenError::GenericError(format!(
                    "Not an JUMPR operation {:?}",
                    instr
                )))
            }
        }
    };

    push_word(word, code);
    Ok(())
}

fn jumps_operation(
    pc: usize,
    code: &mut Vec<u8>,
    instr: &ResolvedInstruction,
) -> Result<(), CodeGenError> {
    let word = {
        let mut res = JumpSOp::default();
        res.set_opcode(8);
        res.set_typ(2);

        match instr {
            ResolvedInstruction::JumpS(target, threshold, condition) => {
                match condition {
                    Condition::Eq => {
                        jumps_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpS(
                                ImmediateValue::FromLabel(pc as u32 + 8),
                                *threshold,
                                Condition::Lt,
                            ),
                        )?;
                        return jumps_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpS(*target, *threshold, Condition::Le),
                        );
                    }
                    Condition::Gt => {
                        jumps_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpS(
                                ImmediateValue::FromLabel(pc as u32 + 8),
                                *threshold,
                                Condition::Le,
                            ),
                        )?;
                        return jumps_operation(
                            code.len(),
                            code,
                            &ResolvedInstruction::JumpS(*target, *threshold, Condition::Ge),
                        );
                    }
                    _ => (),
                };

                let step = (target.value() as i32 - pc as i32) / 4;
                let step = if step < 0 {
                    (((step * -1) as u32) & 0b111_1111) | 0b1000_0000
                } else {
                    step as u32
                };

                res.set_step(step);
                res.set_threshold(threshold.value());
                res.set_condition(match condition {
                    Condition::Lt => 0,
                    Condition::Le => 2,
                    Condition::Ge => 1,
                    _ => {
                        return Err(CodeGenError::GenericError(format!(
                            "Condition not allowed here {:?}",
                            condition
                        )))
                    }
                });
                res.0
            }
            _ => {
                return Err(CodeGenError::GenericError(format!(
                    "Not an JUMPS operation {:?}",
                    instr
                )))
            }
        }
    };

    push_word(word, code);
    Ok(())
}

fn halt_operation(code: &mut Vec<u8>, _instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = HaltOp::default();
    word.set_opcode(11);
    push_word(word.0, code);
    Ok(())
}

fn wake_operation(code: &mut Vec<u8>, _instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = WakeOp::default();
    word.set_opcode(9);
    word.set_zero(0);
    word.set_one(1);
    push_word(word.0, code);
    Ok(())
}

fn sleep_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = SleepOp::default();
    word.set_opcode(9);
    word.set_one(1);
    match instr {
        ResolvedInstruction::Sleep(sleep_reg) => {
            word.set_sleep_reg(sleep_reg.value());
        }
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an SLEEP operation {:?}",
                instr
            )))
        }
    }
    push_word(word.0, code);
    Ok(())
}

fn wait_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = WaitOp::default();
    word.set_opcode(4);
    match instr {
        ResolvedInstruction::Wait(cycles) => {
            word.set_cycles(cycles.value());
        }
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an WAIT operation {:?}",
                instr
            )))
        }
    }
    push_word(word.0, code);
    Ok(())
}

fn adc_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = AdcOp::default();
    word.set_opcode(5);
    match instr {
        ResolvedInstruction::Adc(reg, sar_sel, mux) => {
            word.set_dst_reg(*reg as u32);
            word.set_sel(sar_sel.value());
            word.set_sar_mux(mux.value());
        }
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an ADC operation {:?}",
                instr
            )))
        }
    }
    push_word(word.0, code);
    Ok(())
}

fn i2c_rw_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = I2cOp::default();
    word.set_opcode(3);
    match instr {
        ResolvedInstruction::I2cRd(sub_addr, high, low, slave_sel) => {
            word.set_rw(0);
            word.set_sub_addr(sub_addr.value());
            word.set_i2c_sel(slave_sel.value());
            word.set_low(low.value());
            word.set_high(high.value());
        }
        ResolvedInstruction::I2cWr(sub_addr, data, high, low, slave_sel) => {
            word.set_rw(1);
            word.set_sub_addr(sub_addr.value());
            word.set_i2c_sel(slave_sel.value());
            word.set_low(low.value());
            word.set_high(high.value());
            word.set_data(data.value());
        }
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an I2C operation {:?}",
                instr
            )))
        }
    }
    push_word(word.0, code);
    Ok(())
}

fn reg_rd_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = RegRdOp::default();
    word.set_opcode(2);
    match instr {
        ResolvedInstruction::RegRd(addr, high, low) => {
            word.set_addr(addr.value());
            word.set_high(high.value());
            word.set_low(low.value());
        }
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an REG_RD operation {:?}",
                instr
            )))
        }
    }
    push_word(word.0, code);
    Ok(())
}

fn reg_wr_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = RegWrOp::default();
    word.set_opcode(1);
    match instr {
        ResolvedInstruction::RegWr(addr, high, low, data) => {
            word.set_addr(addr.value());
            word.set_high(high.value());
            word.set_low(low.value());
            word.set_data(data.value());
        }
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an REG_WR operation {:?}",
                instr
            )))
        }
    }
    push_word(word.0, code);
    Ok(())
}

fn nop_operation(code: &mut Vec<u8>, _instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = HaltOp::default();
    word.set_opcode(4);
    push_word(word.0, code);
    Ok(())
}

fn long_pseudo_instr(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let word = match instr {
        ResolvedInstruction::Long(value) => value.value(),
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an LONG pseudo instruction {:?}",
                instr
            )))
        }
    };
    push_word(word, code);
    Ok(())
}

fn tsens_operation(code: &mut Vec<u8>, instr: &ResolvedInstruction) -> Result<(), CodeGenError> {
    let mut word = TsensOp::default();
    word.set_opcode(10);
    match instr {
        ResolvedInstruction::Tsens(dst, delay) => {
            word.set_delay(delay.value());
            word.set_dreg(*dst as u32);
        }
        _ => {
            return Err(CodeGenError::GenericError(format!(
                "Not an TSENS operation {:?}",
                instr
            )))
        }
    }
    push_word(word.0, code);
    Ok(())
}

fn push_word(word: u32, code: &mut Vec<u8>) {
    code.push((word & 0xff) as u8);
    code.push(((word & 0xff00) >> 8) as u8);
    code.push(((word & 0xff0000) >> 16) as u8);
    code.push(((word & 0xff000000) >> 24) as u8);
}

// esp32ulp-elf-as test.S -o test.out &&  esp32ulp-elf-objcopy  -O binary --only-section=.text  test.out test.x && hexdump -ve '1/1 "0x%.2x,"' test.x
// esp32ulp-elf-as test.S -al

#[test]
fn test_codegen1() {
    let src = "
        .set FOO, 0xff
        MOVE R3, FOO      // Some Comment
        MOVE R0, 1234
        
        label:
        MOVE R0, 5678
        STAGE_DEC 5
        STAGE_INC 9
        STAGE_RST

        ST R1, R2, 64
        LD R0, R3, 800

        JUMP label
        JUMP R0
        JUMP label, EQ
        JUMP R1, OV
        JUMP R2, EQ
        JUMP label, OV

        JUMPR label, 5, LT
        JUMPR label, 23, GE

        JUMPS label, 5, LT
        JUMPS label, 23, LE
        JUMPS label, 23, GE

        JUMPR forward, 5, LT
        JUMPR forward, 5, LT
        forward:";

    let expected = [
        0xf3, 0x0f, 0x80, 0x72, 0x20, 0x4d, 0x80, 0x72, 0xe0, 0x62, 0x81, 0x72, 0x50, 0x00, 0x20,
        0x74, 0x90, 0x00, 0x00, 0x74, 0x00, 0x00, 0x40, 0x74, 0x09, 0x40, 0x00, 0x68, 0x0c, 0x20,
        0x03, 0xd0, 0x08, 0x00, 0x00, 0x80, 0x00, 0x00, 0x20, 0x80, 0x08, 0x00, 0x40, 0x80, 0x01,
        0x00, 0xa0, 0x80, 0x02, 0x00, 0x60, 0x80, 0x08, 0x00, 0x80, 0x80, 0x05, 0x00, 0x18, 0x83,
        0x17, 0x00, 0x1b, 0x83, 0x05, 0x00, 0x1c, 0x85, 0x17, 0x00, 0x1f, 0x85, 0x17, 0x80, 0x20,
        0x85, 0x05, 0x00, 0x04, 0x82, 0x05, 0x00, 0x02, 0x82,
    ];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[test]
fn test_codegen2() {
    let src = "
    Halt
    Wake
    Sleep 0
    Wait 25
    ADC      R1, 0, 1 
    I2C_RD      0x10, 7, 0, 0 
    I2C_WR      0x20, 0x33, 7, 0, 1 
    REG_RD      0x120, 7, 4   
    REG_WR      0x120, 7, 0, 0x10 
";

    let expected = [
        0x00, 0x00, 0x00, 0xb0, 0x01, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x92, 0x19, 0x00, 0x00,
        0x40, 0x05, 0x00, 0x00, 0x50, 0x10, 0x00, 0x38, 0x30, 0x20, 0x33, 0x78, 0x38, 0x20, 0x01,
        0x90, 0x23, 0x20, 0x41, 0x80, 0x13,
    ];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[test]
fn test_codegen3() {
    let src = "
    TSENS R1, 1000
";

    let expected = [0xa1, 0x0f, 0x00, 0xa0];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[test]
fn test_codegen4() {
    let src = "
    NOP
";

    let expected = [0x00, 0x00, 0x00, 0x40];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[test]
fn test_codegen5() {
    let src = "
    .long 0
";

    let expected = [0x00, 0x00, 0x00, 0x00];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[test]
fn test_codegen6() {
    let src = "
    entry:
    NOP
    NOP
    NOP
    NOP
loop:
    MOVE R1, loop
    JUMP R1

.set        val, 0x10
MOVE        R1, val

array:  .long 0
    .long 0
    .long 0
    .long 0

    MOVE R1, array
    MOVE R2, 0x1234
    ST R2, R1, 0      // write value of R2 into the first array element,
                      // i.e. array[0]

    ST R2, R1, 4      // write value of R2 into the second array element
                      // (4 byte offset), i.e. array[1]

    ADD R1, R1, 2     // this increments address by 2 words (8 bytes)
    ST R2, R1, 0      // write value of R2 into the third array element,
                      // i.e. array[2]


    NOP
    ADD R1, R2, R3        //R1 = R2 + R3
    Add R1, R2, 0x1234    //R1 = R2 + 0x1234
    .set value1, 0x03     //constant value1=0x03
    Add R1, R2, value1    //R1 = R2 + value1

    SUB R1, R2, R3             //R1 = R2 - R3
    sub R1, R2, 0x1234         //R1 = R2 - 0x1234

    AND R1, R2, R3          //R1 = R2 & R3
    AND R1, R2, 0x1234      //R1 = R2 & 0x1234

    OR R1, R2, R3           //R1 = R2 | R3
    OR R1, R2, 0x1234       //R1 = R2 | 0x1234

    LSH R1, R2, R3            //R1 = R2 << R3
    LSH R1, R2, 0x03          //R1 = R2 << 0x03

    RSH R1, R2, R3              //R1 = R2 >> R3
    RSH R1, R2, 0x03            //R1 = R2 >> 0x03

    MOVE       R1, R2            //R1 = R2
    MOVE       R1, 0x03          //R1 = 0x03

    ST  R1, R2, 0x12        //MEM[R2+0x12] = R1
    LD  R1, R2, 0x12            //R1 = MEM[R2+0x12]

    JUMP       R1
    JUMP       0x120, EQ 
    JUMP       label 
    JUMP       loop 

label:
    STAGE_RST 
    STAGE_INC      10
    STAGE_DEC      10
    
    HALT 
    WAKE

    SLEEP     1

    WAIT     10

    TSENS     R1, 1000 

    ADC      R3, 0, 1

    I2C_RD      0x10, 7, 0, 0

    I2C_WR      0x20, 0x33, 7, 0, 1 

    REG_RD      0x120, 7, 4

    REG_WR      0x120, 7, 0, 0x10
";

    let expected = [
        0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
        0x40, 0x41, 0x00, 0x80, 0x72, 0x01, 0x00, 0x20, 0x80, 0x01, 0x01, 0x80, 0x72, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x71,
        0x00, 0x80, 0x72, 0x42, 0x23, 0x81, 0x72, 0x06, 0x00, 0x00, 0x68, 0x06, 0x04, 0x00, 0x68,
        0x25, 0x00, 0x00, 0x72, 0x06, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00, 0x40, 0x39, 0x00, 0x00,
        0x70, 0x49, 0x23, 0x01, 0x72, 0x39, 0x00, 0x00, 0x72, 0x39, 0x00, 0x20, 0x70, 0x49, 0x23,
        0x21, 0x72, 0x39, 0x00, 0x40, 0x70, 0x49, 0x23, 0x41, 0x72, 0x39, 0x00, 0x60, 0x70, 0x49,
        0x23, 0x61, 0x72, 0x39, 0x00, 0xa0, 0x70, 0x39, 0x00, 0xa0, 0x72, 0x39, 0x00, 0xc0, 0x70,
        0x39, 0x00, 0xc0, 0x72, 0x29, 0x00, 0x80, 0x70, 0x31, 0x00, 0x80, 0x72, 0x09, 0x10, 0x00,
        0x68, 0x09, 0x10, 0x00, 0xd0, 0x01, 0x00, 0x20, 0x80, 0x20, 0x01, 0x40, 0x80, 0x9c, 0x00,
        0x00, 0x80, 0x10, 0x00, 0x00, 0x80, 0x00, 0x00, 0x40, 0x74, 0xa0, 0x00, 0x00, 0x74, 0xa0,
        0x00, 0x20, 0x74, 0x00, 0x00, 0x00, 0xb0, 0x01, 0x00, 0x00, 0x90, 0x01, 0x00, 0x00, 0x92,
        0x0a, 0x00, 0x00, 0x40, 0xa1, 0x0f, 0x00, 0xa0, 0x07, 0x00, 0x00, 0x50, 0x10, 0x00, 0x38,
        0x30, 0x20, 0x33, 0x78, 0x38, 0x20, 0x01, 0x90, 0x23, 0x20, 0x41, 0x80, 0x13,
    ];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[test]
fn test_codegen7() {
    let src = "
    backward:
    JUMPR backward, 33, EQ
    JUMPR backward, 33, LT
    JUMPR backward, 33, LE
    JUMPR backward, 33, GT
    JUMPR backward, 33, GE

    JUMPR backward, 33, EQ
    JUMPR backward, 33, LT
    JUMPR backward, 33, LE
    JUMPR backward, 33, GT
    JUMPR backward, 33, GE

    JUMPR forward, 33, EQ
    JUMPR forward, 33, LT
    JUMPR forward, 33, LE
    JUMPR forward, 33, GT
    JUMPR forward, 33, GE

    JUMPR forward, 33, EQ
    JUMPR forward, 33, LT
    JUMPR forward, 33, LE
    JUMPR forward, 33, GT
    JUMPR forward, 33, GE

    JUMPS backward, 33, EQ
    JUMPS backward, 33, LT
    JUMPS backward, 33, LE
    JUMPS backward, 33, GT
    JUMPS backward, 33, GE

    JUMPS backward, 33, EQ
    JUMPS backward, 33, LT
    JUMPS backward, 33, LE
    JUMPS backward, 33, GT
    JUMPS backward, 33, GE

    JUMPS forward, 33, EQ
    JUMPS forward, 33, LT
    JUMPS forward, 33, LE
    JUMPS forward, 33, GT
    JUMPS forward, 33, GE

    JUMPS forward, 33, EQ
    JUMPS forward, 33, LT
    JUMPS forward, 33, LE
    JUMPS forward, 33, GT
    JUMPS forward, 33, GE

    forward:

";

    let expected = [
        0x22, 0x00, 0x05, 0x82, 0x21, 0x00, 0x03, 0x83, 0x21, 0x00, 0x04, 0x83, 0x22, 0x00, 0x06,
        0x83, 0x22, 0x00, 0x09, 0x83, 0x21, 0x00, 0x0b, 0x83, 0x22, 0x00, 0x05, 0x82, 0x21, 0x00,
        0x0f, 0x83, 0x21, 0x00, 0x10, 0x83, 0x22, 0x00, 0x12, 0x83, 0x22, 0x00, 0x15, 0x83, 0x21,
        0x00, 0x17, 0x83, 0x22, 0x00, 0x05, 0x82, 0x21, 0x00, 0x4f, 0x82, 0x21, 0x00, 0x4c, 0x82,
        0x22, 0x00, 0x4a, 0x82, 0x22, 0x00, 0x49, 0x82, 0x21, 0x00, 0x47, 0x82, 0x22, 0x00, 0x05,
        0x82, 0x21, 0x00, 0x43, 0x82, 0x21, 0x00, 0x40, 0x82, 0x22, 0x00, 0x3e, 0x82, 0x22, 0x00,
        0x3d, 0x82, 0x21, 0x00, 0x3b, 0x82, 0x21, 0x00, 0x04, 0x84, 0x21, 0x00, 0x33, 0x85, 0x21,
        0x00, 0x34, 0x85, 0x21, 0x00, 0x37, 0x85, 0x21, 0x00, 0x05, 0x84, 0x21, 0x80, 0x3a, 0x85,
        0x21, 0x80, 0x3c, 0x85, 0x21, 0x00, 0x04, 0x84, 0x21, 0x00, 0x41, 0x85, 0x21, 0x00, 0x42,
        0x85, 0x21, 0x00, 0x45, 0x85, 0x21, 0x00, 0x05, 0x84, 0x21, 0x80, 0x48, 0x85, 0x21, 0x80,
        0x4a, 0x85, 0x21, 0x00, 0x04, 0x84, 0x21, 0x00, 0x1b, 0x84, 0x21, 0x00, 0x18, 0x84, 0x21,
        0x00, 0x17, 0x84, 0x21, 0x00, 0x05, 0x84, 0x21, 0x80, 0x12, 0x84, 0x21, 0x80, 0x10, 0x84,
        0x21, 0x00, 0x04, 0x84, 0x21, 0x00, 0x0d, 0x84, 0x21, 0x00, 0x0a, 0x84, 0x21, 0x00, 0x09,
        0x84, 0x21, 0x00, 0x05, 0x84, 0x21, 0x80, 0x04, 0x84, 0x21, 0x80, 0x02, 0x84,
    ];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[test]
fn test_codegen8() {
    let src = "
    MOVE R1, 0xffff
    MOVE R0, data
    ST R1, R0, 0
    HALT

    data:
    .long 0";

    let expected = [
        0xf1, 0xff, 0x8f, 0x72, 0x40, 0x00, 0x80, 0x72, 0x01, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
        0xb0, 0x00, 0x00, 0x00, 0x00,
    ];

    let ast = crate::parser::parse(src).unwrap();
    let code = create_codegen().generate(ast);

    assert_byte_array_eq(code, &expected);
}

#[cfg(test)]
fn assert_byte_array_eq(actual: Result<(Vec<u8>, Vec<Label>), CodeGenError>, expected: &[u8]) {
    assert!(actual.is_ok(), "Codegen failed");
    let actual = actual.unwrap().0;

    assert!(
        actual.len() == expected.len(),
        "Array size differs, actual {}, expected {}",
        actual.len(),
        expected.len()
    );

    for i in 0..actual.len() {
        if actual[i] != expected[i] {
            let word_index = i / 4;
            let wanted_word = expected[4 * word_index] as u32
                | (expected[4 * word_index + 1] as u32) << 8
                | (expected[4 * word_index + 2] as u32) << 16
                | (expected[4 * word_index + 3] as u32) << 24;

            let actual_word = actual[4 * word_index] as u32
                | (actual[4 * word_index + 1] as u32) << 8
                | (actual[4 * word_index + 2] as u32) << 16
                | (actual[4 * word_index + 3] as u32) << 24;

            assert!(
                false,
                "Difference at index {}, word {}, wanted 0x{:02x}, given 0x{:02x}\nActual: {:032b}\nWanted: {:032b}\nat 0x{:04x}",
                i,
                i / 4,
                expected[i],
                actual[i],
                actual_word,
                wanted_word,
                word_index * 4,
            );
        }
    }
}
