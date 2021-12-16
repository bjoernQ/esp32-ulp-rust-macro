#[cfg(test)]
use ariadne::{Label, Report, ReportKind, Source};

use peg::str::LineCol;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Reg {
    R0 = 0,
    R1 = 1,
    R2 = 2,
    R3 = 3,
}

#[derive(Debug, Clone)]
pub(crate) enum Value {
    Number(u32),
    Identifier(String),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum Condition {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
    Ov,
}

#[derive(Debug, Clone)]
pub(crate) enum Instruction {
    Move(Reg, Reg),
    MoveImmediate(Reg, Value),
    Store(Reg, Reg, Value),
    Load(Reg, Reg, Value),
    Add(Reg, Reg, Reg),
    AddImmediate(Reg, Reg, Value),
    Nop,
    Sub(Reg, Reg, Reg),
    SubImmediate(Reg, Reg, Value),
    And(Reg, Reg, Reg),
    AndImmediate(Reg, Reg, Value),
    Or(Reg, Reg, Reg),
    OrImmediate(Reg, Reg, Value),
    Lsh(Reg, Reg, Reg),
    LshImmediate(Reg, Reg, Value),
    Rsh(Reg, Reg, Reg),
    RshImmediate(Reg, Reg, Value),
    Jump(Reg),
    JumpImmediate(Value),
    JumpConditional(Reg, Condition),
    JumpConditionalImmediate(Value, Condition),
    JumpR(Value, Value, Condition),
    JumpS(Value, Value, Condition),
    StageRst,
    StageInc(Value),
    StageDec(Value),
    Halt,
    Wake,
    Sleep(Value),
    Wait(Value),
    Tsens(Reg, Value),
    Adc(Reg, Value, Value),
    I2cRd(Value, Value, Value, Value),
    I2cWr(Value, Value, Value, Value, Value),
    RegRd(Value, Value, Value),
    RegWr(Value, Value, Value, Value),
    Comment,
    Label(String),
    Set(String, Value),
    Long(Value),
    Global(String),
}

peg::parser! {
    grammar pparser() for str {
        rule i(literal: &'static str)
            = input:$([_]*<{literal.len()}>)
            {? if input.eq_ignore_ascii_case(literal) { Ok(()) } else { Err(literal) } }

        rule keyword_chars()  = ['A'..='Z'|'0'..='9']

        rule ws()  = [' '|'\r'|'\n']

        rule space()  = [' '| '\t']

        rule newline()  = ['\r'|'\n']

        rule eof() = ![_]

        rule any() = [' '] / ['\t'] / ['a'..='z'] / ['A'..='Z'] / ['0'..='9'] / ['.'] / ['ö'] / ['ä'] / [',']
            / ['ü'] / ['Ö'] / ['Ä'] / ['Ü'] / ['-'] / ['_'] / ['('] / [')'] / [','] / ['['] / [']'] / ['+']  / ['=']
            / ['&'] / ['|'] / ['<'] / ['>']

        rule identifier_char() = ['a'..='z'] / ['A'..='Z'] / ['0'..='9'] / ['_']

        rule register() -> Reg
            = s:$(i("R0") / i("R1") / i("R2") / i("R3")) {
                match s.to_uppercase().as_str() {
                    "R0" => Reg::R0,
                    "R1" => Reg::R1,
                    "R2" => Reg::R2,
                    "R3" => Reg::R3,
                    _ => panic!()
                }
            }

        rule condition() -> Condition
            = s:$(i("EQ") / i("LT") / i("LE") / i("GT") / i("GE") / i("OV")) {
                match s.to_uppercase().as_str() {
                    "EQ" => Condition::Eq,
                    "LT" => Condition::Lt,
                    "LE" => Condition::Le,
                    "GT" => Condition::Gt,
                    "GE" => Condition::Ge,
                    "OV" => Condition::Ov,
                    _ => panic!()
                }
            }

        rule number_hex() -> Value = "0x" s:$(['0'..='9'|'a'..='f'|'A'..='F']+) {
            Value::Number(u32::from_str_radix(s, 16).unwrap())
        }

        rule number_bin() -> Value = "0b" s:$(['0'..='1']+) {
            Value::Number(u32::from_str_radix(s, 2).unwrap())
        }

        rule number() -> Value = s:$(['0'..='9']+) {
            Value::Number(u32::from_str_radix(s, 10).unwrap())
        }

        rule identifier() -> Value
            = s:$(identifier_char()*) { Value::Identifier(s.to_string()) }

        rule number_or_symbol() -> Value
            = v:(number_hex() / number_bin() / number() / identifier()) {
                v
            }

        rule move_instr() -> Instruction
            = i("MOVE")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())* {
                Instruction::Move(r0, r1)
            }

        rule move_immediate_instr() -> Instruction
            = i("MOVE")(space())+r0:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::MoveImmediate(r0, l)
            }

        rule store_instr() -> Instruction
            = i("ST")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::Store(r0, r1, l)
            }

        rule load_instr() -> Instruction
            = i("LD")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::Load(r0, r1, l)
            }

        rule add_immediate_instr() -> Instruction
            = i("ADD")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::AddImmediate(r0, r1, l)
            }

        rule add_instr() -> Instruction
            = i("ADD")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*r3:(register())(space())* {
                Instruction::Add(r0, r1, r3)
            }

        rule nop_instr() -> Instruction
            = i("NOP")(space())* {
                Instruction::Nop
            }

        rule sub_immediate_instr() -> Instruction
            = i("SUB")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::SubImmediate(r0, r1, l)
            }

        rule sub_instr() -> Instruction
            = i("SUB")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*r3:(register())(space())* {
                Instruction::Sub(r0, r1, r3)
            }

        rule and_immediate_instr() -> Instruction
            = i("AND")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::AndImmediate(r0, r1, l)
            }

        rule and_instr() -> Instruction
            = i("AND")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*r3:(register())(space())* {
                Instruction::And(r0, r1, r3)
            }

        rule or_immediate_instr() -> Instruction
            = i("OR")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::OrImmediate(r0, r1, l)
            }

        rule or_instr() -> Instruction
            = i("OR")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*r3:(register())(space())* {
                Instruction::Or(r0, r1, r3)
            }

        rule lsh_immediate_instr() -> Instruction
            = i("LSH")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::LshImmediate(r0, r1, l)
            }

        rule lsh_instr() -> Instruction
            = i("LSH")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*r3:(register())(space())* {
                Instruction::Lsh(r0, r1, r3)
            }

        rule rsh_immediate_instr() -> Instruction
            = i("RSH")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::RshImmediate(r0, r1, l)
            }

        rule rsh_instr() -> Instruction
            = i("RSH")(space())+r0:(register())(space())*[','](space())*r1:(register())(space())*[','](space())*r3:(register())(space())* {
                Instruction::Rsh(r0, r1, r3)
            }

        rule jump_instr() -> Instruction
            = i("JUMP")(space())+r0:(register())(space())* {
                Instruction::Jump(r0)
            }

        rule jump_immediate_instr() -> Instruction
            = i("JUMP")(space())+v:(number_or_symbol())(space())* {
                Instruction::JumpImmediate(v)
            }

        rule jump_conditional_instr() -> Instruction
            = i("JUMP")(space())+r0:(register())(space())*[','](space())*c:(condition())(space())* {
                Instruction::JumpConditional(r0, c)
            }

        rule jump_conditional_immediate_instr() -> Instruction
            = i("JUMP")(space())+v:(number_or_symbol())(space())*[','](space())*c:(condition())(space())* {
                Instruction::JumpConditionalImmediate(v, c)
            }

        rule jumpr_conditional_instr() -> Instruction
            = i("JUMPR")(space())+step:(number_or_symbol())(space())*[','](space())thr:(number_or_symbol())(space())*[','](space())*c:(condition())(space())* {
                Instruction::JumpR(step, thr, c)
            }

        rule jumps_conditional_instr() -> Instruction
            = i("JUMPS")(space())+step:(number_or_symbol())(space())*[','](space())thr:(number_or_symbol())(space())*[','](space())*c:(condition())(space())* {
                Instruction::JumpS(step, thr, c)
            }

        rule stage_rst_instr() -> Instruction
            = i("STAGE_RST")(space())* {
                Instruction::StageRst
            }

        rule stage_inc_instr() -> Instruction
            = i("STAGE_INC")(space())+value:(number_or_symbol())(space())* {
                Instruction::StageInc(value)
            }

        rule stage_dec_instr() -> Instruction
            = i("STAGE_DEC")(space())+value:(number_or_symbol())(space())* {
                Instruction::StageDec(value)
            }

        rule halt_instr() -> Instruction
            = i("HALT")(space())* {
                Instruction::Halt
            }

        rule wake_instr() -> Instruction
            = i("WAKE")(space())* {
                Instruction::Wake
            }

        rule sleep_instr() -> Instruction
            = i("SLEEP")(space())+v:(number_or_symbol())(space())* {
                Instruction::Sleep(v)
            }

        rule wait_instr() -> Instruction
            = i("WAIT")(space())+v:(number_or_symbol())(space())* {
                Instruction::Wait(v)
            }

        rule tsens_instr() -> Instruction
            = i("TSENS")(space())+r0:(register())(space())*[','](space())*l:(number_or_symbol())(space())* {
                Instruction::Tsens(r0, l)
            }

        rule adc_instr() -> Instruction
            = i("ADC")(space())+r0:(register())(space())*[','](space())*l0:(number_or_symbol())(space())*[','](space())*l1:(number_or_symbol())(space())* {
                Instruction::Adc(r0, l0, l1)
            }

        rule i2c_rd_instr() -> Instruction
            = i("I2C_RD")(space())+l0:(number_or_symbol())(space())*[','](space())*l1:(number_or_symbol())(space())*[','](space())*l2:(number_or_symbol())(space())*[','](space())*l3:(number_or_symbol())(space())* {
                Instruction::I2cRd(l0, l1, l2, l3)
            }

        rule i2c_wr_instr() -> Instruction
            = i("I2C_WR")(space())+l0:(number_or_symbol())(space())*[','](space())*l1:(number_or_symbol())(space())*[','](space())*l2:(number_or_symbol())(space())*[','](space())*l3:(number_or_symbol())(space())*[','](space())*l4:(number_or_symbol())(space())* {
                Instruction::I2cWr(l0, l1, l2, l3, l4)
            }

        rule reg_rd_instr() -> Instruction
            = i("REG_RD")(space())+l0:(number_or_symbol())(space())*[','](space())*l1:(number_or_symbol())(space())*[','](space())*l2:(number_or_symbol())(space())* {
                Instruction::RegRd(l0, l1, l2)
            }

        rule reg_wr_instr() -> Instruction
            = i("REG_WR")(space())+l0:(number_or_symbol())(space())*[','](space())*l1:(number_or_symbol())(space())*[','](space())*l2:(number_or_symbol())(space())*[','](space())*l3:(number_or_symbol())(space()*) {
                Instruction::RegWr(l0, l1, l2, l3)
            }

        rule comment() -> Instruction = "//" any()* { Instruction::Comment }

        rule block_comment() -> Instruction = "/*" (any() / newline())* "*/" { Instruction::Comment }

        rule label() -> Instruction
            = s:$(identifier_char()*) ":"(space())* {
                Instruction::Label(s[..s.len()].to_string())
            }

        rule set_pseudo_instr() -> Instruction
            = i(".set")(space())+label:(identifier())(space())*[','](space())*l:(number_or_symbol())(space())* {
                if let Value::Identifier(ident) = label { Instruction::Set(ident, l) } else { panic!() }
            }

        rule long_pseudo_instr() -> Instruction
            = i(".long")(space())+l:(number_or_symbol())(space())* {
                Instruction::Long(l)
            }

        rule global_pseudo_instr() -> Instruction
            = i(".global")(space())+label:(identifier())(space())* {
                if let Value::Identifier(ident) = label { Instruction::Global(ident) } else { panic!() }
            }

        rule instr() -> Instruction
            = ((space() / newline())*) c:(
                move_instr() /
                move_immediate_instr() /
                store_instr() /
                load_instr() /
                add_instr() /
                add_immediate_instr() /
                nop_instr() /
                sub_instr() /
                sub_immediate_instr() /
                and_instr() /
                and_immediate_instr() /
                or_instr() /
                or_immediate_instr() /
                lsh_instr() /
                lsh_immediate_instr() /
                rsh_instr() /
                rsh_immediate_instr() /
                jump_conditional_instr() /
                jump_conditional_immediate_instr() /
                jump_instr() /
                jump_immediate_instr() /
                jumpr_conditional_instr() /
                jumps_conditional_instr() /
                stage_rst_instr() /
                stage_inc_instr() /
                stage_dec_instr() /
                halt_instr() /
                wake_instr() /
                sleep_instr() /
                wait_instr() /
                tsens_instr() /
                adc_instr() /
                i2c_rd_instr() /
                i2c_wr_instr() /
                reg_rd_instr() /
                reg_wr_instr() /
                comment() /
                block_comment() /
                label() /
                set_pseudo_instr() /
                long_pseudo_instr() /
                global_pseudo_instr()
            ) ((space() / newline())*) { c }

        pub(crate) rule parse() -> Vec<Instruction>
            = ((newline()*) (ws()*)) c:(instr()) ** (newline()*)  {
                c
            }
    }
}

pub(crate) fn parse(src: &str) -> Result<Vec<Instruction>, peg::error::ParseError<LineCol>> {
    pparser::parse(src)
}

#[cfg(test)]
pub(crate) fn print_error(src: &str, error: &peg::error::ParseError<LineCol>) {
    Report::build(ReportKind::Error, (), 34)
        .with_message(error.to_string())
        .with_label(
            Label::new(error.location.offset..(error.location.offset + 1))
                .with_message("Error here"),
        )
        .finish()
        .print(Source::from(src))
        .unwrap();
}

#[cfg(test)]
fn parse_and_print_error(src: &str) -> Result<Vec<Instruction>, peg::error::ParseError<LineCol>> {
    let res = parse(src);

    match &res {
        Ok(_) => (),
        Err(err) => print_error(src, &err),
    }

    println!("{:?}", res);
    res
}

#[test]
fn test_simple() {
    let res = parse_and_print_error("\r\n MOVE R0, R1\r\n   ADD  R3, R3, 1     //Zeiger auf nächsten 32-Bit-Speicherplatz\n MOVE R1,R2  ");
    assert!(res.is_ok())
}

#[test]
fn test_simple2() {
    let res = parse_and_print_error(
        "ST   R0, R3, 0 //2.Element über Offset adressieren\nADD  R3, R3, 1  ",
    );
    assert!(res.is_ok())
}

#[test]
fn test_simple2_identifier() {
    let res =
        parse("ST   R0, R3, myvalue //2.Element über Offset adressieren\nADD  R3, R3, VALUE  ");
    assert!(res.is_ok())
}

#[test]
fn test_numbers() {
    let res = parse_and_print_error(
        "ST   R0, R3, 0 \n   ST   R0, R3, 0xabcd \n ST   R0, R3, 0b1010111011 \n \n",
    );
    assert!(res.is_ok())
}

#[test]
fn test_comments() {
    let res = parse_and_print_error("move R0, R1 // test\n\nMove R2,R3\n");
    assert!(res.is_ok())
}

#[test]
fn test2() {
    let res = parse_and_print_error(
        "
        label:
        MOVE R3, 555      // Some Comment
        MOVE R0, 1234
        ST   R0, R3, 0
        
        MOVE R0, 5678

        ST   R0, R3, 4    //2.Element über Offset adressieren
        //oder       
        ADD  R3, R3, 1	 //Zeiger auf nächsten 32-Bit-Speicherplatz
        ST   R0, R3, 0    //2.Element mit Nulloffset adressieren
        ST   R0, R3, 0    //2.Element mit Nulloffset adressieren
        JUMP label
        ",
    );

    assert!(res.is_ok())
}

#[test]
fn parse_complex_thing() {
    let res = parse_and_print_error(
        "is_rdy_for_wakeup:                   // Read RTC_CNTL_RDY_FOR_WAKEUP bit
    AND r0, r0, 1
    JUMP is_rdy_for_wakeup, eq    // Retry until the bit is set
    WAKE                          // Trigger wake up
    REG_WR 0x006, 24, 24, 0       // Stop ULP timer (clear RTC_CNTL_ULP_CP_SLP_TIMER_EN)
    HALT                          // Stop the ULP program
    // After these instructions, SoC will wake up,
    // and ULP will not run again until started by the main program.",
    );

    assert!(res.is_ok())
}

#[test]
fn parse_others() {
    let res = parse_and_print_error(
        "is_rdy_for_wakeup:                   // Read RTC_CNTL_RDY_FOR_WAKEUP bit
    .set foo, 0x234
    .set foo, 234
    .set foo2, foo
    .global my_label
    // After these instructions, SoC will wake up,
    // and ULP will not run again until started by the main program.
    
    JUMPR       label, 1, GE
    JUMPS       label, 1, GE

some_more_stuff:
    move r1, r0
    jump p1_status_changed, eq

	move r3, p1_status
    rsh r0, r1, 7
	and r0, r0, 1
	st r0, r3, 0

	move r3, p2_status
    rsh r0, r1, 8
	and r0, r0, 1
	st r0, r3, 0

	move r3, p3_status
    rsh r0, r1, 9
	and r0, r0, 1
	st r0, r3, 0
	// check if p1 status changed
    rsh r0, r1, 7
	and r0, r0, 1
	move r3, p1_status_next
	ld r3, r3, 0
	add r3, r0, r3
    and r3, r3, 1
	jump p1_status_changed, eq
	// check if p2 status changed
    rsh r0, r1, 8
	and r0, r0, 1
	move r3, p2_status_next
	ld r3, r3, 0
	add r3, r0, r3
    and r3, r3, 1
	jump p2_status_changed, eq

	/*  check if p3 status changed */
    rsh r0, r1, 9
	and r0, r0, 1
	move r3, p3_status_next
	ld r3, r3, 0
	add r3, r0, r3
    and r3, r3, 1
	jump p3_status_changed, eq

/*
multi line comments
*/

    halt

     ADC      R1, 0, 1 
     TSENS     R1, 1000 

     REG_RD      0x120, 7, 4  
     REG_WR      0x120, 7, 0, 0x10 
     I2C_WR      0x20, 0x33, 7, 0, 1
     I2C_RD      0x10, 7, 0, 0    

     WAIT     10 
     SLEEP     1  
    
     label:
     .long 0
    ",
    );

    assert!(res.is_ok())
}
