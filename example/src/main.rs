#![no_std]
#![no_main]

extern crate ulp_macro;
use core::fmt::Write;
use esp32_hal as hal;
use hal::{pac, Serial};
use panic_halt as _;
use ulp_macro::ulp_asm;
use xtensa_lx_rt::entry;

#[entry]
fn main() -> ! {
    // this generates a [u8] named ulp_code
    // and for every label a variable name "ulp_label_" and the label name
    // containing the offset in bytes
    ulp_asm!(
        "
        MOVE R0, data
        entry:
        LD R1, R0, 0
        ADD R1, R1, 1
        ST R1, R0, 0
        WAIT 0xffff
        JUMP entry

        data:
        .long 0
    "
    );

    let peripherals = pac::Peripherals::take().unwrap();

    let sens = peripherals.SENS;
    let rtc = peripherals.RTC_CNTL;

    rtc.state0.write(|w| w.ulp_cp_slp_timer_en().clear_bit());
    for _ in 0..100_000 {}
    sens.sar_start_force
        .write(|w| unsafe { w.pc_init().bits(0) });
    sens.sar_start_force
        .write(|w| w.ulp_cp_force_start_top().clear_bit());
    rtc.timer5.write(|w| unsafe { w.min_slp_val().bits(2) });

    let rtc_slow_mem = 0x5000_0000 as *mut u8;

    for i in 0..ulp_code.len() {
        unsafe {
            rtc_slow_mem.offset(i as isize).write_volatile(ulp_code[i]);
        }
    }

    let mut serial0 = Serial::new(peripherals.UART0);
    writeln!(serial0, "Hello world! Hello ULP!").unwrap();

    let data_ptr = unsafe { rtc_slow_mem.offset(ulp_label_data) as *mut u32 };

    writeln!(serial0, "data is {}", unsafe {
        data_ptr.read_volatile() & 0xffff
    })
    .unwrap();
    rtc.state0.write(|w| w.ulp_cp_slp_timer_en().set_bit());

    loop {
        for _ in 0..20_000 {}
        writeln!(serial0, "data is {}", unsafe {
            data_ptr.read_volatile() & 0xffff
        })
        .unwrap();
    }
}
