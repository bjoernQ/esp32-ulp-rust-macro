#![no_std]
#![no_main]

extern crate ulp_macro;
use esp32_hal as hal;
use esp_println::println;
use hal::{clock::ClockControl, peripherals::Peripherals, prelude::*, Delay};
use panic_halt as _;
use ulp_macro::ulp_asm;

#[entry]
fn main() -> ! {
    let peripherals = Peripherals::take();
    let system = peripherals.DPORT.split();
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();

    let ulp_code = ulp_asm!(
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

    unsafe {
        // ideally this should be in the HAL
        let sens = &*esp32::SENS::PTR;
        let rtc = &*esp32::RTC_CNTL::PTR;

        rtc.state0.write(|w| w.ulp_cp_slp_timer_en().clear_bit());
        sens.sar_start_force.write(|w| w.pc_init().bits(0));
        sens.sar_start_force
            .write(|w| w.ulp_cp_force_start_top().clear_bit());
        rtc.timer5.write(|w| w.min_slp_val().bits(2));
    }

    // load code to RTC ram
    ulp_code.load();

    println!("Hello world! Hello ULP!");

    // getter and setters are named as the label with a prefix
    println!("data is {}", ulp_code.get_data());
    unsafe {
        // ideally this should be in the HAL
        let rtc = &*esp32::RTC_CNTL::PTR;
        rtc.state0.write(|w| w.ulp_cp_slp_timer_en().set_bit());
    }

    let mut delay = Delay::new(&clocks);
    loop {
        delay.delay_ms(500u32);
        println!("data is {}", ulp_code.get_data());
    }
}
