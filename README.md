# ESP32 FSM-ULP Rust Macro

This is a Rust macro which enables writing ULP assembly in right in your Rust source files.

Example

```Rust
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
```

Have a look at the `example` folder.
