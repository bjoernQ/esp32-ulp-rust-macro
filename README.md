# ESP32 ULP Rust Macro

This is a Rust macro which enables writing ULP assembly in right in your Rust source files.

Example

```Rust
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
```

Have a look at the `example` folder.

There are currently still a lot of things that needs improvement
- [ ] proper error handling and reporting
- [ ] support ESP32S2
