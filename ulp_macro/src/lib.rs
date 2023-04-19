#![feature(proc_macro_span)]

use litrs::StringLit;
use proc_macro::{self, TokenStream};
use quote::format_ident;
use quote::quote;

mod parser;
use parser::parse;
use quote::quote_spanned;

mod codegen;
use crate::codegen::create_codegen;
use crate::codegen::CodeGen;

#[proc_macro]
pub fn ulp_asm(input: TokenStream) -> TokenStream {
    let first_token = input.into_iter().next().expect("no input");
    let arg = StringLit::try_from(&first_token).expect("Expecting a string argument");
    let src = arg.value();

    let ast = parse(src);
    match ast {
        Ok(ast) => {
            let code = create_codegen().generate(ast);

            match code {
                Ok((code, labels)) => {
                    let code_len = code.len();

                    let mut accessors = Vec::new();
                    for lbl in labels {
                        let getter_name = format_ident!("get_{}", lbl.name);
                        let setter_name = format_ident!("set_{}", lbl.name);
                        let address = lbl.address;

                        accessors.push(quote! {
                            fn #getter_name(&self) -> u16 {
                                unsafe {(((#address + 0x5000_0000)  as *const u32).read_volatile() & 0xffff) as u16}
                            }

                            fn #setter_name(&self, value: u16) {
                                unsafe {((#address + 0x5000_0000) as *mut u32).write_volatile(value as u32)}
                            }

                        });
                    }

                    let tokens = quote! {
                        {
                            struct _Ulp {
                                code: [u8; #code_len],
                            }

                            impl _Ulp {
                                #(#accessors)*

                                fn load(&self) {
                                    unsafe {
                                        core::ptr::copy_nonoverlapping(
                                            self.code.as_ptr() as *const u8,
                                            0x5000_0000 as *mut u8,
                                            #code_len as usize
                                        );
                                    }
                                }
                            }

                            _Ulp {
                                code: [ #(#code),*  ]
                            }
                        }
                    };

                    tokens.into()
                }
                Err(error) => {
                    let error_msg = format!("{:?}", error);
                    let span = first_token.span().into();
                    let tokens = quote_spanned! {span=> compile_error!(#error_msg)};
                    tokens.into()
                }
            }
        }
        Err(error) => {
            let error_msg = error.to_string();
            let span = match first_token {
                proc_macro::TokenTree::Group(_) => todo!(),
                proc_macro::TokenTree::Ident(_) => todo!(),
                proc_macro::TokenTree::Punct(_) => todo!(),
                proc_macro::TokenTree::Literal(ref lit) => lit
                    .subspan(error.location.offset..(error.location.offset + 1))
                    .unwrap_or_else(|| first_token.span()),
            }
            .into();

            let tokens = quote_spanned! {span=> compile_error!(#error_msg)};
            tokens.into()
        }
    }
}
