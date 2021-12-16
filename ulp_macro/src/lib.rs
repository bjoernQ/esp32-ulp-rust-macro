#![feature(proc_macro_span)]

use litrs::StringLit;
use proc_macro::{self, TokenStream};
use proc_macro2::Ident;
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
                    let label_names: Vec<Ident> = labels
                        .iter()
                        .map(|label| format_ident!("ulp_label_{}", label.name))
                        .collect();
                    let label_values: Vec<u32> = labels.iter().map(|label| label.address).collect();

                    let tokens = quote! {
                       let ulp_code = [ #(#code),*  ];

                       #(let #label_names = #label_values as isize;)*
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
