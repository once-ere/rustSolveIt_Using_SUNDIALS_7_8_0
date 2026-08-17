//! `posim` — the physical_object simulator front end.
//!
//! Modes:
//! - no arguments: interactive notebook REPL (`In[n]`/`Out[n]` cells)
//! - `--script <file>`: batch-execute a command script
//! - `--notebook <file>`: dynamic notebook — execute the file (which
//!   opens the scene window via `SCENE CREATE`), then stay in the
//!   interactive REPL with the GUI alive and ready to Start
//! - `--machine`: line-delimited JSON protocol for front ends
//!   (e.g. the JupyterLab wrapper kernel in `jupyter/`)
//! - `--help`
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

mod lexer;
mod machine;
mod notebook;
mod parser;
mod qm;
mod qm2;
mod qm3;
mod scene;
mod special;
mod vm;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => notebook::repl(),
        Some("--machine") => machine::serve(),
        Some("--script") => match args.get(1) {
            Some(path) => {
                if let Err(e) = notebook::run_script(path) {
                    eprintln!("posim: {e}");
                    std::process::exit(1);
                }
            }
            None => {
                eprintln!("posim: --script needs a file path");
                std::process::exit(2);
            }
        },
        Some("--notebook") => match args.get(1) {
            Some(path) => {
                if let Err(e) = notebook::run_notebook(path) {
                    eprintln!("posim: {e}");
                    std::process::exit(1);
                }
            }
            None => {
                eprintln!("posim: --notebook needs a file path");
                std::process::exit(2);
            }
        },
        Some("--help" | "-h") => {
            println!("posim — physical_object simulator (sundials_rs numerical backend)");
            println!();
            println!("usage: posim [--script <file> | --notebook <file> | --machine | --help]");
            println!();
            println!("{}", vm::HELP_TEXT);
        }
        Some(other) => {
            eprintln!("posim: unknown argument `{other}` (try --help)");
            std::process::exit(2);
        }
    }
}
