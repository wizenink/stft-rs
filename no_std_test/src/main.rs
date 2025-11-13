#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use core::panic::PanicInfo;
use num_traits::Float;
use stft_rs::prelude::*;

use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

// We need a memory block.
// 'static mut' is generally unsafe, but we only access it once during init.
static mut HEAP_MEM: [u8; 1024 * 4096] = [0; 1024 * 4096];

// Panic handler (required for no_std)
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    let mut i = 0;
    while i < n {
        *dest.add(i) = *src.add(i);
        i += 1;
    }
    dest
}

#[no_mangle]
pub unsafe extern "C" fn memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    if src < dest as *const u8 {
        // Copy backwards to handle overlap
        let mut i = n;
        while i > 0 {
            i -= 1;
            *dest.add(i) = *src.add(i);
        }
    } else {
        // Copy forwards
        let mut i = 0;
        while i < n {
            *dest.add(i) = *src.add(i);
            i += 1;
        }
    }
    dest
}

#[no_mangle]
pub unsafe extern "C" fn memset(dest: *mut u8, c: i32, n: usize) -> *mut u8 {
    let mut i = 0;
    while i < n {
        *dest.add(i) = c as u8;
        i += 1;
    }
    dest
}

#[no_mangle]
pub unsafe extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
    let mut i = 0;
    while i < n {
        let a = *s1.add(i);
        let b = *s2.add(i);
        if a != b {
            return a as i32 - b as i32;
        }
        i += 1;
    }
    0
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    unsafe {
        ALLOCATOR.lock().init(HEAP_MEM.as_mut_ptr(), 1024 * 4096);
    }

    test_stft_no_std();

    loop {}
    // Exit (in a real embedded system, this would loop forever)
}

#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

fn test_stft_no_std() {
    // Create a simple test signal
    let mut signal = Vec::new();
    for i in 0..4096 {
        let t = i as f32 / 44100.0;
        let sample = (2.0 * 3.14159 * 440.0 * t).sin();
        signal.push(sample);
    }

    let config = StftConfigF32::default_4096();

    let stft = BatchStftF32::new(config.clone());
    let spectrum = stft.process(&signal);

    let istft = BatchIstftF32::new(config);
    let _reconstructed = istft.process(&spectrum);

    // If we get here without panicking, it works!
    // In a real system, you could output this via UART, etc.
}
