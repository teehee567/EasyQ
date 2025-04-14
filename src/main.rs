mod app;
mod audio;
mod config;
mod ui;
mod effects;

use eframe::{NativeOptions, egui};
use app::SystemWideEQ;

fn main() {
    // Initialize native options with the appropriate fields
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "System-Wide Equalizer",
        options,
        Box::new(|cc| Ok(Box::new(SystemWideEQ::new(cc)))),
    )
    .expect("Failed to start application");
}
